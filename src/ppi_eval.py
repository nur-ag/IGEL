import os
import json

import dill

import numpy as np
import igraph as ig

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support

from samplers import *
from embedders import *
from aggregators import *
from learning_utils import EarlyStopping, uniformly_random_samples, random_bfs_samples


DATASET_PATH = '{}/../PPI'.format(os.path.dirname(os.path.realpath(__file__)))
STRUCTURAL_VECTOR_LENGTH = 30
NUM_ATTENTION_HEADS = 8
NUM_HIDDEN = 1
HIDDEN_UNITS = 200
NUM_OUTPUTS = 80
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 200
DISPLAY_EPOCH = 3
NODES_TO_SAMPLE = 0
SAMPLING_MODEL = None # log_degree_sampler
INCLUDE_NODE = True
MODEL_DEPTH = 4
AGGREGATION_FN = combine_mean

TRAINING_PATIENCE = 20
TRAINING_MINIMUM_CHANGE = 0
CHECKPOINT_PATH = '{}/checkpoint-depth-{}.pt'.format(DATASET_PATH, MODEL_DEPTH)

USE_CUDA = True 
MOVE_TO_CUDA = USE_CUDA and torch.cuda.is_available()
device = torch.device('cuda') if MOVE_TO_CUDA else torch.device('cpu')

PRECOMPUTE_ATTRIBUTES = False

# Load the Graphs
all_splits = ['train', 'valid', 'test']
splits = {}
for split in all_splits:
    split_path = '{}/ppi-{}.edgelist'.format(DATASET_PATH, split)
    G = ig.Graph.Read_Ncol(split_path, names=True, weights=False, directed=False)
    G.vs['id'] = [int(n) for n in G.vs['name']]
    if PRECOMPUTE_ATTRIBUTES:
        G.vs['pagerank'] = G.pagerank()
        G.vs['betweenness'] = G.betweenness()
        G.vs['degree'] = G.degree()
        G.vs['log_degree'] = [np.log(d) + 1 for d in G.degree()]
        G.vs['clust_coeff'] = G.transitivity_local_undirected(mode='zero')
    splits[split] = G

print('Loaded 3 graphs!')

# Load feature matrices and prepare the standard scaler
train_indices = [n.index for n in splits['train'].vs]
features = np.load('{}/ppi-feats.npy'.format(DATASET_PATH)).astype(np.float32)
features = StandardScaler().fit(features[train_indices]).transform(features)
features = torch.from_numpy(features).float().detach().to(device)
labels = {int(k): v for k, v in json.load(open('{}/ppi-class_map.json'.format(DATASET_PATH))).items()}
labels = torch.FloatTensor([v for k, v in sorted(labels.items(), key=lambda x: x[0])]).detach().to(device)

agg_features = STRUCTURAL_VECTOR_LENGTH + features.shape[-1]
num_labels = labels.shape[-1]
num_features = features.shape[-1]

print('Loaded features and labels!')

# Define the model
class StaticStructSamplingModel(nn.Module):
    def __init__(self, graph_model, graph_outs, num_labels):
        super(StaticStructSamplingModel, self).__init__()
        self.graph_model = graph_model
        self.out = nn.Linear(graph_outs, num_labels)

    def forward(self, node_seq, G):
        g_out = self.graph_model(node_seq, G)
        out = self.out(g_out)
        return out


G_train = splits['train']
G_valid = splits['valid']
static_features = StaticFeatureEmbedder('id', features).to(device)
struct_features = StructuralEmbedder(G_train, STRUCTURAL_VECTOR_LENGTH, distance=1, use_distances=True, device=device)
node_features = MultiEmbedderAggregator([static_features, struct_features]).to(device)

# The model has as many sampled, attention residual graph embedding layers as specified
graph_model = node_features
graph_features = agg_features
for d in range(MODEL_DEPTH):
    graph_model = SamplingAggregator(graph_model, graph_features, aggregation=AGGREGATION_FN, num_hidden=NUM_HIDDEN, hidden_units=HIDDEN_UNITS, output_units=NUM_OUTPUTS, num_attention_heads=NUM_ATTENTION_HEADS, nodes_to_sample=NODES_TO_SAMPLE, sampling_model=SAMPLING_MODEL, include_node=INCLUDE_NODE).to(device)
    graph_features = max(1, NUM_ATTENTION_HEADS) * NUM_OUTPUTS + (graph_features * INCLUDE_NODE)

model = StaticStructSamplingModel(graph_model, graph_features, num_labels).to(device)

# Do the mapping beforehand
_ = struct_features.mapping(G_train.vs, G_train)
print('Prepared the node structural mapping...')

# Let's do some training
criterion = nn.MultiLabelSoftMarginLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

from tqdm import tqdm, trange
from time import time

early_stopping = EarlyStopping(TRAINING_PATIENCE, CHECKPOINT_PATH, TRAINING_MINIMUM_CHANGE)
epoch_bar = trange(1, NUM_EPOCHS + 1, desc='Epoch')
for epoch in epoch_bar:
    model = model.train()
    all_batches = list(random_bfs_samples(G_train, BATCH_SIZE))
    losses = []
    start_time = time()
    batches_bar = tqdm(all_batches, desc='Batch')
    for batch_indices in batches_bar:
        batch = G_train.vs[batch_indices]
        optimizer.zero_grad()
        output = model(batch, G_train)
        node_indices = batch['id']
        node_labels = labels[node_indices]
        loss = criterion(output, node_labels)
        losses.append(loss.data.mean().tolist())

        loss.backward()
        optimizer.step()
        running_loss = np.mean(losses)
        batches_bar.set_postfix(loss=running_loss)
    end_time = time()
    
    tqdm.write('Epoch {} took {:.2f} seconds, with a total running loss of {:.3f}.'.format(epoch, end_time - start_time, running_loss))

    model = model.eval()
    metrics = {}
    for split in ['valid']:
        G_split = splits[split]
        num_instances = len(G_split.vs)
        node_labels = labels[G_split.vs['id']]

        pred = model(G_split.vs, G_split)
        loss = criterion(pred, node_labels)
        split_pred = torch.sigmoid(pred).detach().reshape(num_instances, -1).cpu().numpy().round()
        split_true = node_labels.reshape(num_instances, -1).cpu().numpy()

        values = precision_recall_fscore_support(split_pred, split_true, average='micro')
        prec, recall, f1, _ = values
        split_loss = np.mean(loss.data.mean().tolist())
        metrics['{}_f1'.format(split)] = f1
        metrics['{}_loss'.format(split)] = split_loss
        if epoch % DISPLAY_EPOCH == 0 or epoch == NUM_EPOCHS:
            tqdm.write('Split: {} - Precision: {:.3f} - Recall: {:.3f} - F1-Score: {:.3f} - Loss: {:.3f}\n'.format(split, prec, recall, f1, split_loss))
    epoch_bar.set_postfix(**metrics)

    if early_stopping(metrics['valid_loss'], model):
        tqdm.write('Early stopping at epoch {}/{} with a best validation loss of {:.3f}.'.format(epoch, NUM_EPOCHS, early_stopping.best_loss))
        break

# Load the best model and eval on test
print('Loading the best model to evaluate on the test set.')
model = torch.load(CHECKPOINT_PATH, pickle_module=dill).eval()

G_test = splits['test']
num_instances = len(G_test.vs)
node_labels = labels[G_test.vs['id']]

pred = model(G_test.vs, G_test)
loss = criterion(pred, node_labels)
split_pred = torch.sigmoid(pred).detach().reshape(num_instances, -1).cpu().numpy().round()
split_true = node_labels.reshape(num_instances, -1).cpu().numpy()

values = precision_recall_fscore_support(split_pred, split_true, average='micro')
prec, recall, f1, _ = values
split_loss = np.mean(loss.data.mean().tolist())

print('Best model results on the test set. F1-score: {:.3f} - Loss: {:.3f}'.format(f1, split_loss))

