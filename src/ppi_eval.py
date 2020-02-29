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
from learning_utils import EarlyStopping, uniformly_random_samples, random_bfs_samples, random_walk_samples, graph_random_walks


DATASET_PATH = '{}/../data/PPI'.format(os.path.dirname(os.path.realpath(__file__)))
GRAPH_KEY = 'ppi'

BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 200

DISPLAY_EPOCH = 3

STRUCTURAL_VECTOR_LENGTH = 100
STRUCTURAL_GATES_LENGTH = 200
STRUCTURAL_GATES_STEPS = 3
STRUCTURAL_DISTANCE = 1
STRUCTURAL_TRANSFORM_OUTPUT = True
STRUCTURAL_COUNTS_FUNCTION = 'concat_both'
STRUCTURAL_AGGREGATOR_FUNCTION = 'mean'
STRUCTURAL_USE_DISTANCE_SYMBOLS = True

NUM_HIDDEN = 1
HIDDEN_UNITS = 200
NUM_OUTPUTS = 256
INCLUDE_NODE = True
MODEL_DEPTH = 0
AGGREGATION_FN = combine_max

NODES_TO_SAMPLE = 0
SAMPLING_MODEL = None # log_degree_sampler

NUMBER_OF_PEEKS = 4
PEEKING_UNITS = 200

NUM_ATTENTION_HEADS = 8
ATTENTION_AGG_FN = attention_sum
ATTENTION_OUTS_PER_HEAD = ATTENTION_AGG_FN == attention_concat
ATTENTION_LAST_LAYER_AGG_FN = attention_concat
ATTENTION_LAST_LAYER_OUTS_PER_HEAD = ATTENTION_LAST_LAYER_AGG_FN == attention_concat

BATCH_SAMPLES_FN = random_bfs_samples

TRAINING_PATIENCE = 20
TRAINING_MINIMUM_CHANGE = 0
CHECKPOINT_PATH = '{}/checkpoint-depth-{}.pt'.format(DATASET_PATH, MODEL_DEPTH)
TRAINING_PATIENCE_METRIC = 'valid_f1'
TRAINING_SIGN = 1 if TRAINING_MINIMUM_CHANGE == 'val_loss' else -1

RANDOM_WALK_LENGTH = 0
WINDOW_SIZE = 10
NEGATIVES_PER_POSITIVE = 10

TRAIN_NEGATIVE_SAMPLING = RANDOM_WALK_LENGTH > 0

USE_CUDA = True 
MOVE_TO_CUDA = USE_CUDA and torch.cuda.is_available()
device = torch.device('cuda') if MOVE_TO_CUDA else torch.device('cpu')

PRECOMPUTE_ATTRIBUTES = False

# Load the Graphs
all_splits = ['train', 'valid', 'test']
splits = {}
for split in all_splits:
    split_path = '{}/{}-{}.edgelist'.format(DATASET_PATH, GRAPH_KEY, split)
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
features = np.load('{}/{}-feats.npy'.format(DATASET_PATH, GRAPH_KEY)).astype(np.float32)
features = StandardScaler().fit(features[train_indices]).transform(features)
features = torch.from_numpy(features).float().detach().to(device)
labels = {int(k): v for k, v in json.load(open('{}/{}-class_map.json'.format(DATASET_PATH, GRAPH_KEY))).items()}
labels = torch.FloatTensor([v for k, v in sorted(labels.items(), key=lambda x: x[0])]).detach().to(device)

struct_features = STRUCTURAL_VECTOR_LENGTH if STRUCTURAL_GATES_STEPS <= 0 or STRUCTURAL_GATES_LENGTH <= 0 else STRUCTURAL_GATES_LENGTH
agg_features = struct_features + features.shape[-1]
num_labels = labels.shape[-1]
num_features = features.shape[-1]

print('Loaded features and labels!')

# Define the models
class GraphOutputModel(nn.Module):
    def __init__(self, graph_model, graph_outs, num_labels):
        super(GraphOutputModel, self).__init__()
        self.graph_model = graph_model
        self.out = nn.Linear(graph_outs, num_labels)

    def forward(self, node_seq, G):
        g_out = self.graph_model(node_seq, G)
        out = self.out(g_out)
        return out

class NegativeSamplingModel(nn.Module):
    def __init__(self, graph_model):
        super(NegativeSamplingModel, self).__init__()
        self.graph_model = graph_model
        self.out = nn.Linear(1, 1)

    def forward(self, sources, targets, G):
        all_nodes = sorted({n for n in sources + targets})
        nodes_mapping = {n: i for i, n in enumerate(all_nodes)}
        as_node_seq = G.vs[all_nodes]
        tensor = self.graph_model(as_node_seq, G)
        source_indices = [nodes_mapping[n] for n in sources]
        target_indices = [nodes_mapping[n] for n in targets]
        sources_tensor = tensor[source_indices, :]
        targets_tensor = tensor[target_indices, :]
        product_tensor = (sources_tensor * targets_tensor).sum(axis=1)
        out = self.out(product_tensor.reshape(-1, 1))
        return out


G_train = splits['train']
G_valid = splits['valid']
structural_mapper = StructuralMapper(G_train, distance=STRUCTURAL_DISTANCE, use_distances=STRUCTURAL_USE_DISTANCE_SYMBOLS)
static_features = StaticFeatureEmbedder('id', features).to(device)
if STRUCTURAL_GATES_STEPS <= 0 or STRUCTURAL_GATES_LENGTH <= 0:
    struct_features = SimpleStructuralEmbedder(STRUCTURAL_VECTOR_LENGTH, structural_mapper, device=device)
else:
    struct_features = GatedStructuralEmbedder(STRUCTURAL_VECTOR_LENGTH, STRUCTURAL_GATES_LENGTH, STRUCTURAL_GATES_STEPS, structural_mapper, STRUCTURAL_TRANSFORM_OUTPUT, STRUCTURAL_COUNTS_FUNCTION, STRUCTURAL_AGGREGATOR_FUNCTION, device=device)
node_features = MultiEmbedderAggregator([static_features, struct_features]).to(device)

# The model has as many sampled, attention residual graph embedding layers as specified
graph_model = node_features
graph_features = agg_features
for d in range(MODEL_DEPTH):
    if d == MODEL_DEPTH - 1:
        ATTENTION_AGG_FN = ATTENTION_LAST_LAYER_AGG_FN    
        ATTENTION_OUTS_PER_HEAD = ATTENTION_LAST_LAYER_OUTS_PER_HEAD
    graph_model = SamplingAggregator(graph_model, graph_features, aggregation=AGGREGATION_FN, num_hidden=NUM_HIDDEN, hidden_units=HIDDEN_UNITS, output_units=NUM_OUTPUTS, num_attention_heads=NUM_ATTENTION_HEADS, nodes_to_sample=NODES_TO_SAMPLE, sampling_model=SAMPLING_MODEL, include_node=INCLUDE_NODE, attention_aggregator=ATTENTION_AGG_FN, attention_outputs_by_head=ATTENTION_OUTS_PER_HEAD, number_of_peeks=NUMBER_OF_PEEKS, peeking_units=PEEKING_UNITS, device=device).to(device)
    graph_features = max(1, NUM_ATTENTION_HEADS if ATTENTION_OUTS_PER_HEAD else 1) * NUM_OUTPUTS + (graph_features * INCLUDE_NODE)

if TRAIN_NEGATIVE_SAMPLING:
    model = NegativeSamplingModel(graph_model).to(device)
else:
    model = GraphOutputModel(graph_model, graph_features, num_labels).to(device)

# Let's do some training
criterion = nn.MultiLabelSoftMarginLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

from tqdm import tqdm, trange
from time import time

early_stopping = EarlyStopping(TRAINING_PATIENCE, CHECKPOINT_PATH, TRAINING_MINIMUM_CHANGE, TRAINING_SIGN)
epoch_bar = trange(1, NUM_EPOCHS + 1, desc='Epoch')
for epoch in epoch_bar:
    all_batches = list(BATCH_SAMPLES_FN(G_train, BATCH_SIZE))
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
        losses.append(loss.data.detach().mean().item())

        loss.backward()
        optimizer.step()
        running_loss = np.mean(losses)
        batches_bar.set_postfix(loss=running_loss)
    end_time = time()
    
    tqdm.write('Epoch {} took {:.2f} seconds, with a total running loss of {:.3f}.'.format(epoch, end_time - start_time, running_loss))
    with torch.no_grad():
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

        if early_stopping(metrics[TRAINING_PATIENCE_METRIC], model):
            tqdm.write('Early stopping at epoch {}/{} with a best validation score of {:.3f}.'.format(epoch, NUM_EPOCHS, early_stopping.best_metric))
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

