import os
import json

import numpy as np
import igraph as ig

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support

from samplers import *


DATASET_PATH = '{}/../PPI'.format(os.path.dirname(os.path.realpath(__file__)))
STRUCTURAL_VECTOR_LENGTH = 30
NUM_ATTENTION_HEADS = 8
NUM_HIDDEN = 3
HIDDEN_UNITS = 120
NUM_OUTPUTS = 40
BATCH_SIZE = 32
LEARNING_RATE = 0.005
NUM_EPOCHS = 100
DISPLAY_EPOCH = 5
NODES_TO_SAMPLE = 0
SAMPLING_MODEL = None # log_degree_sampler

USE_CUDA = True 
MOVE_TO_CUDA = USE_CUDA and torch.cuda.is_available()
device = torch.device('cuda') if MOVE_TO_CUDA else torch.device('cpu')

# Load the Graphs
all_splits = ['train', 'valid', 'test']
splits = {}
for split in all_splits:
    split_path = '{}/ppi-{}.edgelist'.format(DATASET_PATH, split)
    G = ig.Graph.Read_Ncol(split_path, names=True, weights=False, directed=False)
    G.vs['id'] = [int(n) for n in G.vs['name']]
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
sampling_features = NUM_ATTENTION_HEADS * NUM_OUTPUTS
num_labels = labels.shape[-1]
num_features = features.shape[-1]

print('Loaded features and labels!')

# Define the model
from embedders import StaticFeatureEmbedder, StructuralEmbedder
from aggregators import MultiEmbedderAggregator, SamplingAggregator


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
sfe = StaticFeatureEmbedder('id', features).to(device)
ste = StructuralEmbedder(G_train, STRUCTURAL_VECTOR_LENGTH, distance=1, use_distances=True, device=device)
mea = MultiEmbedderAggregator([sfe, ste]).to(device)
sa = SamplingAggregator(mea, agg_features, output_units=NUM_OUTPUTS, num_attention_heads=NUM_ATTENTION_HEADS).to(device)
saa = SamplingAggregator(sa, sampling_features, output_units=NUM_OUTPUTS, num_attention_heads=NUM_ATTENTION_HEADS).to(device)

sa_beef = SamplingAggregator(mea, agg_features, num_hidden=NUM_HIDDEN, hidden_units=HIDDEN_UNITS, output_units=NUM_OUTPUTS, num_attention_heads=NUM_ATTENTION_HEADS, nodes_to_sample=NODES_TO_SAMPLE, sampling_model=SAMPLING_MODEL).to(device)
sa_beef2 = SamplingAggregator(sa_beef, sampling_features, num_hidden=NUM_HIDDEN, hidden_units=HIDDEN_UNITS, output_units=NUM_OUTPUTS, num_attention_heads=NUM_ATTENTION_HEADS, nodes_to_sample=NODES_TO_SAMPLE, sampling_model=SAMPLING_MODEL).to(device)
sa_beef3 = SamplingAggregator(sa_beef2, sampling_features, num_hidden=NUM_HIDDEN, hidden_units=HIDDEN_UNITS, output_units=NUM_OUTPUTS, num_attention_heads=NUM_ATTENTION_HEADS, nodes_to_sample=NODES_TO_SAMPLE, sampling_model=SAMPLING_MODEL).to(device)

#model = StaticStructSamplingModel(sa0, 2 * agg_features * NUM_ATTENTION_HEADS, num_labels).to(device)
model = StaticStructSamplingModel(sa_beef, sampling_features, num_labels).to(device)

# Do the mapping beforehand
_ = ste.mapping(G_train.vs, G_train)
print('Prepared the node structural mapping...')

# Let's do some training
def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i+n]

criterion = nn.MultiLabelSoftMarginLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

indices = [node.index for node in G_train.vs]
from tqdm import tqdm, trange
from time import time
for epoch in trange(1, NUM_EPOCHS + 1, desc='Epoch'):
    model = model.train()
    np.random.shuffle(indices)
    all_batches = [batch for batch in chunks(indices, BATCH_SIZE)]
    losses = []
    start_time = time()
    for batch_indices in tqdm(all_batches, desc='Batch'):
        batch = G_train.vs[batch_indices]

        optimizer.zero_grad()
        output = model(batch, G_train)
        node_indices = batch['id']
        node_labels = labels[node_indices]
        loss = criterion(output, node_labels)
        losses.append(loss.data.mean().tolist())
        loss.backward()
        optimizer.step()
    end_time = time()
    running_loss = np.mean(losses)

    tqdm.write('Epoch {} took {} seconds, with a total running loss of {}.'.format(epoch, end_time - start_time, running_loss))
    if epoch % DISPLAY_EPOCH == 0:
        model = model.eval()
        for split in ['valid']:
            G_split = splits[split]
            num_instances = len(G_split.vs)
            pred = model(G_split.vs, G_split)
            split_pred = torch.sigmoid(pred).detach().reshape(num_instances, -1).cpu().numpy().round()
            split_true = labels[G_split.vs['id']].reshape(num_instances, -1).cpu().numpy()

            values = precision_recall_fscore_support(split_pred, split_true, average='micro')
            values = [split] + list(values)
            tqdm.write('Split: {} - Precision: {} - Recall: {} - F1-Score: {} - Support: {}\n'.format(*values))

