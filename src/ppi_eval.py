import json

import numpy as np
import igraph as ig

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support


DATASET_PATH = '/Users/nur/Desktop/PhD/IGEL/graph/PPI'
STRUCTURAL_VECTOR_LENGTH = 30
NUM_ATTENTION_HEADS = 8
NUM_OUTPUTS = 40

features = np.load('{}/ppi-feats.npy'.format(DATASET_PATH)).astype(np.float32)
features = StandardScaler().fit_transform(features)
features = torch.from_numpy(features).float().detach()
labels = {int(k): v for k, v in json.load(open('{}/ppi-class_map.json'.format(DATASET_PATH))).items()}
labels = torch.FloatTensor([v for k, v in sorted(labels.items(), key=lambda x: x[0])]).detach()

agg_features = STRUCTURAL_VECTOR_LENGTH + features.shape[-1]
sampling_features = NUM_ATTENTION_HEADS * NUM_OUTPUTS
num_labels = labels.shape[-1]

all_splits = ['train', 'valid', 'test']
splits = {}
for split in all_splits:
    split_path = '{}/ppi-{}.edgelist'.format(DATASET_PATH, split)
    G = ig.Graph.Read_Ncol(split_path, names=True, weights=False, directed=False)
    G.vs['id'] = [int(n) for n in G.vs['name']]
    splits[split] = G

print('Loaded 3 graphs!')


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
sfe = StaticFeatureEmbedder('id', features)
ste = StructuralEmbedder(G_train, STRUCTURAL_VECTOR_LENGTH, distance=1, use_distances=True)
mea = MultiEmbedderAggregator([sfe, ste])
sa = SamplingAggregator(mea, agg_features, output_units=NUM_OUTPUTS, num_attention_heads=NUM_ATTENTION_HEADS)
saa = SamplingAggregator(sa, sampling_features, output_units=NUM_OUTPUTS, num_attention_heads=NUM_ATTENTION_HEADS)

sa_beef = SamplingAggregator(mea, agg_features, num_hidden=3, hidden_units=120, output_units=NUM_OUTPUTS, num_attention_heads=NUM_ATTENTION_HEADS)
sa_beef2 = SamplingAggregator(sa_beef, sampling_features, num_hidden=3, hidden_units=120, output_units=NUM_OUTPUTS, num_attention_heads=NUM_ATTENTION_HEADS)

sa0 = SamplingAggregator(mea, agg_features, output_units=0, hidden_units=0, num_hidden=0, num_attention_heads=NUM_ATTENTION_HEADS)

sm1 = SamplingAggregator(mea, agg_features, output_units=NUM_OUTPUTS, num_attention_heads=NUM_ATTENTION_HEADS, nodes_to_sample=10)
sm2 = SamplingAggregator(sm1, sampling_features, output_units=NUM_OUTPUTS, num_attention_heads=NUM_ATTENTION_HEADS, nodes_to_sample=25)

sa1 = SamplingAggregator(sfe, features.shape[-1], output_units=NUM_OUTPUTS, num_attention_heads=NUM_ATTENTION_HEADS)
sa2 = SamplingAggregator(sa1, sampling_features, output_units=NUM_OUTPUTS, num_attention_heads=NUM_ATTENTION_HEADS)

#model = StaticStructSamplingModel(sa0, 2 * agg_features * NUM_ATTENTION_HEADS, num_labels)
model = StaticStructSamplingModel(sa_beef2, sampling_features, num_labels)

def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i+n]

# Let's do some training
criterion = nn.MultiLabelSoftMarginLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

batch_size = 512
indices = [node.index for node in G_train.vs]
from time import time
for epoch in range(1, 51):
    np.random.shuffle(indices)
    start_time = time()
    losses = []
    for batch_indices in chunks(indices, batch_size):
        batch = G_train.vs[batch_indices]

        optimizer.zero_grad()
        output = model(batch, G_train)
        node_indices = batch['id']
        node_labels = labels[node_indices]
        loss = criterion(output, node_labels)
        losses.append(loss.data.mean())
        loss.backward()
        optimizer.step()
    end_time = time()
    running_loss = np.mean(losses)
    
    print('Epoch {} took {} seconds, with a total running loss of {}.'.format(epoch, end_time - start_time, running_loss))
    if epoch % 5 == 0:
        for split in ['valid']:
            G_split = splits[split]
            num_instances = len(G_split.vs)
            pred = model(G_split.vs, G_split)
            split_pred = torch.sigmoid(pred).detach().reshape(num_instances, -1).numpy().round()
            split_true = labels[G_split.vs['id']].reshape(num_instances, -1).numpy()

            values = precision_recall_fscore_support(split_pred, split_true, average='micro')
            values = [split] + list(values)
            print('Split: {} - Precision: {} - Recall: {} - F1-Score: {} - Support: {}\n'.format(*values))

