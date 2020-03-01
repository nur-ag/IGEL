import os
import json
import logging

import dill

import numpy as np
import igraph as ig

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler

from samplers import *
from embedders import *
from aggregators import *
from learning import EarlyStopping, GraphNetworkTrainer
from models import NodeClassificationModel, NegativeSamplingModel
from batching import uniformly_random_samples, random_bfs_samples, random_walk_samples, graph_random_walks, negative_sampling_generator, negative_sampling_batcher


DATASET_PATH = '{}/../data/PPI'.format(os.path.dirname(os.path.realpath(__file__)))
GRAPH_KEY = 'ppi'

BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 200

DISPLAY_EPOCH = 3

STRUCTURAL_VECTOR_LENGTH = 100
STRUCTURAL_GATES_LENGTH = 0
STRUCTURAL_GATES_STEPS = 4
STRUCTURAL_DISTANCE = 1
STRUCTURAL_TRANSFORM_OUTPUT = True
STRUCTURAL_COUNTS_FUNCTION = 'concat_both'
STRUCTURAL_AGGREGATOR_FUNCTION = 'mean'
STRUCTURAL_USE_DISTANCE_SYMBOLS = True

NUM_HIDDEN = 1
HIDDEN_UNITS = 256
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

RANDOM_WALK_LENGTH = 80
WINDOW_SIZE = 10
NEGATIVES_PER_POSITIVE = 10

TRAIN_NEGATIVE_SAMPLING = RANDOM_WALK_LENGTH > 0

TRAINING_PATIENCE = 20
TRAINING_MINIMUM_CHANGE = 0
CHECKPOINT_PATH = '{}/checkpoint-{}-depth-{}.pt'.format(DATASET_PATH, 'unsup' if TRAIN_NEGATIVE_SAMPLING else 'sup', MODEL_DEPTH)
TRAINING_PATIENCE_METRIC = 'valid_f1'
TRAINING_SIGN = 1 if TRAINING_MINIMUM_CHANGE == 'val_loss' else -1

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

G_train = splits['train']
G_valid = splits['valid']
G_test = splits['test']
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
    model = NodeClassificationModel(graph_model, graph_features, num_labels).to(device)

# Let's do some training
def batch_iterator_fn():
    if TRAIN_NEGATIVE_SAMPLING:
        batch_gen = BATCH_SAMPLES_FN(G_train, BATCH_SIZE)
        ns_gen = negative_sampling_generator(G_train, batch_gen, WINDOW_SIZE, NEGATIVES_PER_POSITIVE)
        pair_labels = negative_sampling_batcher(ns_gen, BATCH_SIZE)
        all_batches = ((pair, torch.Tensor(label).to(device).reshape(-1, 1)) for pair, label in pair_labels)
    else:
        all_batches = [(G_train.vs[batch], labels[G_train.vs[batch]['id']]) for batch in BATCH_SAMPLES_FN(G_train, BATCH_SIZE)]
    return all_batches

criterion = nn.MultiLabelSoftMarginLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
early_stopping = EarlyStopping(TRAINING_PATIENCE, CHECKPOINT_PATH, TRAINING_PATIENCE_METRIC, TRAINING_MINIMUM_CHANGE, TRAINING_SIGN)

validation_data = None if TRAIN_NEGATIVE_SAMPLING else (G_valid.vs, labels[G_valid.vs['id']], G_valid)
test_data = None if TRAIN_NEGATIVE_SAMPLING else (G_test.vs, labels[G_test.vs['id']], G_test)
trainer = GraphNetworkTrainer(model, optimizer, criterion, display_epochs=DISPLAY_EPOCH, early_stopping=early_stopping)
trainer.fit(batch_iterator_fn, G_train, NUM_EPOCHS, validation_data=validation_data, test_data=test_data)
