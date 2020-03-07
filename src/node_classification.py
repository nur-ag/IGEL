import os
import json
import random

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from graph import load_graph
from models import NodeInferenceModel, NegativeSamplingModel
from learning import GraphNetworkTrainer, EarlyStopping
from batching import chunks, graph_random_walks, negative_sampling_generator, negative_sampling_batcher
from embedders import StaticFeatureEmbedder
from parameters import IGELParameters, NegativeSamplingParameters, TrainingParameters
from aggregators import MultiEmbedderAggregator
from model_utils import make_early_stopping, make_structural_model, train_negative_sampling

GRAPH_KEY = 'ppi'
GRAPH_PATH = '{}/../data/PPI/'.format(os.path.dirname(os.path.realpath(__file__)))
DEFAULT_DEVICE = torch.device('cpu')
SEED = 1337

LINK_PREDICTION_OUTPUTS = 1

NEGATIVE_SAMPLING_OPTIONS = NegativeSamplingParameters()

SIMPLE_MODEL_OPTIONS = IGELParameters(model_type='simple', vector_length=256, encoding_distance=2, use_distance_labels=True, neg_sampling_parameters=NEGATIVE_SAMPLING_OPTIONS)
GATED_MODEL_OPTIONS = IGELParameters(model_type='gated', vector_length=256, encoding_distance=2, use_distance_labels=True, gates_length=64, gates_steps=4, transform_output=True, counts_function='concat_both', aggregator_function='mean', neg_sampling_parameters=NEGATIVE_SAMPLING_OPTIONS)

TRAINING_OPTIONS = TrainingParameters(batch_size=64, learning_rate=0.001, weight_decay=0.0, epochs=500, display_epochs=1, batch_samples_fn='bfs', problem_type='multilabel')
UNSUPERVISED_TRAINING_OPTIONS = None

EARLY_STOPPING_METRIC = 'valid_f1'
CHECKPOINT_PATH = '{}/{}.checkpoint.pt'.format(GRAPH_PATH, GRAPH_KEY)
EARLY_STOPPING = EarlyStopping(patience=50, file_path=CHECKPOINT_PATH, metric=EARLY_STOPPING_METRIC, minimum_change=0.0, metric_sign=1 if EARLY_STOPPING_METRIC == 'valid_loss' else -1)

def load_node_splits(graph_path, splits_path):
    G = load_graph(graph_path)
    id_to_idx = {node['id']: node.index for node in G.vs}
    with open(splits_path, 'r') as f:
        split_data = json.load(f)
        splits = {k: (G, G.vs[[id_to_idx[n] for n in nodes]]) for k, nodes in split_data.items()}
    return splits

def load_graph_splits(root_path, prefix, all_splits):
    splits = {}
    for split in all_splits:
        split_path = '{}/{}-{}.edgelist'.format(root_path, prefix, split)
        G = load_graph(split_path)
        splits[split] = (G, G.vs)
    return splits

def load_splits(root_path, prefix, split_by_nodes=False):
    all_splits = ['train', 'valid', 'test']
    if split_by_nodes:
        graph_path = '{}/{}.edgelist'.format(root_path, prefix)
        splits_path = '{}/{}-splits.json'.format(root_path, prefix)
        splits = load_node_splits(graph_path, splits_path)
    else:
        splits = load_graph_splits(root_path, prefix, all_splits)
    return splits

def train_node_classification(splits, labels, embedding_model, model_options, training_options, early_stopping, features_size, device):
    G_train, train_nodes = splits['train']
    G_valid, valid_nodes = splits.get('valid', (None, None))
    G_test, test_nodes = splits.get('test', (None, None))

    # Prepare the splits for training
    validation_data = None if valid_nodes is None else (valid_nodes, labels[valid_nodes['id']], G_valid)
    test_data = None if test_nodes is None else (test_nodes, labels[test_nodes['id']], G_test)

    # Prepare the model, loss and optimizer
    vector_size = model_options.vector_length if model_options.model_type == 'simple' else model_options.gates_length
    model = NodeInferenceModel(embedding_model, vector_size + features_size, labels.shape[-1])
    criterion = nn.CrossEntropyLoss() if training_options.problem_type == 'multiclass' else nn.MultiLabelSoftMarginLoss()
    optimizer = optim.Adam(model.parameters(), lr=training_options.learning_rate, weight_decay=training_options.weight_decay)
    
    # Prepare the trainer and fit the graph
    trainer = GraphNetworkTrainer(model,
                                  optimizer, 
                                  criterion, 
                                  early_stopping=early_stopping,
                                  display_epochs=training_options.display_epochs, 
                                  problem_type=training_options.problem_type)
    batch_samples_fn = lambda: [(G_train.vs[batch], labels[G_train.vs[batch]['id']]) 
                                for batch in training_options.batch_samples_fn(G_train, train_nodes, training_options.batch_size)]
    trainer.fit(batch_samples_fn, G_train, num_epochs=training_options.epochs, validation_data=validation_data, test_data=test_data)
    return trainer, model

def node_classification_experiment(data_path,
                   data_prefix,
                   model_options=None,
                   training_options=None,
                   unsupervised_training_options=None,
                   use_features=True,
                   rescale_features=True,
                   split_by_nodes=False,
                   early_stopping=None,
                   device=DEFAULT_DEVICE,
                   seed=SEED):
    # Load splits
    splits = load_splits(data_path, data_prefix, split_by_nodes)
    G_train, train_nodes = splits['train']
    
    # Prepare labels
    labels = {int(k): v for k, v in json.load(open('{}/{}-class_map.json'.format(data_path, data_prefix))).items()}
    label_type = torch.LongTensor if training_options.problem_type == 'multiclass' else torch.FloatTensor
    labels = label_type([v for k, v in sorted(labels.items(), key=lambda x: x[0])]).detach().to(device)

    # Load features if needed
    if use_features:
        feature_matrix = np.load('{}/{}-feats.npy'.format(data_path, data_prefix)).astype(np.float32)
        if rescale_features:
            feature_matrix = StandardScaler().fit(feature_matrix[train_nodes['id']]).transform(feature_matrix)
        feature_matrix = torch.from_numpy(feature_matrix).float().detach().to(device)
        features_size = feature_matrix.shape[-1]
        features = StaticFeatureEmbedder('id', feature_matrix).to(device)
    else:
        features = None
        features_size = 0

    # Prepare the components and train the embedding
    mapper, igel_model = make_structural_model(G_train, model_options, device)

    # Train and freeze the model if we follow the unsupervised representation setting
    if model_options.neg_sampling_parameters is not None and unsupervised_training_options is not None:
        igel_model = train_negative_sampling(splits, model, model_options.neg_sampling_parameters, unsupervised_training_options, device)
        igel_model.requires_grad = False
        for param in igel_model.parameters():
            param.requires_grad = False

    # Build the complete model with features if specified
    igel_model = igel_model if features is None else MultiEmbedderAggregator([features, igel_model]).to(device)

    # Train the classification model
    trainer, model = train_node_classification(splits, labels, igel_model, model_options, training_options, early_stopping, features_size, device)

    # Evaluate and compute the final metrics
    metrics = trainer._test_metrics
    return metrics, metrics


model, metrics = node_classification_experiment(GRAPH_PATH, GRAPH_KEY, SIMPLE_MODEL_OPTIONS, TRAINING_OPTIONS, UNSUPERVISED_TRAINING_OPTIONS, early_stopping=EARLY_STOPPING)
print('Model reached {} test F1-score'.format(metrics['test_f1']))
