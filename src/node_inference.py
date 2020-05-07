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
from learning import GraphNetworkTrainer, EarlyStopping, set_seed
from batching import chunks, graph_random_walks, negative_sampling_generator, negative_sampling_batcher
from embedders import StaticFeatureEmbedder
from parameters import IGELParameters, AggregationParameters, NegativeSamplingParameters, TrainingParameters, DeepNNParameters
from aggregators import MultiEmbedderAggregator
from model_utils import make_structural_model, make_aggregation_model, train_negative_sampling

GRAPH_KEY = 'cora'
GRAPH_PATH = '{}/../data/cora/'.format(os.path.dirname(os.path.realpath(__file__)))
SEED = 1

USE_CUDA = True 
MOVE_TO_CUDA = USE_CUDA and torch.cuda.is_available()
DEFAULT_DEVICE = torch.device('cuda') if MOVE_TO_CUDA else torch.device('cpu')

NEGATIVE_SAMPLING_OPTIONS = NegativeSamplingParameters(random_walk_length=100, window_size=10, negatives_per_positive=10, minimum_negative_distance=2)

INFERENCE_MODEL_OPTIONS = DeepNNParameters(input_size=0, hidden_size=0, output_size=0, depth=0, activation='linear')
SIMPLE_MODEL_OPTIONS = IGELParameters(model_type='simple', vector_length=0, encoding_distance=4, use_distance_labels=True, neg_sampling_parameters=NEGATIVE_SAMPLING_OPTIONS)

TRAINING_OPTIONS = TrainingParameters(batch_size=10024, learning_rate=0.01, weight_decay=0.00001, epochs=1000, display_epochs=1, batch_samples_fn='uniform', problem_type='multiclass')
UNSUPERVISED_TRAINING_OPTIONS = TrainingParameters(batch_size=10024, learning_rate=0.2, weight_decay=0.0, epochs=1, display_epochs=1, batch_samples_fn='uniform', problem_type='unsupervised')

AGGREGATION_OPTIONS = AggregationParameters(node_dropouts=[0.3, 0.0], 
                                            output_sizes=[15, 7],
                                            activations=['elu', 'linear'],
                                            aggregations=['sum', 'sum'],
                                            include_nodes=[True, True],
                                            nodes_to_sample=[0, 0],
                                            num_attention_heads=[8, 1],
                                            attention_aggregators=['concat', 'sum'],
                                            attention_dropouts=[0.3, 0.0])

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


def train_node_inference(splits, labels, embedding_model, model_options, training_options, inference_options, early_stopping, embedding_size, device):
    G_train, train_nodes = splits['train']
    G_valid, valid_nodes = splits.get('valid', (None, None))
    G_test, test_nodes = splits.get('test', (None, None))

    # Prepare the splits for training
    validation_data = None if valid_nodes is None else (valid_nodes, labels[valid_nodes['id']], G_valid)
    test_data = None if test_nodes is None else (test_nodes, labels[test_nodes['id']], G_test)

    # Prepare the model, loss and optimizer
    total_labels = labels.shape[-1] if training_options.problem_type != 'multiclass' else (labels.max().item() + 1)
    if inference_options.output_size is None:
        total_labels = 0
    model = NodeInferenceModel(embedding_model, embedding_size, total_labels, hidden_size=inference_options.hidden_size, depth=inference_options.depth, activation=inference_options.activation).to(device)
    criterion = nn.CrossEntropyLoss() if training_options.problem_type == 'multiclass' else nn.MultiLabelSoftMarginLoss()
    optimizer = optim.Adam(model.parameters(), lr=training_options.learning_rate, weight_decay=training_options.weight_decay)
    
    # Prepare the trainer and fit the graph
    trainer = GraphNetworkTrainer(model,
                                  optimizer, 
                                  criterion, 
                                  early_stopping=early_stopping,
                                  display_epochs=training_options.display_epochs, 
                                  problem_type=training_options.problem_type)
    batch_samples_fn = lambda: ((G_train.vs[batch], labels[G_train.vs[batch]['id']]) 
                                for batch in training_options.batch_samples_fn(G_train, train_nodes, training_options.batch_size))
    trainer.fit(batch_samples_fn, G_train, num_epochs=training_options.epochs, validation_data=validation_data, test_data=test_data)
    return trainer, model


def node_inference_experiment(data_path, 
                              data_prefix, 
                              model_options, 
                              training_options, 
                              unsupervised_training_options=None,
                              aggregation_options=None,
                              inference_options=None,
                              use_features=True,
                              rescale_features=False,
                              split_by_nodes=True,
                              freeze_structural_model=True, 
                              early_stopping=EARLY_STOPPING,
                              device=DEFAULT_DEVICE, 
                              experiment_id=SEED, 
                              use_seed=True):
    seed = None
    if use_seed:
        seed = experiment_id
        set_seed(seed)

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
    structural_size = igel_model.output_size
    if structural_size > 0 and model_options.neg_sampling_parameters is not None and unsupervised_training_options is not None:
        igel_model = train_negative_sampling(G_train, igel_model, model_options.neg_sampling_parameters, unsupervised_training_options, device)

        if freeze_structural_model:
            igel_model.requires_grad = False
            for param in igel_model.parameters():
                param.requires_grad = False

    # Build the embedding model with features if specified
    if features is None and structural_size == 0:
        raise RuntimeException('You need either structural or feature embeddings!')

    if structural_size:
        embedding_model = igel_model if features is None else MultiEmbedderAggregator([features, igel_model]).to(device)
    else:
        embedding_model = features

    # Prepare the aggregation model if specified
    embedding_size = features_size + structural_size
    full_model, output_size = make_aggregation_model(embedding_model, embedding_size, aggregation_options, device)

    # Train the classification model
    trainer, model = train_node_inference(splits, labels, full_model, model_options, training_options, inference_options, early_stopping, output_size, device)

    # Evaluate and compute the final metrics
    metrics = trainer._test_metrics
    return metrics, metrics
