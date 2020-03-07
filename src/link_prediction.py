import random

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from graph import load_graph, sample_edges, edge_difference, generate_negative_edges
from models import EdgeInferenceModel, NegativeSamplingModel
from learning import GraphNetworkTrainer
from batching import chunks
from parameters import IGELParameters, NegativeSamplingParameters, TrainingParameters
from model_utils import make_structural_model, train_negative_sampling

GRAPH_PATH = 'data/Facebook/Facebook.edgelist'
EDGES_TO_SAMPLE = 0.5
VALID_TEST_SPLIT = 0.5
DEFAULT_DEVICE = torch.device('cpu')
SEED = 1337

LINK_PREDICTION_OUTPUTS = 1

NEGATIVE_SAMPLING_OPTIONS = NegativeSamplingParameters(80, 10, 10)

SIMPLE_MODEL_OPTIONS = IGELParameters(model_type='simple', vector_length=100, encoding_distance=2, use_distance_labels=True, counts_transform='log', neg_sampling_parameters=NEGATIVE_SAMPLING_OPTIONS)
GATED_MODEL_OPTIONS = IGELParameters(model_type='gated', vector_length=100, encoding_distance=2, use_distance_labels=True, gates_length=64, gates_steps=4, transform_output=True, counts_function='concat_both', aggregator_function='mean', counts_transform='log', neg_sampling_parameters=NEGATIVE_SAMPLING_OPTIONS)

TRAINING_OPTIONS = TrainingParameters(batch_size=65536, learning_rate=0.001, weight_decay=0.0, epochs=5, display_epochs=1, batch_samples_fn='uniform', problem_type='unsupervised')
LP_TRAINING_OPTIONS = TrainingParameters(batch_size=256, learning_rate=0.01, weight_decay=0.0, epochs=500, display_epochs=1, batch_samples_fn='uniform', problem_type='unsupervised')

def train_link_prediction(G, edges, labels, structural_model, model_options, training_options, edge_function, device):
    vector_size = model_options.vector_length if model_options.model_type == 'simple' else model_options.gates_length
    lp_model = EdgeInferenceModel(structural_model, vector_size, LINK_PREDICTION_OUTPUTS, edge_function)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(lp_model.parameters(), lr=training_options.learning_rate, weight_decay=training_options.weight_decay)
    trainer = GraphNetworkTrainer(lp_model,
                                  optimizer, 
                                  criterion, 
                                  display_epochs=training_options.display_epochs, 
                                  problem_type='unsupervised')
    def edge_batch_fn():
        edge_labels = zip(edges, labels)
        as_chunks = chunks(edge_labels, link_prediction_batch_size)
        for chunk in as_chunks:
            X, y = zip(*chunk)
            src, dst = zip(*X)
            y_tensor = torch.Tensor(list(y)).to(device).reshape(-1, 1)
            yield ((src, dst), y_tensor)
    trainer.fit(lambda: edge_batch_fn(), G, num_epochs=training_options.epochs)
    return trainer, lp_model

def evaluate_model(link_model, edges, labels, G):
    scores = link_model(zip(*edges), G).detach().numpy()
    labels = np.asarray(labels).reshape(scores.shape)
    return roc_auc_score(labels, scores)

def link_prediction_experiment(graph_path, 
                   model_options=None,
                   training_options=None,
                   link_prediction_training_options=None,
                   edge_function='mul',
                   edges_to_sample=EDGES_TO_SAMPLE,
                   edges_valid_to_test=VALID_TEST_SPLIT,
                   freeze_structural_model=True,
                   link_prediction_scale=1,
                   device=DEFAULT_DEVICE,
                   seed=SEED):
    G = load_graph(GRAPH_PATH)
    G_train = sample_edges(G, edges_to_sample, connected=True, seed=seed)
    id_to_indices = {n['id']: n.index for n in G_train.vs}

    # Compute all the edges needed for training and validating link prediction
    link_eval_edges_pos = edge_difference(G, G_train) * link_prediction_scale
    link_eval_edges_neg = generate_negative_edges(link_eval_edges_pos, G, link_prediction_scale)
    link_eval_edges = [(id_to_indices[a], id_to_indices[b]) for a, b in link_eval_edges_pos + link_eval_edges_neg]
    link_eval_labels = [1] * len(link_eval_edges_pos) + [0] * len(link_eval_edges_neg)

    # Prepare the splits that will be used for training and evaluation
    X_link, X_eval, y_link, y_eval = train_test_split(link_eval_edges, 
                                                      link_eval_labels,
                                                      train_size=edges_valid_to_test, 
                                                      test_size=1.0 - edges_valid_to_test,
                                                      random_state=seed)

    # Prepare the components and train the embedding and link prediction models
    mapper, model = make_structural_model(G_train, model_options, device)
    trained_model = train_negative_sampling(G_train, model, model_options.neg_sampling_parameters, training_options, device)
    
    # Freeze the model -- unsupervised representation setting with no retraining
    if freeze_structural_model:
        trained_model.requires_grad = False
        for param in trained_model.parameters():
            param.requires_grad = False

    trainer, link_model = train_link_prediction(G_train, X_link, y_link, model, model_options, link_prediction_training_options, edge_function, device)

    # Evaluate and compute the final metrics
    metrics = evaluate_model(link_model, X_eval, y_eval, G_train)
    return link_model, metrics


model, metrics = link_prediction_experiment(GRAPH_PATH, SIMPLE_MODEL_OPTIONS, TRAINING_OPTIONS, LP_TRAINING_OPTIONS)
print('AuROC: {}'.format(metrics))
