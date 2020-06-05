import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append('{}/../src/'.format(dir_path))

import time
import json
import torch
import igraph as ig
import pandas as pd

from itertools import product

from experiment_utils import create_experiment_params
from models import NegativeSamplingModel
from learning import set_seed
from model_utils import make_structural_model, train_negative_sampling
from parameters import IGELParameters, NegativeSamplingParameters, TrainingParameters


CPU = torch.device('cpu')
CUDA = torch.device('cuda')
DEVICE = GPU if len(sys.argv) > 1 and sys.argv[1] == 'gpu' else CPU

NUM_EPOCHS = 3
NUM_EXPERIMENTS = 5
BATCH_SIZE = 25000
EXPERIMENT_SEEDS = [x for x in range(1, NUM_EXPERIMENTS + 1)]
GRAPH_NODES = [64, 256, 1024, 4096, 16384, 65536]
EDGE_AMOUNTS = [2, 4, 8, 16]
ENCODING_DISTANCES = [1, 2, 3]


def make_parameters(encoding_distance):
    ns_parameters = NegativeSamplingParameters(random_walk_length=50,
                                               window_size=5,
                                               negatives_per_positive=5,
                                               minimum_negative_distance=1)
    igel_parameters = IGELParameters(model_type='simple',
                                     vector_length=25,
                                     encoding_distance=encoding_distance,
                                     use_distance_labels=True,
                                     counts_function='scale_norm', 
                                     counts_transform='log',
                                     neg_sampling_parameters=ns_parameters)
    training_parameters = TrainingParameters(batch_size=BATCH_SIZE,
                                             learning_rate=0.05,
                                             weight_decay=0.0,
                                             epochs=NUM_EPOCHS,
                                             display_epochs=1,
                                             batch_samples_fn='uniform',
                                             problem_type='unsupervised')
    return igel_parameters, training_parameters


def run_timed_experiment(G, igel_parameters, training_parameters, device):
    start_time = time.time()
    mapper, model = make_structural_model(G, igel_parameters, device)
    trained_model = train_negative_sampling(G, model, igel_parameters.neg_sampling_parameters, training_parameters, device)
    end_time = time.time()
    return end_time - start_time

def run_experiment_set(num_nodes, 
                       num_edges, 
                       encoding_distance, 
                       seeds=EXPERIMENT_SEEDS, 
                       device=DEVICE):
    total_edges = num_nodes * num_edges
    igel_parameters, training_parameters = make_parameters(encoding_distance)
    experiment_data = []
    for seed in seeds:
        set_seed(seed)
        G = ig.Graph.Erdos_Renyi(num_nodes, m=total_edges)
        G.vs['id'] = [node.index for node in G.vs]
        G.vs['name'] = [str(node.index) for node in G.vs]
        result = run_timed_experiment(G, 
                                      igel_parameters, 
                                      training_parameters, 
                                      device)
        result_dict = {
            'Nodes': num_nodes,
            'AverageEdges': num_edges,
            'ExperimentSeed': seed,
            'EncodingDistance': encoding_distance,
            'TotalRuntime': result
        }
        experiment_data.append(result_dict)
    return experiment_data

# Attempt to use a slurm array job reference and go sequential if it's not possible
experiment_data = [product_tuple for product_tuple in product(GRAPH_NODES, EDGE_AMOUNTS, ENCODING_DISTANCES)]
try:
    env_var = os.environ['SLURM_ARRAY_TASK_ID']
    task_index = int(env_var)
    num_nodes, num_edges, encoding_distance = experiment_data[task_index]
    results = run_experiment_set(num_nodes, num_edges, encoding_distance)
    for result_dict in results:
        print(json.dumps(result_dict))
except KeyError:
    for (num_nodes, num_edges, encoding_distance) in experiment_data:
        experiment_set_results = run_experiment_set(num_nodes, num_edges, encoding_distance)
        for result_dict in experiment_set_results:
            print(json.dumps(result_dict))
