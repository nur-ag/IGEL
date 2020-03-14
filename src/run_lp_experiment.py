import os
import json
import random

import torch
from filelock import FileLock

from parameters import IGELParameters, NegativeSamplingParameters, TrainingParameters
from link_prediction import link_prediction_experiment
from experiment_utils import generate_experiment_tuples, tuple_to_dictionary

GRAPH_PATH = 'data/Facebook/Facebook.edgelist'
OUTPUT_PATH = 'output/Facebook-result.jsonl'
OUTPUT_LOCK_PATH = OUTPUT_PATH + '.lock'
LP_TRAINING_OPTIONS = TrainingParameters(batch_size=512, learning_rate=0.1, weight_decay=0.0, epochs=100, display_epochs=1, batch_samples_fn='uniform', problem_type='unsupervised')

USE_CUDA = True 
MOVE_TO_CUDA = USE_CUDA and torch.cuda.is_available()
DEVICE = torch.device('cuda') if MOVE_TO_CUDA else torch.device('cpu')

NUM_EXPERIMENTS = 10
EXPERIMENTAL_CONFIG = {
    'epochs': [1],
    'batch_size': [8192],
    'learning_rate': [1.0, 0.2, 0.04, 0.01],
    'problem_type': ['unsupervised'],
    'batch_samples_fn': ['uniform'],
    'display_epochs': [1],
    'weight_decay': [0.0],
    'random_walk_length': [40, 80, 120],
    'window_size': [5, 10, 15, 20],
    'negatives_per_positive': [1, 2, 5, 10, 20],
    'encoding_distance': [1, 2],
    'vector_length': [32, 64, 128],
    'model_type': ['simple'],
    'use_distance_labels': [True],
    'transform_output': [True],
    'gates_steps': [4],
    'counts_transform': ['identity', 'log'],
    'counts_function': ['concat_both'],
    'aggregator_function': ['mean']
}


def load_finished_experiments(experiments_path):
    seen_experiments = set()
    if not os.path.exists(experiments_path):
        return seen_experiments

    with open(experiments_path) as f:
        for line in f:
            experiment_dict = json.loads(line.strip())['config']
            config_as_tuple = tuple([t for t in sorted(experiment_dict.items())])
            seen_experiments.add(config_as_tuple)
    return seen_experiments


def create_experiment_params(experiment_dict):
    ns_params = {'random_walk_length', 'window_size', 'negatives_per_positive'}
    ns_dict = {k: v for (k, v) in experiment_dict.items() if k in ns_params}
    ns_opt = NegativeSamplingParameters(**ns_dict)

    model_params = {'encoding_distance', 'vector_length', 'model_type', 'use_distance_labels', 'gates_steps', 'counts_transform', 'counts_function', 'aggregator_function'}
    model_dict = {k: v for (k, v) in experiment_dict.items() if k in model_params}
    model_dict['gates_length'] = model_dict['vector_length']
    model_dict['neg_sampling_parameters'] = ns_opt
    model_options = IGELParameters(**model_dict)

    training_params = {'epochs', 'batch_size', 'learning_rate', 'problem_type', 'batch_samples_fn', 'display_epochs', 'weight_decay'}
    training_dict = {k: v for (k, v) in experiment_dict.items() if k in training_params}
    training_options = TrainingParameters(**training_dict)
    return model_options, training_options


def run_experiment(experiment):
    experiment_as_dict = tuple_to_dictionary(experiment)
    model_options, training_options = create_experiment_params(experiment_as_dict)

    results = []
    for n in range(1, NUM_EXPERIMENTS + 1):
        model, metrics = link_prediction_experiment(GRAPH_PATH, model_options, training_options, LP_TRAINING_OPTIONS, device=DEVICE, seed=n)
        results.append(metrics)
    results_dict = {'config': experiment_as_dict, 'results': results}
    return results_dict


def run_and_save_experiment(experiment):
    results_dict = run_experiment(experiment)
    print(experiment, np.mean(results_dict['results']))
    with FileLock(OUTPUT_LOCK_PATH, timeout=5):
        with open(OUTPUT_PATH, 'a+') as f:
            results_json = json.dumps(results_dict)
            f.write('{}\n'.format(results_json))


seen_experiments = load_finished_experiments(OUTPUT_PATH)
experiments = [e for e in generate_experiment_tuples(EXPERIMENTAL_CONFIG) 
                 if e not in seen_experiments]
random.shuffle(experiments)
for experiment in experiments:
    run_and_save_experiment(experiment)
