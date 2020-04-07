import os
import json
import random
import threading

import torch
import torch.multiprocessing as mp
import numpy as np
from filelock import FileLock

from parameters import IGELParameters, NegativeSamplingParameters, TrainingParameters
from link_prediction import link_prediction_experiment
from experiment_utils import generate_experiment_tuples, tuple_to_dictionary

GRAPH_PATH = 'data/Facebook/Facebook.edgelist'
OUTPUT_PATH = 'output/Facebook-result.jsonl'
OUTPUT_LOCK_PATH = OUTPUT_PATH + '.lock'
LP_TRAINING_OPTIONS = TrainingParameters(batch_size=512, learning_rate=0.1, weight_decay=0.0, epochs=20, display_epochs=1, batch_samples_fn='uniform', problem_type='unsupervised')

USE_CUDA = True
MOVE_TO_CUDA = USE_CUDA and torch.cuda.is_available()

NUM_WORKERS = 6
mp = mp.get_context('forkserver')

NUM_EXPERIMENTS = 10
EXPERIMENTAL_CONFIG = {
    'epochs': [1, 3, 5],
    'batch_size': [2048],
    'learning_rate': [0.5, 0.1, 0.05, 0.01],
    'problem_type': ['unsupervised'],
    'batch_samples_fn': ['uniform'],
    'display_epochs': [1],
    'weight_decay': [0.0],
    'random_walk_length': [20, 30, 40, 60, 80],
    'window_size': [2, 5, 8, 10],
    'negatives_per_positive': [2, 5, 8, 10],
    'encoding_distance': [1, 2],
    'vector_length': [64, 128, 256],
    'model_type': ['simple', 'gated'],
    'use_distance_labels': [True],
    'gates_steps': [2, 3, 4],
    'counts_transform': ['uniform', 'identity', 'log'],
    'counts_function': ['scale_norm'],
    'aggregator_function': ['sum']
}


def load_finished_experiments(experiments_path):
    seen_experiments = set()
    if not os.path.exists(experiments_path):
        return seen_experiments

    with open(experiments_path) as f:
        for line in f:
            experiment_dict = json.loads(line.strip())['config']
            config_as_tuple = tuple([t for t in sorted(experiment_dict.items()) if t[0] in EXPERIMENTAL_CONFIG])
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


def compute_stats(results):
    return {'min': np.min(results),
            'max': np.max(results),
            'std': np.std(results),
            'avg': np.mean(results)}


def run_experiment(experiment):
    experiment_as_dict = tuple_to_dictionary(experiment)
    model_options, training_options = create_experiment_params(experiment_as_dict)
    device = torch.device('cuda') if MOVE_TO_CUDA else torch.device('cpu')

    results = []
    for n in range(1, NUM_EXPERIMENTS + 1):
        model, metrics = link_prediction_experiment(GRAPH_PATH, model_options, training_options, LP_TRAINING_OPTIONS, device=device, seed=n)
        results.append(metrics)
    results_dict = {'config': experiment_as_dict, 'results': results, 'stats': compute_stats(results)}
    return results_dict


def run_and_save_experiment(experiment):
    results_dict = run_experiment(experiment)
    print(experiment, results_dict['results'], np.mean(results_dict['results']))
    with FileLock(OUTPUT_LOCK_PATH, timeout=5):
        with open(OUTPUT_PATH, 'a+') as f:
            results_json = json.dumps(results_dict)
            f.write('{}\n'.format(results_json))

def run_experiments_on_thread(experiments):
    for experiment in experiments:
        run_and_save_experiment(experiment)

if __name__ == '__main__':
    seen_experiments = load_finished_experiments(OUTPUT_PATH)
    experiments = [e for e in generate_experiment_tuples(EXPERIMENTAL_CONFIG)
                     if e not in seen_experiments]
    random.shuffle(experiments)

    workers = []
    mp.freeze_support()
    for index in range(NUM_WORKERS):
        p = mp.Process(target=run_experiments_on_thread, args=(experiments[index::NUM_WORKERS],))
        p.start()
        workers.append(p)

    for w in workers:
        w.join()


