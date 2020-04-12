import os
import json
import dill
import random
import argparse
import threading

import torch
import torch.multiprocessing as mp
import numpy as np
from filelock import FileLock

from parameters import IGELParameters, NegativeSamplingParameters, TrainingParameters
from link_prediction import link_prediction_experiment
from experiment_utils import generate_experiment_tuples, tuple_to_dictionary, hash_dictionary

KEY = 'Facebook' 
#KEY = 'CA-AstroPh'
GRAPH_PATH = 'data/{}/{}.edgelist'.format(KEY, KEY)
OUTPUT_PATH = 'output/{}-result.jsonl'.format(KEY)
MODEL_OUTPUT_PATH = 'output/{}'.format(KEY)
EXPERIMENTAL_PATH = 'configs/linkPrediction.json'
OUTPUT_LOCK_PATH = OUTPUT_PATH + '.lock'
LP_TRAINING_OPTIONS = TrainingParameters(batch_size=2048, learning_rate=0.05, weight_decay=0.0, epochs=30, display_epochs=1, batch_samples_fn='uniform', problem_type='unsupervised')

NUM_WORKERS = 1
mp = mp.get_context('forkserver')

NUM_EXPERIMENTS = 3


def load_finished_experiments(experiments_path, experimental_config):
    seen_experiments = set()
    if not os.path.exists(experiments_path):
        return seen_experiments

    with open(experiments_path) as f:
        for line in f:
            experiment_dict = json.loads(line.strip())['config']
            config_as_tuple = tuple([t for t in sorted(experiment_dict.items()) if t[0] in experimental_config])
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


def run_experiment(experiment, num_attempts, graph_path, model_output_path, move_to_cuda):
    experiment_as_dict = tuple_to_dictionary(experiment)
    experiment_hash = hash_dictionary(experiment_as_dict)[:8]
    model_options, training_options = create_experiment_params(experiment_as_dict)
    device = torch.device('cuda') if move_to_cuda else torch.device('cpu')

    results = []
    for n in range(1, num_attempts + 1):
        model, metrics = link_prediction_experiment(graph_path, 
                                                    model_options, 
                                                    training_options, 
                                                    LP_TRAINING_OPTIONS, 
                                                    freeze_structural_model=experiment_as_dict['freeze_embeddings'], 
                                                    device=device, 
                                                    experiment_id=n, 
                                                    use_seed=experiment_as_dict['use_seed'])
        results.append(metrics)
        if model_output_path.strip():
            experiment_model_path = '{}.{}-{}.model'.format(model_output_path, experiment_hash, n)
            torch.save(model, experiment_model_path, pickle_module=dill)
            print('Finished experiment {} with result: {}. Storing model in {}.'.format(n, metrics, experiment_model_path))
        else:
            print('Finished experiment {} with result: {}.'.format(n, metrics))
    results_dict = {'config': experiment_as_dict, 'results': results, 'stats': compute_stats(results), 'hash': experiment_hash}
    return results_dict


def run_and_save_experiment(experiment, 
                            num_attempts, 
                            graph_path, 
                            output_path,
                            model_output_path,
                            move_to_cuda):
    results_dict = run_experiment(experiment, num_attempts, graph_path, model_output_path, move_to_cuda)
    print(experiment, results_dict['results'], np.mean(results_dict['results']))
    with FileLock('{}.lock'.format(output_path), timeout=5):
        with open(output_path, 'a+') as f:
            results_json = json.dumps(results_dict)
            f.write('{}\n'.format(results_json))

def run_experiments_on_thread(experiments, 
                              num_attempts, 
                              graph_path, 
                              output_path, 
                              model_output_path, 
                              move_to_cuda):
    for experiment in experiments:
        run_and_save_experiment(experiment, num_attempts, 
                                graph_path, output_path, 
                                model_output_path,
                                move_to_cuda)


def run(num_workers, num_attempts, graph_path, output_path, model_output_path, experimental_config, move_to_cuda):
    seen_experiments = load_finished_experiments(output_path, experimental_config)
    experiments = [e for e in generate_experiment_tuples(experimental_config)
                     if e not in seen_experiments]
    random.shuffle(experiments)

    workers = []
    mp.freeze_support()
    for index in range(num_workers):
        arguments = (experiments[index::num_workers], num_attempts, graph_path, output_path, model_output_path, move_to_cuda)
        proc = mp.Process(target=run_experiments_on_thread, args=arguments)
        proc.start()
        workers.append(proc)

    for w in workers:
        w.join()


def parse_arguments():
    main_args = argparse.ArgumentParser()
    main_args.add_argument('-w', '--num-workers', help='Number of workers to use when running the experiment, each running experiments in parallel.', type=int, default=NUM_WORKERS)
    main_args.add_argument('-n', '--num-attempts', help='Number of experiment attempts to run per experiment configuration.', type=int, default=NUM_EXPERIMENTS)
    main_args.add_argument('-g', '--graph-path', help='Path to the graph edgelist file to be used.', type=str, default=GRAPH_PATH)
    main_args.add_argument('-o', '--output-path', help='Path to store the experiment results.', type=str, default=OUTPUT_PATH)
    main_args.add_argument('-e', '--experimental-config', help='Path to the hyperparameter search configuration to run experiments from.', type=str, default=EXPERIMENTAL_PATH)
    main_args.add_argument('-m', '--model-output-path', help='Path to store the trained models from the experiment. If empty string, no model will be stored.', type=str, default=MODEL_OUTPUT_PATH)
    main_args.add_argument('-c', '--cuda', help='Flag to specify that cuda should be used.', action='store_true')
    return main_args.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    num_workers = args.num_workers
    num_attempts = args.num_attempts
    graph_path = args.graph_path
    output_path = args.output_path
    model_output_path = args.model_output_path
    experimental_config = json.load(open(args.experimental_config))
    move_to_cuda = args.cuda and torch.cuda.is_available()
    run(num_workers, num_attempts, graph_path, output_path, model_output_path, experimental_config, move_to_cuda)

