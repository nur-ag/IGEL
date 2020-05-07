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

from learning import EarlyStopping
from node_inference import node_inference_experiment
from experiment_utils import create_experiment_params, generate_experiment_dicts, tuple_to_dictionary, hash_dictionary, compute_stats

GRAPH_PREFIX = 'cora' 
GRAPH_PATH = 'data/cora'

OUTPUT_PATH = 'output/{}-result.jsonl'.format(GRAPH_PREFIX)
MODEL_OUTPUT_PATH = 'output/{}'.format(GRAPH_PREFIX)
EXPERIMENTAL_PATH = 'configs/transductiveNodeInference.json'

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
            config_as_string = json.dumps(experiment_dict)
            seen_experiments.add(config_as_string)
    return seen_experiments


def run_experiment(experiment, num_attempts, data_path, data_prefix, model_output_path, move_to_cuda):
    experiment_hash = hash_dictionary(experiment)[:8]
    model_options, training_options, unsupervised_training_options, aggregation_options, inference_options = create_experiment_params(experiment)
    device = torch.device('cuda') if move_to_cuda else torch.device('cpu')

    patience = experiment['patience']
    early_stopping_metric = experiment['early_stopping_metric']
    checkpoint_path = '{}/checkpoint.pt'.format(data_path)
    results = []
    experiment_metrics = []
    for n in range(1, num_attempts + 1):
        early_stopping = EarlyStopping(patience=patience, file_path=checkpoint_path, metric=early_stopping_metric, minimum_change=0.0, metric_sign=1 if early_stopping_metric == 'valid_loss' else -1)
        model, metrics = node_inference_experiment(data_path, 
                                                   data_prefix, 
                                                   model_options, 
                                                   training_options,
                                                   unsupervised_training_options,
                                                   aggregation_options,
                                                   inference_options, 
                                                   use_features=experiment['use_features'],
                                                   rescale_features=experiment['rescale_features'],
                                                   split_by_nodes=experiment['split_by_nodes'],
                                                   freeze_structural_model=experiment['freeze_embeddings'], 
                                                   early_stopping=early_stopping,
                                                   device=device, 
                                                   experiment_id=n, 
                                                   use_seed=experiment['use_seed'])
        result = metrics['test_f1']
        results.append(result)
        experiment_metrics.append(metrics)
        if model_output_path.strip():
            experiment_model_path = '{}.{}-{}.model'.format(model_output_path, experiment_hash, n)
            torch.save(model, experiment_model_path, pickle_module=dill)
            print('Finished experiment {} with result: {}. Storing model in {}.'.format(n, result, experiment_model_path))
        else:
            print('Finished experiment {} with result: {}.'.format(n, result))
    results_dict = {'config': experiment, 'results': results, 'metrics': experiment_metrics, 'stats': compute_stats(results), 'hash': experiment_hash}
    return results_dict


def run_and_save_experiment(experiment, 
                            num_attempts, 
                            data_path, 
                            data_prefix, 
                            output_path,
                            model_output_path,
                            move_to_cuda):
    results_dict = run_experiment(experiment, num_attempts, data_path, data_prefix, model_output_path, move_to_cuda)
    print(experiment, results_dict['results'], np.mean(results_dict['results']))
    with FileLock('{}.lock'.format(output_path), timeout=5):
        with open(output_path, 'a+') as f:
            results_json = json.dumps(results_dict)
            f.write('{}\n'.format(results_json))

def run_experiments_on_thread(experiments, 
                              num_attempts, 
                              data_path, 
                              data_prefix, 
                              output_path, 
                              model_output_path, 
                              move_to_cuda):
    for experiment in experiments:
        run_and_save_experiment(experiment, num_attempts, 
                                data_path, data_prefix, 
                                output_path, model_output_path,
                                move_to_cuda)


def run(num_workers, num_attempts, graph_path, graph_prefix, output_path, model_output_path, experimental_config, move_to_cuda):
    seen_experiments = load_finished_experiments(output_path, experimental_config)
    experiments = [e for e in generate_experiment_dicts(experimental_config)
                     if json.dumps(e) not in seen_experiments]
    random.shuffle(experiments)

    workers = []
    mp.freeze_support()
    for index in range(num_workers):
        arguments = (experiments[index::num_workers], num_attempts, graph_path, graph_prefix, output_path, model_output_path, move_to_cuda)
        proc = mp.Process(target=run_experiments_on_thread, args=arguments)
        proc.start()
        workers.append(proc)

    for w in workers:
        w.join()


def parse_arguments():
    main_args = argparse.ArgumentParser()
    main_args.add_argument('-w', '--num-workers', help='Number of workers to use when running the experiment, each running experiments in parallel.', type=int, default=NUM_WORKERS)
    main_args.add_argument('-n', '--num-attempts', help='Number of experiment attempts to run per experiment configuration.', type=int, default=NUM_EXPERIMENTS)
    main_args.add_argument('-g', '--graph-path', help='Path to the graph data files to be used.', type=str, default=GRAPH_PATH)
    main_args.add_argument('-p', '--graph-prefix', help='Prefix for the files to be used.', type=str, default=GRAPH_PREFIX)
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
    graph_prefix = args.graph_prefix
    output_path = args.output_path
    model_output_path = args.model_output_path
    experimental_config = json.load(open(args.experimental_config))
    move_to_cuda = args.cuda and torch.cuda.is_available()
    run(num_workers, num_attempts, graph_path, graph_prefix, output_path, model_output_path, experimental_config, move_to_cuda)

