import json
import hashlib
import numpy as np
from itertools import product

from parameters import DeepNNParameters, IGELParameters, NegativeSamplingParameters, TrainingParameters, AggregationParameters


UNSUPERVISED_PREFIX = 'unsupervised_'


def hash_dictionary(dictionary):
    as_string = json.dumps(dictionary)
    hashed = hashlib.md5(as_string.encode())
    return hashed.hexdigest()


def generate_experiment_tuples(experiment_dict):
    all_sets = [[(i, v) for v in values] 
                for (i, values) in sorted(experiment_dict.items())]
    return product(*all_sets)


def generate_experiment_dicts(experiment_dict):
    all_sets = [[(i, v) for v in values] 
                for (i, values) in sorted(experiment_dict.items())]
    return ({k: v for k, v in tuples} for tuples in product(*all_sets))


def tuple_to_dictionary(experiment_tuple):
    return {key: value for (key, value) in experiment_tuple}


def parse_negative_sampling_params(experiment_dict):
    ns_params = {'random_walk_length', 'window_size', 'negatives_per_positive'}
    ns_dict = {k: v for (k, v) in experiment_dict.items() if k in ns_params}
    if 'minimum_negative_distance' not in ns_dict:
        ns_dict['minimum_negative_distance'] = experiment_dict['encoding_distance']
    ns_opt = NegativeSamplingParameters(**ns_dict)
    return ns_opt


def parse_aggregation_params(experiment_dict):
    agg_params = {'node_dropouts', 'output_sizes', 'activations', 'aggregations', 'include_nodes', 'nodes_to_sample', 'num_attention_heads', 'attention_aggregators', 'attention_dropouts'}
    agg_dict = {k: v for (k, v) in experiment_dict.items() if k in agg_params}
    agg_options = AggregationParameters(**agg_dict)
    return agg_options


def parse_model_params(experiment_dict, ns_opt=None):
    model_params = {'encoding_distance', 'vector_length', 'model_type', 'use_distance_labels', 'gates_steps', 'counts_transform', 'counts_function', 'aggregator_function'}
    model_dict = {k: v for (k, v) in experiment_dict.items() if k in model_params}
    model_dict['gates_length'] = model_dict['vector_length']
    model_dict['neg_sampling_parameters'] = ns_opt
    model_options = IGELParameters(**model_dict)
    return model_options


def parse_training_params(experiment_dict):
    training_params = {'epochs', 'batch_size', 'learning_rate', 'problem_type', 'batch_samples_fn', 'display_epochs', 'weight_decay'}
    training_dict = {k: v for (k, v) in experiment_dict.items() if k in training_params}
    training_options = TrainingParameters(**training_dict)
    return training_options


def parse_inference_params(experiment_dict):
    inference_params = {'input_size', 'hidden_size', 'output_size', 'depth', 'activation'}
    inference_dict = {k: v for (k, v) in experiment_dict.items() if k in inference_params}
    inference_options = DeepNNParameters(**inference_dict)
    return inference_options


def create_experiment_params(experiment_dict):
    ns_opt = parse_negative_sampling_params(experiment_dict)
    training_options = parse_training_params(experiment_dict)
    model_options = parse_model_params(experiment_dict, ns_opt)
    aggregation_options = parse_aggregation_params(experiment_dict)

    unsupervised_dict = {k.replace(UNSUPERVISED_PREFIX, ''): v for k, v in experiment_dict.items() if k.startswith(UNSUPERVISED_PREFIX)}
    unsupervised_training_options = parse_training_params(unsupervised_dict)
    inference_options = parse_inference_params(experiment_dict)
    return model_options, training_options, unsupervised_training_options, aggregation_options, inference_options


def compute_stats(results):
    return {'min': np.min(results),
            'max': np.max(results),
            'std': np.std(results),
            'avg': np.mean(results)}

