from batching import batch_dictionary_mapping
from aggregators import attention_concat, attention_sum, attention_mean, attention_max, combine_sum, combine_mean, combine_max
import torch.nn as nn


ACTIVATIONS = {
    'elu': nn.ELU, 
    'relu': nn.ReLU, 
    'gelu': nn.GELU, 
    'relu6': nn.ReLU6, 
    'tanh': nn.Tanh, 
    'sigmoid': nn.Sigmoid, 
    'softmax': nn.Softmax,
    'linear': nn.Identity
}

ATTENTION_AGGREGATIONS = {
    'max': attention_max,
    'sum': attention_sum,
    'mean': attention_mean,
    'concat': attention_concat,
}

AGGREGATIONS = {
    'max': combine_max,
    'sum': combine_sum,
    'mean': combine_mean,
}


class DeepNNParameters():
    def __init__(self,
                 input_size=64, 
                 hidden_size=64, 
                 output_size=64, 
                 depth=1, 
                 activation='relu'):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.depth = depth
        self.activation = ACTIVATIONS.get(activation, activation)


class IGELParameters():
    def __init__(self,
                 model_type='simple',
                 vector_length=128,
                 encoding_distance=2,
                 use_distance_labels=True,
                 gates_length=0,
                 gates_steps=0,
                 counts_function='concat_both',
                 aggregator_function='mean',
                 counts_transform='log',
                 neg_sampling_parameters=None):
        self.model_type = model_type
        self.vector_length = vector_length
        self.encoding_distance = encoding_distance
        self.use_distance_labels = use_distance_labels
        self.gates_length = gates_length
        self.gates_steps = gates_steps
        self.counts_function = counts_function
        self.aggregator_function = aggregator_function
        self.neg_sampling_parameters = neg_sampling_parameters
        self.counts_transform = counts_transform


class AggregationParameters():
    def __init__(self, 
                 node_dropouts=[],         # [0.0, 0.0, 0.5], 
                 output_sizes=[],          # [20, 20, 6],
                 activations=[],           # ['elu', 'elu', 'elu'],
                 aggregations=[],          # ['sum', 'max', 'concat'],
                 include_nodes=[],         # [False, False, True],
                 nodes_to_sample=[],       # [0, 0, 5],
                 num_attention_heads=[],   # [0, 0, 6],
                 attention_aggregators=[], # ['concat', 'concat', 'concat'],
                 attention_dropouts=[]):   # [0.0, 0.0, 0.5]
        self.node_dropouts = node_dropouts
        self.output_sizes = output_sizes
        self.activations = [ACTIVATIONS.get(act, act) for act in activations]
        self.aggregations = [AGGREGATIONS.get(agg, agg) for agg in aggregations]
        self.include_nodes = include_nodes
        self.nodes_to_sample = nodes_to_sample
        self.num_attention_heads = num_attention_heads
        self.attention_aggregators = [ATTENTION_AGGREGATIONS.get(att_agg, att_agg) 
                                      for att_agg in attention_aggregators]
        self.attention_outputs_per_heads = [att_agg == attention_concat for att_agg in self.attention_aggregators]
        self.attention_dropouts = attention_dropouts


class TrainingParameters():
    def __init__(self,
                 batch_size=512,
                 learning_rate=0.001,
                 weight_decay=0.0,
                 epochs=1,
                 display_epochs=1,
                 batch_samples_fn='uniform',
                 problem_type='unsupervised'):
        if batch_samples_fn not in batch_dictionary_mapping:
            raise ValueError('Unknown batch sampling function "{}"!'.format(batch_samples_fn))
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.display_epochs = display_epochs
        self.batch_samples_fn = batch_dictionary_mapping[batch_samples_fn]
        self.problem_type = problem_type


class NegativeSamplingParameters():
    def __init__(self,
                 random_walk_length=80,
                 window_size=10,
                 negatives_per_positive=10,
                 minimum_negative_distance=2):
        self.random_walk_length = random_walk_length
        self.window_size = window_size
        self.negatives_per_positive = negatives_per_positive
        self.minimum_negative_distance = minimum_negative_distance
