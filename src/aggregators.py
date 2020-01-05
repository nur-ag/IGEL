import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


def attention_concat(tensor):
    '''Computes the concatenation of the attention masked tensor'''
    return tensor.reshape(tensor.shape[0], -1)


def attention_sum(tensor):
    '''Computes the sum of the attention masks tensor'''
    return tensor.sum(dim=1)


def combine_sum(tensor):
    '''Computes the sum of the node neighbourhood tensor'''
    return tensor.sum(dim=0)


def combine_mean(tensor):
    '''Computes the mean of the node neighbourhood tensor'''
    return tensor.mean(dim=0)


class MultiEmbedderAggregator(nn.Module):
    def __init__(self, embedders, aggregation=lambda x: torch.cat(x, -1)):
        super(MultiEmbedderAggregator, self).__init__()
        self.embedders = embedders
        self.aggregation = aggregation

    def forward(self, node_seq, G):
        outputs = [e(node_seq, G) for e in self.embedders]
        merged = self.aggregation(outputs)
        return merged


class SamplingAggregator(nn.Module):
    def __init__(self, 
                 embedder, 
                 input_units, 
                 hidden_units=100, 
                 output_units=20, 
                 num_hidden=1, 
                 activation=nn.ELU,
                 aggregation=combine_sum,
                 include_node=False,
                 nodes_to_sample=0,
                 sample_with_replacement=True,
                 sampling_model=None,
                 attend_over_dense=True,
                 num_attention_heads=5,
                 attention_activation=nn.ELU,
                 attention_aggregator=attention_concat,
                 attention_max=nn.Softmax):
        super(SamplingAggregator, self).__init__()

        self.embedder = embedder
        self.aggregation = aggregation
        self.input_units = input_units
        self.output_units = output_units if output_units > 0 else input_units
        self.hidden_units = hidden_units if num_hidden > 0 else self.output_units
        self.num_hidden = num_hidden
        self.include_node = include_node
        self.nodes_to_sample = nodes_to_sample
        self.sample_with_replacement = sample_with_replacement
        self.sampling_model = sampling_model
        self.attend_over_dense = attend_over_dense

        # Normal DNN layers over the source
        layers = []
        if output_units > 0:
            layers.append(nn.Linear(self.input_units * 2, self.hidden_units))
            layers.append(activation())

        for i in range(num_hidden - 1):
            layers.append(nn.Linear(self.hidden_units, self.hidden_units))
            layers.append(activation())

        if num_hidden > 0:
            layers.append(nn.Linear(self.hidden_units, self.output_units))
            layers.append(activation())

        self.dense = nn.Sequential(*layers)

        # Attention over the inputs to compute
        self.num_attention_heads = num_attention_heads 
        attention_heads = None
        attention_input = self.output_units if self.attend_over_dense else self.input_units * 2
        if num_attention_heads > 0:
            attention_heads = nn.Sequential(nn.Linear(attention_input, self.num_attention_heads), attention_activation())
        self.attention_heads = attention_heads
        self.attention_aggregator = attention_aggregator
        self.attention_max = attention_max(dim=1)

    def sample_neighbourhood(self, node, G):
        neighbour_indices = G.neighbors(node)
        neighbours = G.vs[neighbour_indices]
        if self.nodes_to_sample == 0:
            return neighbours

        if len(neighbours) < self.nodes_to_sample and not self.sample_with_replacement:
            return neighbours
            
        sample_weights = self.compute_sample_weights(node, neighbours, G)
        sampled_neighbours = np.random.choice(neighbours, 
                                              self.nodes_to_sample, 
                                              self.sample_with_replacement, 
                                              sample_weights)
        return sampled_neighbours

    def compute_sample_weights(self, node, neighbours, G):
        if self.sampling_model is None:
            return None
        probabilities = self.sampling_model(node, neighbours, G)
        return np.asarray(probabilities)

    def compute_attention(self, attention, tensors):
        heads = self.num_attention_heads
        nodes = len(tensors)
        nouts = self.output_units

        attention_prb = self.attention_max(attention)
        attention_rep = attention_prb.reshape(heads, nodes, 1)
        tensors_rep = tensors.repeat(heads, 1, 1)
        tensors_att = attention_rep * tensors_rep
        tensors_tsp = tensors_att.transpose(0, 1)
        return self.attention_aggregator(tensors_tsp)

    def get_node_set(self, node_seq, G):
        node_set = {node.index for node in node_seq}
        node_neighbours = {node.index: self.sample_neighbourhood(node, G) for node in node_seq}
        neighbours_set = {node.index for neighbours in node_neighbours.values() for node in neighbours}
        complete_set = (node_set | neighbours_set)
        return complete_set, node_neighbours

    def forward(self, node_seq, G):
        complete_set, node_neighbours = self.get_node_set(node_seq, G)

        # Compute tensors for the whole batch and find mappings
        tensors = self.embedder(G.vs[complete_set], G)
        node_mappings = {n: i for i, n in enumerate(complete_set)}

        # Compute neighbourhood aggregation on every node in the input batch
        node_tensors = []
        for node_index, neighbourhood in node_neighbours.items():
            neigh_indices = [[node_mappings[neigh.index], node_mappings[node_index]]
                                for neigh in neighbourhood]
            neigh_tensors = tensors[neigh_indices, :]
            neigh_tensors = neigh_tensors.reshape(len(neighbourhood), self.input_units * 2)

            # Compute dense and attention layers
            tensor = self.dense(neigh_tensors)
            if self.attention_heads:
                attention_input = tensor if self.attend_over_dense else neigh_tensors
                attention = self.attention_heads(attention_input)
                tensor = self.compute_attention(attention, tensor)

            # At this point, we aggregate over a tensor for the node with shape:
            # (num_neighbours, output_units * min(1, num_attention_heads))
            node_output = self.aggregation(tensor)
            if self.include_node:
                node_output = torch.cat([node_output, tensors[node_mappings[node_index]]])
            node_tensors.append(node_output)
        return torch.stack(node_tensors)

