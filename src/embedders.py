import math

import igraph as ig

import torch
import torch.nn as nn

from structural import StructuralMapper
from activations import Sparsemax


class NodeEmbedder(nn.Module):
    def __init__(self, data, node_key='id', requires_grad=True):
        super(NodeEmbedder, self).__init__()
        self.node_key = node_key
        self.matrix = nn.Parameter(data, requires_grad=requires_grad)

    def forward(self, node_seq, G):
        return self.matrix[node_seq[self.node_key]]


class StaticFeatureEmbedder(nn.Module):
    def __init__(self, feature_field, feature_tensor):
        super(StaticFeatureEmbedder, self).__init__()
        self.feature_field = feature_field
        self.feature_tensor = feature_tensor

    def forward(self, node_seq, G):
        indices = node_seq[self.feature_field]
        tensors = self.feature_tensor[indices]
        return tensors


def count_transform(counts, function):
    if function == 'identity':
        return counts
    if function == 'log':
        return [math.log(c + 1, 2) for c in counts]
    if function == 'sqrt':
        return [math.sqrt(c) for c in counts]
    if function == 'uniform':
        return [1 for c in counts]
    raise ValueError('Unknown counts transform "{}"'.format(function))


class SimpleStructuralEmbedder(nn.Module):
    def __init__(self, 
                 vector_size,
                 structural_mapper,
                 counts_transform='log',
                 device=torch.device('cpu')):
        super(SimpleStructuralEmbedder, self).__init__()
        self.structural_mapper = structural_mapper
        self.counts_transform = counts_transform
        self.device = device
        self.output_size = vector_size
        self.matrix = nn.Parameter(structural_mapper.mapping_matrix(vector_size), requires_grad=True).to(device)

    def forward(self, node_seq, G):
        indices, counts = zip(*self.structural_mapper.mapping(node_seq, G))

        # Apply the counts function
        counts = [count_transform(cnt, self.counts_transform) for cnt in counts]

        # Determine the torch package to use if we use CUDA or not
        _torch = torch.cuda if self.device.type == 'cuda' else torch

        # Prepare sparse vector coordinates
        coordinates = _torch.LongTensor([
            [i for i, idx in enumerate(indices) for _ in range(len(idx))],
            [i for idx in indices for i in idx]])
        values = _torch.FloatTensor([c for cnt in counts for c in cnt])

        # Create the sparse vector to multiply by the embedding matrix
        batch_size = len(indices)
        num_values = self.matrix.data.shape[0]
        structure_tensor = _torch.sparse.FloatTensor(coordinates, values, torch.Size([batch_size, num_values]))

        # Compute the total per-node degree counts to normalize 
        total_count_vector = torch.sparse.sum(structure_tensor, dim=1)

        # Sparse matrix multiplication to get the features off the embedding matrix and renormalize
        embed = torch.sparse.mm(structure_tensor, self.matrix)
        output = embed / total_count_vector._values().reshape(batch_size, 1) 
        return output


class GatedStructuralEmbedder(nn.Module):
    def __init__(self, 
                 vector_size,
                 output_size,
                 num_aggregations,
                 structural_mapper,
                 count_function='scale_norm',
                 aggregation_function='sum',
                 counts_transform='log',
                 device=torch.device('cpu')):
        super(GatedStructuralEmbedder, self).__init__()
        self.structural_mapper = structural_mapper
        self.counts_transform = counts_transform
        self.device = device
        self.num_aggregations = num_aggregations
        self.output_size = output_size
        self.count_function = count_function
        self.aggregation_function = aggregation_function
        self.gated = nn.GRUCell(vector_size, output_size).to(device)

        raw_matrix = structural_mapper.mapping_matrix(vector_size).to(device)
        self.matrix = nn.Parameter(raw_matrix, requires_grad=True)

        num_embeddings = len(raw_matrix)
        emb_layer = nn.Embedding(num_embeddings, vector_size)
        emb_layer.weight = self.matrix
        self.embedding = emb_layer.to(device)

    def compute_counts(self, tensor, counts_tensor):
        if self.count_function is None:
            return tensor
        if self.count_function == 'ignore':
            return tensor
        if self.count_function == 'scale':
            return tensor * counts_tensor
        if self.count_function == 'scale_norm':
            return tensor * (counts_tensor / counts_tensor.sum())
        raise ValueError('Invalid counts function "{}"!'.format(self.count_function))

    def compute_aggregation(self, tensor):
        if self.aggregation_function == 'mean':
            return tensor.mean(axis=0)
        if self.aggregation_function == 'max':
            return tensor.max(axis=0).values
        if self.aggregation_function == 'sum':
            return tensor.sum(axis=0)
        raise ValueError('Invalid aggregation function "{}"!'.format(self.aggregation_function))

    def forward(self, node_seq, G):
        if self.device.type == 'cuda' and \
           self.aggregation_function == 'sum' and \
           self.count_function == 'scale_norm':
           return self.forward_embedding(node_seq, G)

        outputs = []
        for indices, counts in self.structural_mapper.mapping(node_seq, G):
            counts = count_transform(counts, self.counts_transform)
            counts_tensor = torch.Tensor(counts).to(self.device).reshape(-1, 1)
            tensor = self.matrix[indices, :]
            indices_size = len(indices)
            hidden = torch.zeros(tensor.shape[-1])
            for i in range(self.num_aggregations):
                new_hidden = self.gated(tensor, hidden.reshape(1, -1).repeat(indices_size, 1))
                with_counts = self.compute_counts(new_hidden, counts_tensor)
                hidden = self.compute_aggregation(with_counts)
            outputs.append(hidden)
        outputs = torch.stack(outputs)
        return outputs

    def forward_embedding(self, node_seq, G):
        index_counts   = [(list(i), list(c)) for i, c in self.structural_mapper.mapping(node_seq, G)]
        num_elements   = len(index_counts)
        max_length     = max(len(i) for (i, _) in index_counts)
        indices_matrix = [i + [0] * (max_length - len(i)) for (i, _) in index_counts]
        counts_matrix  = [count_transform(c, self.counts_transform) + [0] * (max_length - len(c)) 
                          for (_, c) in index_counts]

        # Preliminary: just raw embedding and aggregating
        indices  = torch.LongTensor(indices_matrix).to(self.device)
        counts   = torch.FloatTensor(counts_matrix).to(self.device).reshape(num_elements, max_length, 1)
        counts   = counts / counts.sum(axis=1, keepdim=True)
        embedded = self.embedding(indices).reshape(num_elements * max_length, -1)
        hidden   = torch.zeros(num_elements, 1, self.matrix.shape[-1]).to(self.device)

        # Iteratively refine the representation through gating
        for i in range(self.num_aggregations):
            hidden_repeat = hidden.repeat(1, max_length, 1)
            hidden_flat   = hidden_repeat.reshape(num_elements * max_length, -1)
            hidden_update = self.gated(embedded, hidden_flat)
            hidden_broad  = hidden_update.reshape(num_elements, max_length, -1)
            hidden        = (hidden_broad * counts).sum(axis=1, keepdim=True)
        return hidden.reshape(num_elements, -1)


