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
                 transform_output=True,
                 count_function='concat_both',
                 aggregation_function='mean',
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
        self.matrix = nn.Parameter(structural_mapper.mapping_matrix(vector_size), requires_grad=True).to(device)
        extra_concat_units = count_function.startswith('concat') + count_function.endswith('both')
        self.gated = nn.GRUCell(vector_size + extra_concat_units, output_size).to(device)
        self.output_transform = nn.Linear(output_size, output_size).to(device) if transform_output else None

    def compute_counts(self, tensor, counts):
        if self.count_function is None:
            return tensor
        counts_tensor = torch.Tensor(counts).to(self.device).reshape(-1, 1)
        if self.count_function == 'scale':
            return tensor * counts_tensor
        if self.count_function == 'scale_norm':
            return tensor * (counts_tensor / counts_tensor.sum())
        if self.count_function == 'concat':
            return torch.cat([tensor, counts_tensor], axis=1)
        if self.count_function == 'concat_norm':
            return torch.cat([tensor, counts_tensor / counts_tensor.sum()], axis=1)
        if self.count_function == 'concat_both':
            return torch.cat([tensor, counts_tensor, counts_tensor / counts_tensor.sum()], axis=1)
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
        outputs = []
        for indices, counts in self.structural_mapper.mapping(node_seq, G):
            counts = count_transform(counts, self.counts_transform)
            tensor = self.compute_counts(self.matrix[indices, :], counts)
            indices_size = len(indices)
            hidden = torch.zeros(self.output_size).to(self.device)
            for i in range(max(1, self.num_aggregations)):
                new_hidden = self.gated(tensor, hidden.reshape(1, -1).repeat(indices_size, 1))
                hidden = self.compute_aggregation(new_hidden)
            outputs.append(hidden)
        outputs = torch.stack(outputs)
        if self.output_transform is not None:
            outputs = self.output_transform(outputs)
        return outputs

