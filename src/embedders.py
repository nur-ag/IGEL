import igraph as ig

import torch
import torch.nn as nn
from collections import Counter


class NodeEmbedder(nn.Module):
    def __init__(self, data, requires_grad=True):
        super(NodeEmbedder, self).__init__()
        self.matrix = nn.Parameter(data, requires_grad=requires_grad)

    def forward(self, node_seq, G):
        return self.matrix[node_seq['id']]


class StaticFeatureEmbedder(nn.Module):
    def __init__(self, feature_field, feature_tensor):
        super(StaticFeatureEmbedder, self).__init__()
        self.feature_field = feature_field
        self.feature_tensor = feature_tensor

    def forward(self, node_seq, G):
        indices = node_seq[self.feature_field]
        tensors = self.feature_tensor[indices]
        return tensors


class StructuralEmbedder(nn.Module):
    def __init__(self, 
                 G,
                 num_features,  
                 distance=1,
                 use_distances=False,
                 cache_field='neigh_deg'):
        super(StructuralEmbedder, self).__init__()
        self.distance = distance
        self.use_distances = use_distances
        self.cache_field = cache_field
        self.max_degree = max(G.degree())
        
        total_elements = (distance + 1) * self.max_degree
        structural_data = torch.rand(total_elements + 1, num_features)

        # Degrees that do not appear in the graph are set to 0 
        # to avoid random values for unseen graphs
        valid_degrees = {d for d in G.degree()}
        unseen_degrees = [d for d in range(1, self.max_degree + 1) if d not in valid_degrees]
        zero_degrees = [r * self.max_degree + d for d in unseen_degrees for r in range(distance + 1)]
        structural_data[zero_degrees] = 0.0
        self.matrix = nn.Parameter(structural_data)

    def compute_mapping(self, node_seq, G):
        node_indices = [node.index for node in node_seq]
        neighbours_seq = G.neighborhood(node_indices, order=self.distance)
        deg_seq = []
        for node, neighbours in zip(node_seq, neighbours_seq):
            G_n = G.induced_subgraph(neighbours)
            deg = G_n.degree()
            if self.use_distances:
                sub_n = next(v for v in G_n.vs if v['name'] == node['name'])
                deg_dist = zip((d[0] for d in G_n.shortest_paths_dijkstra(target=sub_n)), deg)
                deg = [dist * self.max_degree + deg for (dist, deg) in deg_dist]
            deg = list(zip(*Counter(deg).most_common()))
            deg_seq.append(deg)
        return deg_seq

    def mapping(self, node_seq, G):
        if self.cache_field not in G.vs.attribute_names():
            G.vs[self.cache_field] = [None for node in G.vs]

        if node_seq[self.cache_field] and None not in node_seq[self.cache_field]:
            return node_seq[self.cache_field]

        unmapped = [node for node in node_seq if node[self.cache_field] is None]
        for node, deg in zip(unmapped, self.compute_mapping(unmapped, G)):
            node[self.cache_field] = deg

        return node_seq[self.cache_field]

    def forward(self, node_seq, G):
        indices, counts = zip(*self.mapping(node_seq, G))
        coordinates = torch.LongTensor([
            [i for i, idx in enumerate(indices) for _ in range(len(idx))],
            [i for idx in indices for i in idx]])
        values = torch.FloatTensor([c for cnt in counts for c in cnt])

        # Create the sparse vector to multiply by the embedding matrix
        batch_size = len(indices)
        num_values = self.matrix.data.shape[0]
        structure_tensor = torch.sparse.FloatTensor(coordinates, values, torch.Size([batch_size, num_values]))

        # Compute the total per-node degree counts to normalize 
        total_count_vector = torch.sparse.sum(structure_tensor, dim=1).to_dense().reshape(batch_size, 1) 

        # Sparse matrix multiplication to get the features off the embedding matrix and renormalize
        embed = torch.sparse.mm(structure_tensor, self.matrix)
        output = embed / total_count_vector
        return output
        
