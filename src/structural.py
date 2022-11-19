import math
import json
import numpy as np
import torch
import torch.nn.init as init
from multiprocessing import cpu_count, Pool
from collections import Counter


def get_relative_degrees(node, dist_vector):
    '''Computes the relative degree of a node given a distance vector indexed by node indices.

    Relative degrees are the counts of edges from one node to neighbours that are:

    -1: one step closer,
     0: at the same distance, or
     1: one step further away

    from a third node towards which relative positioning is computed.
    '''
    outgoing_edge_indices = [i for e in node.all_edges() for i in e.tuple if i != node.index]
    distances = [dist_vector[index] - dist_vector[node.index] for index in outgoing_edge_indices]
    distance_freqs = {d: c for d, c in Counter(distances).most_common()}
    return [distance_freqs.get(i, 0) for i in [-1, 1]] + [node.degree()]


def compute_degree_mapping_chunk(chunk):
    '''Chunk-based wrapper around compute_degree_mapping, to evaluate on chunks in parallel.'''
    return [compute_degree_mapping(idx, name, nbrs, G, use_d, use_rel, max_deg_bnd) 
            for (idx, name, nbrs, G, use_d, use_rel, max_deg_bnd) in chunk]


def compute_degree_mapping(node_index, node_name, neighbours, G, use_distances, use_relative_degrees, max_degree_bound):
    '''Degree mapping computation as a stateless function to call in parallel.'''
    G_n = G.induced_subgraph(neighbours)
    sub_n = next(v for v in G_n.vs if v["name"] == node_name)
    if use_relative_degrees or use_distances:
        dist = [d[0] for d in G_n.shortest_paths_dijkstra(target=sub_n)]
    if use_relative_degrees:
        edge_degs = [get_relative_degrees(v, dist) for v in G_n.vs]
        encoding = [tuple(min(d, max_degree_bound) for d in rel_degs) for rel_degs in edge_degs]
    else:
        encoding = [(d,) for d in G_n.degree()]
    max_dist = 0
    if use_distances:
        max_dist = max(dist)
        encoding = [[dist * max_degree_bound + d for d in deg] for deg, dist in zip(encoding, dist)]
    max_encoding = max_degree_bound * (max_dist + 1)
    encoding = [[feat_index + i * max_encoding for i, feat_index in enumerate(enc)] for enc in encoding]
    return node_index, list(zip(*Counter([e for enc in encoding for e in enc]).most_common()))


class StructuralMapper:
    def __init__(self, 
                 G,
                 distance=1,
                 use_distances=False,
                 num_workers=cpu_count(),
                 cache_field='neigh_deg',
                 pack_as_arrays=False,
                 device=None,
                 use_relative_degrees=True):
        self.distance = distance
        self.use_distances = use_distances
        self.cache_field = cache_field
        self.num_workers = num_workers
        self.max_degree_bound = max(G.degree()) + 1
        self.source_G = G
        self.device = device
        self.pack_as_arrays = pack_as_arrays
        self.use_relative_degrees = use_relative_degrees

    def pack_mapping_as_array(self, mapping):
        degree, count = mapping
        return (np.asarray(degree, dtype=np.int32),
                np.asarray(count, dtype=np.int32))

    def num_elements(self):
        return (self.distance + 1) * self.max_degree_bound * (3 if self.use_relative_degrees else 1)

    def mapping_matrix(self, matrix_size):
        G = self.source_G
        total_elements = self.num_elements()
        matrix = torch.rand(total_elements, matrix_size, requires_grad=True)

        # using `.to(device)` didn't converge -- this is just slightly ugly
        if self.device.type == 'cuda':
            matrix = matrix.cuda()

        matrix = (matrix - 0.5) / math.sqrt(matrix_size)
        for i in range(1, total_elements):
            rel_index = i % ((self.distance + 1) * self.max_degree_bound)
            degree = (rel_index % self.max_degree_bound)
            distance = (rel_index // self.max_degree_bound)
            matrix[i] *= (math.exp(distance) / math.log(degree + 1)) if degree else 0.0
        matrix = (matrix - matrix.mean(axis=0)) / matrix.std(axis=0)
        return matrix

    def compute_mapping(self, node_seq, G):
        node_indices = [node.index for node in node_seq]
        neighbours_seq = G.neighborhood(node_indices, order=self.distance)
        
        if self.num_workers <= 1:
            deg_seq = []
            for node, neighbours in zip(node_seq, neighbours_seq):
                _, mapping = compute_degree_mapping(node.index, 
                                                    node['name'], 
                                                    neighbours, 
                                                    G, 
                                                    self.use_distances, 
                                                    self.use_relative_degrees,
                                                    self.max_degree_bound)
                if self.pack_as_arrays:
                    mapping = self.pack_mapping_as_array(mapping)
                deg_seq.append(mapping)
        else:
            pool = Pool(processes=self.num_workers)
            as_tuples = [(n.index, n['name'], nbrs, G, self.use_distances, self.use_relative_degrees, self.max_degree_bound) 
                         for (n, nbrs) in zip(node_seq, neighbours_seq)]
            chunks = [as_tuples[x::self.num_workers] for x in range(self.num_workers)]
            chunk_deg = pool.map(compute_degree_mapping_chunk, chunks)
            pool.close()

            # Put them all together
            chunk_res = []
            for chk in chunk_deg:
                chunk_res.extend(chk)

            # Map back to the original values
            per_node = {n: v for (n, v) in chunk_res}
            deg_seq = [per_node[node.index] for node in node_seq]
            if self.pack_as_arrays:
                deg_seq = [self.pack_mapping_as_array(mapping) for mapping in deg_seq]
        return deg_seq

    def mapping(self, node_seq, G):
        if self.cache_field not in G.vs.attribute_names():
            G.vs[self.cache_field] = [None for node in G.vs]

        # If the fields are defined for all, just return
        node_seq_data = node_seq[self.cache_field]
        if node_seq_data is not None and len(node_seq_data) and None not in node_seq_data:
            return node_seq_data

        unmapped = [node for node in node_seq if node[self.cache_field] is None]
        for node, deg in zip(unmapped, self.compute_mapping(unmapped, G)):
            node[self.cache_field] = deg

        return node_seq[self.cache_field]

    def cache_mapping(self, G, cache_path):
        node_mapping = self.mapping(G.vs, G)
        with open(cache_path, 'w') as f:
            for mapping in node_mapping:
                if self.pack_as_arrays:
                    degree, count = mapping
                    mapping = [mapping.tolist(), count.tolist()]
                json_mapping = json.dumps(mapping)
                f.write('{}\n'.format(json_mapping))

    def load_mapping(self, G, cache_path):
        mapping = []
        with open(cache_path, 'r') as f:
            for line in f:
                values = json.loads(line)
                if self.pack_as_arrays:
                    values = self.pack_mapping_as_array(values)
                mapping.append(values)
        G.vs[self.cache_field] = mapping
