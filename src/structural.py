import torch
from multiprocessing import cpu_count, Pool
from collections import Counter


def compute_degree_mapping_chunk(chunk):
    '''Chunk-based wrapper around compute_degree_mapping, to evaluate on chunks in parallel.'''
    return [compute_degree_mapping(idx, name, nbrs, G, use_d, max_deg) 
            for (idx, name, nbrs, G, use_d, max_deg) in chunk]


def compute_degree_mapping(node_index, node_name, neighbours, G, use_distances, max_degree):
    '''Degree mapping computation as a stateless function to call in parallel.'''
    G_n = G.induced_subgraph(neighbours)
    deg = [node_deg if node_deg < max_degree else max_degree for node_deg in G_n.degree()]
    if use_distances:
        sub_n = next(v for v in G_n.vs if v['name'] == node_name)
        deg_dist = zip((d[0] for d in G_n.shortest_paths_dijkstra(target=sub_n)), deg)
        deg = [dist * max_degree + deg for (dist, deg) in deg_dist]
    return node_index, list(zip(*Counter(deg).most_common()))


class StructuralMapper:
    def __init__(self, 
                 G,
                 distance=1,
                 use_distances=False,
                 num_workers=cpu_count(),
                 cache_field='neigh_deg'):
        self.distance = distance
        self.use_distances = use_distances
        self.cache_field = cache_field
        self.num_workers = num_workers
        self.max_degree = max(G.degree())
        self.source_G = G

    def num_elements(self):
        return (self.distance + 1) * self.max_degree

    def mapping_matrix(self, matrix_size):
        G = self.source_G
        total_elements = self.num_elements()
        matrix = torch.rand(total_elements + 1, matrix_size)
        total_structures = {value for mapping in self.mapping(G.vs, G) for value in mapping}

        # Zero out matrix entries for unseen matrices
        for deg in range(self.max_degree):
            for distance in range(self.distance):
                structure_index = distance * self.max_degree + deg if self.use_distances else deg
                if structure_index not in total_structures:
                    matrix[structure_index] = 0.0
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
                                                    self.max_degree)
                deg_seq.append(mapping)
        else:
            pool = Pool(processes=self.num_workers)
            as_tuples = [(n.index, n['name'], nbrs, G, self.use_distances, self.max_degree) 
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

