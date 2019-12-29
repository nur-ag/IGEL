import numpy as np


def lambda_node_sampler(node, neighbourhood, G, node_attribute):
    probs = np.asarray(neighbourhood[node_attribute], dtype=np.float32)
    return probs / probs.sum()


pagerank_sampler = lambda n, neigh, G: lambda_node_sampler(n, neigh, G, 'pagerank')
betweenness_sampler = lambda n, neigh, G: lambda_node_sampler(n, neigh, G, 'betweenness')
degree_sampler = lambda n, neigh, G: lambda_node_sampler(n, neigh, G, 'degree')
log_degree_sampler = lambda n, neigh, G: lambda_node_sampler(n, neigh, G, 'log_degree')
clust_coeff_sampler = lambda n, neigh, G: lambda_node_sampler(n, neigh, G, 'clust_coeff')

