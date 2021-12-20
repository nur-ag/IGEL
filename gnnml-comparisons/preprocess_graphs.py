import sys
sys.path.append('../src')

from collections import defaultdict

import numpy as np
from time import time
import igraph as ig
from scipy.io import loadmat, savemat
from structural import StructuralMapper

DATASET_PATHS = ['../../gnn-matlang/dataset/Zinc/raw/zinc.mat', 
                 '../../gnn-matlang/dataset/enzymes/raw/enzymes.mat', 
                 '../../gnn-matlang/dataset/PTC/raw/ptc.mat', 
                 '../../gnn-matlang/dataset/subgraphcount/raw/randomgraph.mat', 
                 '../../gnn-matlang/dataset/mutag/raw/mutag.mat']

dataset = DATASET_PATHS[4]
m = loadmat(dataset)
if dataset.endswith('zinc.mat'):
    m = loadmat(dataset)
    key = 'E'
else:
    key = 'A'


def to_graph_mapping(mapping):
    as_index_count_tuples = map(lambda x: tuple(zip(*x)), mapping)
    sorted_tuples = sorted([x for l in as_index_count_tuples for x in l])
    result = defaultdict(int)
    for k, v in sorted_tuples:
        result[k] += v
    return tuple(sorted(result.items()))


def aggregate_embeddings(graph_ids, embeddings):
    result = {}
    for i, e in sorted(zip(graph_ids, embeddings.tolist())):
        if i not in result:
            result[i] = []
        result[i].append(e)
    return result


def apply_aggregation(agg_dict, operation):
    if operation == 'max':
        func = lambda x: np.max(x, axis=0)
    elif operation == 'sum':
        func = lambda x: np.sum(x, axis=0)
    elif operation == 'mean':
        func = lambda x: np.mean(x, axis=0)
    else:
        raise ValueError(f'Unknown operation: {operation}')
    return {k: func(v) for k, v in agg_dict.items()}


def sort_by_key(emb_dict):
    return np.asarray([v for k, v in sorted(emb_dict.items())])


def encode_and_embed(adjacencies, distance, seeds=5):
    mappings = []
    last = time()
    global_nodes = 0
    global_edges = []
    graph_ids = []
    for i, edges in enumerate(adjacencies):
        G = ig.Graph.Adjacency((edges > 0).tolist())
        G.vs['name'] = [str(node.index) for node in G.vs]
        sm = StructuralMapper(G, distance, use_distances=True, num_workers=1)
        mapping = sm.compute_mapping(G.vs, G)
        mappings.append(mapping)

        # Grow node count and collect edges
        edges = [tuple(map(lambda x: x + global_nodes, e.tuple)) for e in G.es]
        global_nodes += len(G.vs)
        global_edges.extend(edges)
        graph_ids.extend([i] * len(G.vs))
    print('Preparing global graph.')
    G = ig.Graph()
    G.add_vertices(global_nodes)
    G.add_edges(global_edges)
    G.vs['name'] = [str(n.index) for n in G.vs]
    print(f'Global graph is built. It has {len(G.vs)} nodes and {len(G.es)} edges.')
    new = time()
    num_graphs = len(adjacencies)
    print('Done: ', num_graphs, ' took ', new - last)
    graph_mappings = list(map(to_graph_mapping, mappings))
    print('Got:', len(set(graph_mappings)), '/', num_graphs, 'total mappings at distance:', distance)
    max_index = max([a for mapping in graph_mappings for a, _ in mapping])
    mean_length = np.mean(list(map(len, graph_mappings)))
    print('Max index:', max_index, ' Mean mapping length:', mean_length)

    print('Preparing to embed the graph-of-all-graphs')
    from igel_embedder import get_unsupervised_embeddings
    emb_max = []
    emb_sum = []
    emb_mean = []
    for seed in range(0, seeds):
        print(f'Running for seed {seed + 1}/{seeds}. ')
        embeddings = get_unsupervised_embeddings(G, distance, seed)
        embeddings = embeddings.reshape((-1, embeddings.shape[-1]))
        emb_agg = aggregate_embeddings(graph_ids, embeddings)
        emb_max.append(sort_by_key(apply_aggregation(emb_agg, 'max')))
        emb_sum.append(sort_by_key(apply_aggregation(emb_agg, 'sum')))
        emb_mean.append(sort_by_key(apply_aggregation(emb_agg, 'mean')))
    print('Building the feature matrix dictionary.')
    result = {
      'graph_mappings': graph_mappings,
      'embeddings': {
        'max': np.asarray(emb_max),
        'sum': np.asarray(emb_sum),
        'mean': np.asarray(emb_mean)
      }
    }
    return result

if __name__ == '__main__':
    adjacencies = m[key][0]
    for distance in range(1, 5):
        m[f'igel_{distance}'] = encode_and_embed(adjacencies, distance)
    savemat(dataset.replace('.mat', '.igel.mat'), m)
