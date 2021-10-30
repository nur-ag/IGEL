import sys
sys.path.append('../src')

from collections import Counter

import igraph as ig
from structural import StructuralMapper

DECALIN_EDGES = [(0, 1), (0, 2), (0, 6), (1, 5), (1, 9), (2, 3), (3, 4), (4, 5), (6, 7), (7, 8), (8, 9)]
BICYCLOPENTYL_EDGES = [(0, 1), (0, 2), (0, 5), (1, 6), (1, 9), (2, 3), (3, 4), (4, 5), (6, 7), (7, 8), (8, 9)]

PLOT_GRAPHS = False

d_G = ig.Graph()
d_G.add_vertices(10)
d_G.add_edges(DECALIN_EDGES)
d_G.vs['name'] = [str(n.index) for n in d_G.vs]

b_G = ig.Graph()
b_G.add_vertices(10)
b_G.add_edges(BICYCLOPENTYL_EDGES)
b_G.vs['name'] = [str(n.index) for n in b_G.vs]

if PLOT_GRAPHS:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ig.plot(d_G, layout=d_G.layout("fr"), target=ax)
    plt.show()

    fig, ax = plt.subplots()
    ig.plot(b_G, layout=b_G.layout("fr"), target=ax)
    plt.show()
else:
    d_sm_1 = StructuralMapper(d_G, distance=1, use_distances=True, cache_field='neigh_deg_1', num_workers=1)
    b_sm_1 = StructuralMapper(b_G, distance=1, use_distances=True, cache_field='neigh_deg_1', num_workers=1)

    d_mapping_1 = Counter([tuple(sorted(zip(*x))) for x in d_sm_1.mapping(d_G.vs, d_G)])
    b_mapping_1 = Counter([tuple(sorted(zip(*x))) for x in b_sm_1.mapping(b_G.vs, b_G)])
    if d_mapping_1 == b_mapping_1:
        print('IGEL with encoding distance = 1 produces equivalent mappings for Decalin and Bicyclopentyl.', '\n• Decalin:\n\t', d_mapping_1, '\n• Bicyclopentyl:\n\t', b_mapping_1)

    d_sm_2 = StructuralMapper(d_G, distance=2, use_distances=True, cache_field='neigh_deg_2', num_workers=1)
    b_sm_2 = StructuralMapper(b_G, distance=2, use_distances=True, cache_field='neigh_deg_2', num_workers=1)

    d_mapping_2 = Counter([tuple(sorted(zip(*x))) for x in d_sm_2.mapping(d_G.vs, d_G)])
    b_mapping_2 = Counter([tuple(sorted(zip(*x))) for x in b_sm_2.mapping(b_G.vs, b_G)])

    if d_mapping_2 != b_mapping_2:
        print('IGEL with encoding distance = 2 produces different mappings for Decalin and Bicyclopentyl.', '\n• Decalin:\n\t', d_mapping_2, '\n• Bicyclopentyl:\n\t', b_mapping_2)
