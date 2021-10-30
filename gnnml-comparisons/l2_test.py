import sys
sys.path.append('../src')

from collections import Counter

import igraph as ig
from structural import StructuralMapper

COSPECTRAL_EDGES = [(0, 1), (0, 3), (0, 5), (0, 7), (1, 2), (1, 3), (1, 4), (2, 4), (2, 6), (2, 9), (3, 5), (3, 7), (4, 6), (4, 9), (5, 8), (5, 9), (6, 7), (6, 8), (7, 8), (8, 9)]
FOUR_REG_EDGES = [(0, 1), (0, 3), (0, 6), (0, 7), (1, 2), (1, 3), (1, 4), (2, 4), (2, 5), (2, 9), (3, 5), (3, 7), (4, 6), (4, 9), (5, 7), (5, 8), (6, 8), (6, 9), (7, 8), (8, 9)]

PLOT_GRAPHS = False

c_G = ig.Graph()
c_G.add_vertices(10)
c_G.add_edges(COSPECTRAL_EDGES)
c_G.vs['name'] = [str(n.index) for n in c_G.vs]

f_G = ig.Graph()
f_G.add_vertices(10)
f_G.add_edges(FOUR_REG_EDGES)
f_G.vs['name'] = [str(n.index) for n in f_G.vs]

if PLOT_GRAPHS:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ig.plot(c_G, layout=c_G.layout("fr"), target=ax)
    plt.show()

    fig, ax = plt.subplots()
    ig.plot(f_G, layout=f_G.layout("fr"), target=ax)
    plt.show()
else:
    c_sm_1 = StructuralMapper(c_G, distance=1, use_distances=True, cache_field='neigh_deg_1', num_workers=1)
    f_sm_1 = StructuralMapper(f_G, distance=1, use_distances=True, cache_field='neigh_deg_1', num_workers=1)

    c_mapping_1 = Counter([tuple(sorted(zip(*x))) for x in c_sm_1.mapping(c_G.vs, c_G)])
    f_mapping_1 = Counter([tuple(sorted(zip(*x))) for x in f_sm_1.mapping(f_G.vs, f_G)])
    if c_mapping_1 != f_mapping_1:
        print('IGEL with encoding distance = 1 produces different mappings for Cospectral and Four-regular graphs.', '\n• Coespectral:\n\t', c_mapping_1, '\n• Four-Regular:\n\t', f_mapping_1)

    c_sm_2 = StructuralMapper(c_G, distance=2, use_distances=True, cache_field='neigh_deg_2', num_workers=1)
    f_sm_2 = StructuralMapper(f_G, distance=2, use_distances=True, cache_field='neigh_deg_2', num_workers=1)

    c_mapping_2 = Counter([tuple(sorted(zip(*x))) for x in c_sm_2.mapping(c_G.vs, c_G)])
    f_mapping_2 = Counter([tuple(sorted(zip(*x))) for x in f_sm_2.mapping(f_G.vs, f_G)])

    if c_mapping_2 != f_mapping_2:
        print('IGEL with encoding distance = 2 produces different mappings for Cospectral and Four-regular graphs.', '\n• Coespectral:\n\t', c_mapping_2, '\n• Four-Regular:\n\t', f_mapping_2)
