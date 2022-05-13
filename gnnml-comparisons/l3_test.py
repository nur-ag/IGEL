import sys
sys.path.append('../src')

from collections import Counter

import igraph as ig
from wl import weisfeiler_lehman
from structural import StructuralMapper

ROOK_EDGES = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 8), (0, 12), (1, 2), (1, 3), (1, 5), (1, 9), (1, 13), (2, 3), (2, 6), (2, 10), (2, 14), (3, 7), (3, 11), (3, 15), (4, 5), (4, 6), (4, 7), (4, 8), (4, 12), (5, 6), (5, 7), (5, 9), (5, 13), (6, 7), (6, 10), (6, 14), (7, 11), (7, 15), (8, 9), (8, 10,), (8, 11), (8, 12), (9, 10), (9, 11), (9, 13), (10, 11), (10, 14), (11, 15), (12, 13), (12, 14), (12, 15), (13, 14), (13, 15), (14, 15)]
SHRIKHANDE_EDGES = [(0, 1), (0, 3), (0, 4), (0, 5), (0, 12), (0, 15), (1, 2), (1, 5), (1, 6), (1, 12), (1, 13), (2, 3), (2, 6), (2, 7), (2, 13), (2, 14), (3, 4), (3, 7), (3, 14), (3, 15), (4, 5), (4, 7), (4, 8), (4, 9), (5, 6), (5, 9), (5, 10), (6, 7), (6, 10), (6, 11), (7, 8), (7, 11), (8, 9), (8, 11), (8, 12), (8, 13), (9, 10), (9, 13), (9, 14), (10, 11), (10, 14), (10, 15), (11, 12), (11, 15), (12, 13), (12, 15), (13, 14), (14, 15)]

PLOT_GRAPHS = False

r_G = ig.Graph()
r_G.add_vertices(16)
r_G.add_edges(ROOK_EDGES)
r_G.vs['name'] = [str(n.index) for n in r_G.vs]

s_G = ig.Graph()
s_G.add_vertices(16)
s_G.add_edges(SHRIKHANDE_EDGES)
s_G.vs['name'] = [str(n.index) for n in s_G.vs]

if PLOT_GRAPHS:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ig.plot(r_G, layout=r_G.layout("fr"), target=ax)
    plt.show()

    fig, ax = plt.subplots()
    ig.plot(s_G, layout=s_G.layout("fr"), target=ax)
    plt.show()
else:
    print(f'The maximum degree of Rook is {max(r_G.degree())} while for Shrikhande it is {max(s_G.degree())}.')
    wl_1, wl_2 = weisfeiler_lehman(r_G), weisfeiler_lehman(s_G)
    print(f'The 1-WL coloring of Rook is {wl_1} while for Shrikhande is {wl_2}, equal despite not being isomorphic.')
    r_sm_1 = StructuralMapper(r_G, distance=1, use_distances=True, cache_field='neigh_deg_1', num_workers=1)
    s_sm_1 = StructuralMapper(s_G, distance=1, use_distances=True, cache_field='neigh_deg_1', num_workers=1)

    r_mapping_1 = Counter([tuple(sorted(zip(*x))) for x in r_sm_1.mapping(r_G.vs, r_G)])
    s_mapping_1 = Counter([tuple(sorted(zip(*x))) for x in s_sm_1.mapping(s_G.vs, s_G)])
    if r_mapping_1 == s_mapping_1: 
        print('IGEL with encoding distance = 1 produces equivalent mappings for 4x4 Rook and Shrikhande graphs.', '\n• Rook:\n\t', r_mapping_1, '\n• Shrikhande:\n\t', s_mapping_1)

    r_sm_2 = StructuralMapper(r_G, distance=2, use_distances=True, cache_field='neigh_deg_2', num_workers=1)
    s_sm_2 = StructuralMapper(s_G, distance=2, use_distances=True, cache_field='neigh_deg_2', num_workers=1)

    r_mapping_2 = Counter([tuple(sorted(zip(*x))) for x in r_sm_2.mapping(r_G.vs, r_G)])
    s_mapping_2 = Counter([tuple(sorted(zip(*x))) for x in s_sm_2.mapping(s_G.vs, s_G)])

    if r_mapping_2 == s_mapping_2: 
        print('IGEL with encoding distance = 2 produces equivalent mappings for 4x4 Rook and Shrikhande graphs.', '\n• Rook:\n\t', r_mapping_2, '\n• Shrikhande:\n\t', s_mapping_2)

    r_sm_3 = StructuralMapper(r_G, distance=3, use_distances=True, cache_field='neigh_deg_3', num_workers=1)
    s_sm_3 = StructuralMapper(s_G, distance=3, use_distances=True, cache_field='neigh_deg_3', num_workers=1)

    r_mapping_3 = Counter([tuple(sorted(zip(*x))) for x in r_sm_3.mapping(r_G.vs, r_G)])
    s_mapping_3 = Counter([tuple(sorted(zip(*x))) for x in s_sm_3.mapping(s_G.vs, s_G)])

    if r_mapping_3 == s_mapping_3: 
        print('IGEL with encoding distance = 3 produces equivalent mappings for 4x4 Rook and Shrikhande graphs.', '\n• Rook:\n\t', r_mapping_3, '\n• Shrikhande:\n\t', s_mapping_3)

    r_sm_4 = StructuralMapper(r_G, distance=4, use_distances=True, cache_field='neigh_deg_4', num_workers=1)
    s_sm_4 = StructuralMapper(s_G, distance=4, use_distances=True, cache_field='neigh_deg_4', num_workers=1)

    r_mapping_4 = Counter([tuple(sorted(zip(*x))) for x in r_sm_4.mapping(r_G.vs, r_G)])
    s_mapping_4 = Counter([tuple(sorted(zip(*x))) for x in s_sm_4.mapping(s_G.vs, s_G)])

    if r_mapping_4 == s_mapping_4: 
        print('IGEL with encoding distance = 4 produces equivalent mappings for 4x4 Rook and Shrikhande graphs.', '\n• Rook:\n\t', r_mapping_4, '\n• Shrikhande:\n\t', s_mapping_4)
    