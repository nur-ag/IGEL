import sys
from pprint import pformat
sys.path.append("../src")

from collections import Counter
import igraph as ig

import wl
from structural import StructuralMapper

# From: http://vrl.cs.brown.edu/color
COLORS = ["#399283", "#52dcbc", "#0e503e", "#a7cdd8", "#2c457d", "#9159de", "#7e80af", "#fd95e8", "#8c2e63", "#e9c9fa", "#fa41c7", "#8601ee", "#2580fe", "#c0e15c", "#458612", "#4be263"]

FILE_PREFIX = 'LongRefinementD1-3'

LONG_REFN_DEG3 = {
    0: [1],
    1: [0, 2, 3],
    2: [1, 11, 13],
    3: [1, 10, 12],
    4: [5, 7, 10],
    5: [4, 6, 10],
    6: [5, 9, 11],
    7: [4, 8, 11],
    8: [7, 9, 13],
    9: [6, 8, 12],
    10: [3, 4, 5],
    11: [2, 6, 7],
    12: [3, 9, 13],
    13: [2, 8, 12],
}

LONG_REFN_DEG5 = {
    0: [1],
    1: [0, 2, 3, 4, 5],
    2: [1, 3, 5, 7, 10],
    3: [1, 2, 4, 6, 10],
    4: [1, 3, 5, 9, 11],
    5: [1, 2, 4, 8, 11],
    6: [3, 7, 8, 9, 11],
    7: [2, 6, 8, 9, 10],
    8: [5, 6, 7, 10, 11],
    9: [4, 6, 7, 10, 11],
    10: [2, 3, 7, 8, 9],
    11: [4, 5, 6, 8, 9],
}

G_edgelists = LONG_REFN_DEG3

all_edges = [(s, d) for s, dests in G_edgelists.items() for d in dests if d > s]
G = ig.Graph(len(G_edgelists), edges=all_edges)
G.vs["name"] = [str(i) for i in range(len(G_edgelists))]
layout = G.layout(layout="davidson_harel")
print(f"Loaded graph with diameter {G.diameter()} and max degree {max(G.degree())}.")
ig.plot(G, vertex_label=G.vs["name"], layout=layout, target=f'{FILE_PREFIX}.pdf')

equiv_classes = wl.weisfeiler_lehman(G, verbose=True)
print(f"1-WL Equivalence classes: {equiv_classes}")
G.vs["color"] = [COLORS[eq % len(COLORS)] for eq in equiv_classes]
ig.plot(G, vertex_label=G.vs["name"], layout=layout, target=f'{FILE_PREFIX}-1WL.pdf')

prev_ctr, prev_enc = None, None
cum_enc = None
for alpha in range(1, G.diameter() + 1):
    igel = StructuralMapper(G, distance=alpha, use_distances=True, cache_field=f"neigh_deg_{alpha}", num_workers=1)
    igel_enc = [tuple(sorted(zip(*x))) for x in igel.mapping(G.vs, G)]
    igel_equiv = {k: i for i, k in enumerate(sorted(list(set(igel_enc))))}
    igel_equiv_classes = [igel_equiv[k] for k in igel_enc]
    G.vs["color"] = [COLORS[eq % len(COLORS)] for eq in igel_equiv_classes]
    igel_ctr = Counter(igel_enc)
    wl_matches_igel = "match" if wl.bijective_mapping(equiv_classes, igel_enc) else "do not match"
    print(f"For α = {alpha}:")
    print(f"The equivalence classes given by WL coloring {wl_matches_igel} IGEL encodings.")
    print(f"The IGEL fingerprint of the graph is: \n{pformat(igel_ctr)}")
    print()
    ig.plot(G, vertex_label=G.vs["name"], layout=layout, target=f'{FILE_PREFIX}-IGEL-{alpha}.pdf')
    cum_enc = igel_enc if cum_enc is None else [(t2, t1) for t1, t2 in zip(igel_enc, cum_enc)]
    prev_enc = igel_enc
    prev_ctr = igel_ctr

# Cumulative IGEL improves performance, but still does not detect all colors
cum_igel_equiv = {k: i for i, k in enumerate(sorted(list(set(cum_enc))))}
cum_igel_equiv_classes = [cum_igel_equiv[k] for k in cum_enc]
G.vs["color"] = [COLORS[eq % len(COLORS)] for eq in cum_igel_equiv_classes]
cum_igel_ctr = Counter(cum_enc)
wl_matches_cum_igel = "match" if wl.bijective_mapping(equiv_classes, cum_enc) else "do not match"
print(f"Cumulatively for α in {{1..{alpha}}}:")
print(f"The equivalence classes given by WL coloring {wl_matches_igel} cumulative IGEL encodings.")
print(f"The cumulative IGEL fingerprint of the graph is: \n{pformat(cum_igel_ctr)}")
print()

ig.plot(G, vertex_label=G.vs["name"], layout=layout, target=f'{FILE_PREFIX}-IGEL-Cumulative-{alpha}.pdf')
