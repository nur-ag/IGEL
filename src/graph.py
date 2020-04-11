import random
import igraph as ig


def load_graph(path, directed=False, weights=False):
    G = ig.Graph.Read_Ncol(path, names=True, weights=weights, directed=directed)
    G.vs['id'] = [int(n) for n in G.vs['name']]
    return G


def precompute_graph_features(G):
    G.vs['pagerank'] = G.pagerank()
    G.vs['betweenness'] = G.betweenness()
    G.vs['degree'] = G.degree()
    G.vs['log_degree'] = [np.log(d) + 1 for d in G.degree()]
    G.vs['clust_coeff'] = G.transitivity_local_undirected(mode='zero')
    return G


def sample_edges(G, percentage, connected=False, seed=None):
    new_n = int(round(len(G.es) * (1 - percentage)))
    if seed is not None:
        random.seed(seed)

    sub_G = G.copy()
    if not connected:
        edges = list(sub_G.es)
        random.shuffle(edges)
        sub_G.delete_edges([e for e in edges[:new_n]])
        return sub_G

    edges_count = 0
    seen_edges  = set()
    total_edges = len(G.es)
    while edges_count < new_n and len(seen_edges) < total_edges:
        e = random.choice(sub_G.es)
        attributes = e.attributes()
        e_1, e_2 = e.tuple
        if (e_1, e_2) in seen_edges or not sub_G.is_directed() and (e_2, e_1) in seen_edges:
            continue

        e.delete()
        reachable = False
        for x in sub_G.bfsiter(e_1):
            if x.index == e_2:
                reachable = True
                break

        seen_edges.add((e_1, e_2))
        if connected and not reachable and not sub_G.are_connected(e_1, e_2):
            sub_G.add_edge(e_1, e_2, **attributes)
        else:
            edges_count += 1
    if connected:
        assert len(G.components()) == len(sub_G.components())
    return sub_G


def edges_to_ids(G):
    for edge in G.es:
        na, nb = G.vs[edge.tuple]
        if G.is_directed():
            e_a, e_b = sorted([na['id'], nb['id']])
        else:
            e_a, e_b = na['id'], nb['id']
        yield (e_a, e_b)


def edge_difference(G, sub_G):
    sub_G_edges = {edge for edge in edges_to_ids(sub_G)}
    return [edge for edge in edges_to_ids(G) if edge not in sub_G_edges]


def generate_negative_edge(all_ids, real_edges):
    edge = next(iter(real_edges))
    while edge in real_edges:
        edge = (random.choice(all_ids), random.choice(all_ids))
    return edge


def generate_negative_edges(positive_edges, G, num_samples=1):
    all_ids = G.vs['id']
    all_edges = {edge for edge in edges_to_ids(G)}
    all_edges |= set(positive_edges)
    negative_edges = []
    num_samples = max(1, num_samples)
    for i in range(len(positive_edges) * num_samples):
        negative_edge = generate_negative_edge(all_ids, all_edges)
        negative_edges.append(negative_edge)
    return negative_edges

