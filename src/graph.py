import math
import random
import igraph as ig


def load_graph(path, directed=False, weights=False):
    G = ig.Graph.Read_Ncol(path, names=True, weights=weights, directed=directed)
    G.vs['id'] = [int(n) for n in G.vs['name']]
    return G


def precompute_graph_features(G):
    G.vs['pagerank'] = G.pagerank()
    G.vs['betweenness'] = G.betweenness()
    G.vs['closeness'] = G.closeness()
    G.vs['eigen'] = G.evcent()
    G.vs['degree'] = G.degree()
    G.vs['log_degree'] = [math.log(d) + 1 for d in G.degree()]
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


def clone_bridge_graph(G, num_copies=1, bridges_with_copies=1, link_copies=False):
    total_nodes = len(G.vs)
    ids = G.vs['id'] * (num_copies + 1)
    ids_remapped = [num_id + (i // total_nodes) * total_nodes for i, num_id in enumerate(ids)]
    names = G.vs['name'] * (num_copies + 1)

    # Clone the graph with all the attributes
    G = G.copy()
    G.add_vertices(total_nodes * num_copies)
    G.vs['id'] = ids_remapped
    G.vs['id_original'] = ids
    G.vs['name'] = names

    # Clone the corresponding edges
    G.add_edges([(src + copy * total_nodes, dst + copy * total_nodes) 
                        for (src, dst) in G.get_edgelist() 
                        for copy in range(1, num_copies + 1)])

    # Create all the bridges
    if bridges_with_copies > 0:
        bridge_edges = []
        for copy_index in range(num_copies):
            start_shifts = 1
            if link_copies:
                start_shifts = num_copies + 1
            for graph_index in range(start_shifts):
                if graph_index > copy_index:
                    break
                src_shift = graph_index * total_nodes
                dst_shift = (1 + copy_index) * total_nodes
                for bridge in range(bridges_with_copies):
                    src_index = random.randint(0, total_nodes - 1) + src_shift
                    dst_index = random.randint(0, total_nodes - 1) + dst_shift
                    bridge_edges.append((src_index, dst_index))
        G.add_edges(bridge_edges)
    return G

