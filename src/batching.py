import math
import random


def chunks(source, chunk_size):
    current_chunk = []
    for value in source:
        current_chunk.append(value)
        if len(current_chunk) == chunk_size:
            yield current_chunk
            current_chunk = []
    if current_chunk:
        yield current_chunk


def index_samples(G, split_indices, batch_size):
    indices = sorted(split_indices, key=lambda n: n.index)
    indices = [n.index for n in indices]
    for batch in chunks(indices, batch_size):
        yield batch


def degree_sorted_samples(G, split_indices, batch_size, reverse=False):
    indices = sorted(split_indices, key=lambda n: n.degree(), reverse=reverse)
    indices = [n.index for n in indices]
    for batch in chunks(indices, batch_size):
        yield batch


def random_degree_weighted_samples(G, split_indices, batch_size, reverse=False):
    order = sorted(split_indices, key=lambda n: random.random() * math.log(n.degree()), reverse=reverse)
    indices = [n.index for n in order]
    for batch in chunks(indices, batch_size):
        yield batch


def uniformly_random_samples(G, split_indices, batch_size):
    indices = [n.index for n in split_indices]
    random.shuffle(indices)
    for batch in chunks(indices, batch_size):
        yield batch


def random_bfs_samples(G, split_indices, batch_size):
    indices = [n.index for n in split_indices]
    indices_in_split = set(indices)
    total_indices = len(G.vs)
    seen_indices = set()
    current_batch = []
    while len(seen_indices) < total_indices:
        indices = [node.index for node in split_indices
                   if node not in seen_indices]
        if not indices:
            break
        src = random.choice(indices)
        for node in G.bfsiter(src):
            new_index = node.index
            if new_index in seen_indices or new_index not in indices_in_split:
                continue
            seen_indices.add(new_index)
            current_batch.append(new_index)
            if len(current_batch) == batch_size:
                yield current_batch
                current_batch = []
    if current_batch:
        yield current_batch


def random_walk_samples(G, split_indices, batch_size):
    indices = [n.index for n in split_indices]
    indices_in_split = set(indices)
    total_indices = len(G.vs)
    current_batch = [random.choice(indices)]
    seen_indices = set(current_batch)
    while len(seen_indices) < total_indices:
        latest_neighs = G.neighbors(current_batch[-1])
        neighs_left = [n for n in latest_neighs if n not in seen_indices and n in indices_in_split]
        if len(current_batch) == batch_size:
            yield current_batch
            current_batch = []
        if not neighs_left or not current_batch:
            indices = [i for i in indices if i not in seen_indices]
            if not indices:
                break
            random_node = random.choice(indices)
            seen_indices.add(random_node)
            current_batch.append(random_node)
        else:
            random_neigh = random.choice(neighs_left)
            seen_indices.add(random_neigh)
            current_batch.append(random_neigh)
    if current_batch:
        yield current_batch


def random_walk(node, G, length):
    last_node = node
    seen_nodes = 1
    yield last_node
    while seen_nodes < length:
        next_choices = G.neighbors(last_node)
        if next_choices:
            last_node = random.choice(next_choices)
            seen_nodes += 1
            yield last_node
        else:
            break


def graph_random_walks(G, 
                       random_walk_length=80,
                       batch_size=512, 
                       node_sampling_fn=index_samples,
                       random_walk_fn=random_walk):
    node_generator = node_sampling_fn(G, G.vs, batch_size)
    for chunk in node_generator:
        for node in chunk:
            yield [n for n in random_walk_fn(node, G, random_walk_length)]


def negative_sampling_generator(G, 
                                random_walk_generator, 
                                window_size=10, 
                                negatives_per_positive=10):
    indices = [node.index for node in G.vs]
    for walk in random_walk_generator:
        for i, node in enumerate(walk):
            lower_start = max(0, i - window_size)
            upper_end = min(len(walk), i + window_size + 1)
            for index in range(lower_start, upper_end):
                if index == i:
                    continue
                yield node, walk[index], 1
                for n in range(negatives_per_positive):
                    yield node, random.choice(indices), 0


def negative_sampling_batcher(ns_generator, batch_size=512):
    source_list = []
    target_list = []
    labels = []
    for source, target, label in ns_generator:
        source_list.append(source)
        target_list.append(target)
        labels.append(label)
        if len(source_list) == batch_size:
            yield (source_list, target_list), labels
            source_list = []
            target_list = []
            labels = []
    if source_list:
        yield (source_list, target_list), labels
                

def negative_sampling_cacher(ns_batch_generator, batch_size=512, cache_scale=64):
    next_batches = [(([], []), []) for _ in range(cache_scale)]
    for (pairs, label) in ns_batch_generator:
        source, target = pairs
        for i, lbl in enumerate(label):
            index = i % cache_scale
            batch_tup, batch_label = next_batches[index]
            batch_src, batch_dst = batch_tup
            batch_src.append(source[i])
            batch_dst.append(target[i])
            batch_label.append(lbl)
            if len(batch_label) == batch_size:
                yield (batch_src, batch_dst), batch_label
                next_batches[index] = (([], []), [])
    for batch in next_batches:
        yield batch


batch_dictionary_mapping = {
    'index': index_samples,
    'degree': degree_sorted_samples,
    'degree_random': random_degree_weighted_samples,
    'uniform': uniformly_random_samples,
    'bfs': random_bfs_samples,
    'random_walk': random_walk_samples
}
