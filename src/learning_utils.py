import dill
import torch
import random


class EarlyStopping:
    def __init__(self, patience, file_path, minimum_change=0, metric_sign=1):
        self.patience = patience
        self.file_path = file_path
        self.minimum_change = minimum_change
        self.metric_sign = metric_sign
        
        self.counter = 0
        self.best_metric = None
        self.stopped = False

    def __call__(self, metric, model):
        if self.best_metric is None:
            self.best_metric = metric
            self.save_checkpoint(model)
        elif self.metric_sign * metric > self.metric_sign * (self.best_metric - self.minimum_change):
            self.counter += 1
            self.stopped = self.counter >= self.patience
        else:
            self.best_metric = metric
            self.save_checkpoint(model)
            self.counter = 0
        return self.stopped

    def save_checkpoint(self, model):
        torch.save(model, self.file_path, pickle_module=dill)


def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i+n]


def uniformly_random_samples(G, batch_size):
    indices = [node.index for node in G.vs]
    random.shuffle(indices)
    for batch in chunks(indices, batch_size):
        yield batch


def random_bfs_samples(G, batch_size):
    total_indices = len(G.vs)
    seen_indices = set()
    current_batch = []
    while len(seen_indices) < total_indices:
        indices = [node.index for node in G.vs 
                   if node.index not in seen_indices]
        src = random.choice(indices)
        for node in G.bfsiter(src):
            new_index = node.index
            if new_index in seen_indices:
                continue
            seen_indices.add(new_index)
            current_batch.append(new_index)
            if len(current_batch) == batch_size:
                yield current_batch
                current_batch = []
    if current_batch:
        yield current_batch


def random_walk_samples(G, batch_size):
    total_indices = len(G.vs)
    indices = [node.index for node in G.vs]
    current_batch = [random.choice(indices)]
    seen_indices = set(current_batch)
    while len(seen_indices) < total_indices:
        latest_neighs = G.neighbors(current_batch[-1])
        neighs_left = [n for n in latest_neighs if n not in seen_indices]
        if len(current_batch) == batch_size:
            yield current_batch
            current_batch = []
        if not neighs_left or not current_batch:
            indices = [i for i in indices if i not in seen_indices]
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
        last_node = random.choice(next_choices)
        seen_nodes += 1
        yield last_node


def graph_random_walks(G, 
                       random_walk_length=80,
                       batch_size=512, 
                       node_sampling_fn=uniformly_random_samples,
                       random_walk_fn=random_walk):
    node_generator = node_sampling_fn(G, batch_size)
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
                yield (node, walk[index], 1)
                for n in range(negatives_per_positive):
                    yield (node, random.choice(indices), 0)

def negative_sampling_batcher(ns_generator, batch_size=16384):
    source_list = []
    target_list = []
    labels = []
    for source, target, label in ns_generator:
        source_list.append(source)
        target_list.append(target)
        labels.append(label)
        if len(source_list) == batch_size:
            yield source_list, target_list, labels
            source_list = []
            target_list = []
            labels = []
    yield source_list, target_list, labels    

