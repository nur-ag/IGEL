import dill
import torch
import random


class EarlyStopping:
    def __init__(self, patience, file_path, minimum_change=0):
        self.patience = patience
        self.file_path = file_path
        self.minimum_change = minimum_change

        self.counter = 0
        self.best_loss = None
        self.stopped = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_loss - self.minimum_change:
            self.counter += 1
            self.stopped = self.counter >= self.patience
        else:
            self.best_loss = val_loss
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

