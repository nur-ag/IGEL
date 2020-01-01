import dill
import torch
import numpy as np


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


