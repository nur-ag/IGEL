import dill
import torch

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from tqdm import tqdm, trange
from time import time


class InterruptTraining(StopIteration):
    pass


class EarlyStopping:
    def __init__(self, patience, file_path, metric='valid_f1', minimum_change=0, metric_sign=1):
        self.patience = patience
        self.file_path = file_path
        self.metric = metric
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


class GraphNetworkTrainer():
    def __init__(self, model, optimizer, criterion,
                epoch_metrics=['valid_f1', 'valid_loss'],
                display_epochs=3, early_stopping=None,
                scheduler=None, problem_type='multiclass'):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

        self.epoch_metrics = epoch_metrics
        self.display_epochs = display_epochs
        self.early_stopping = early_stopping
        self.scheduler = scheduler
        self.problem_type = problem_type

        # Internal variables used during training
        self._clear_internal_variables()

    def _clear_internal_variables(self):
        self._batch_loss = None
        self._batches_bar = None
        self._current_batch = None
        self._current_loss = None
        self._epoch_bar = None
        self._epoch_start = None
        self._epoch_end = None
        self._epoch_losses = []
        self._epoch_time = None
        self._num_epochs = None
        self._history = []
        self._validation_data = None
        self._test_data = None
        self._test_metrics = None
        return self

    def fit(self, batch_iterator_fn, G, num_epochs, validation_data=None, test_data=None):
        self._clear_internal_variables()
        self._validation_data = validation_data
        self._test_data = test_data
        self.before_training()
        self._num_epochs = num_epochs
        self._epoch_bar = trange(1, num_epochs + 1, desc='Epoch')
        try:
            for epoch in self._epoch_bar:
                self._current_epoch = epoch
                batch_iterator = batch_iterator_fn()
                self._current_loss = self._fit_epoch(batch_iterator, G)
        except InterruptTraining as e:
            self.after_training_interrupted()
        self._epoch_bar = None
        self.after_training()
        return self

    def _fit_epoch(self, batch_iterator, G):
        self.before_epoch()
        self._epoch_losses = []
        self._batches_bar = tqdm(batch_iterator, desc='Batch')
        for batch in self._batches_bar:
            self._current_batch = batch
            loss = self._fit_batch(batch, G)
            self._epoch_losses.append(loss)
        self._batches_bar = None
        self.after_epoch()
        return np.mean(self._epoch_losses)

    def _fit_batch(self, batch, G):
        self.before_batch()
        self.optimizer.zero_grad()
        node_seq, labels = batch
        output = self.model(node_seq, G)
        loss = self.criterion(output, labels)
        loss.backward()
        if self.scheduler is not None:
            self.scheduler.step(loss)
        self.optimizer.step()
        self._batch_loss = loss.data.detach().mean().item()
        self.after_batch()
        return self._batch_loss

    def predict_raw(self, node_seq, G):
        with torch.no_grad():
            self.model = self.model.eval()
            pred_raw = self.model(node_seq, G)
            return pred_raw

    def predict_proba(self, node_seq, G):
        num_instances = len(node_seq)
        pred_raw = self.predict_raw(node_seq, G)
        pred_proba = torch.sigmoid(pred_raw).detach().reshape(num_instances, -1).cpu().numpy()
        return pred_proba

    def predict(self, node_seq, G):
        return self.predict_proba(node_seq, G)

    def compute_metrics(self, split, node_seq, labels, G):
        num_instances = len(node_seq)
        pred = self.predict(node_seq, G)
        pred_raw = self.predict_raw(node_seq, G)
        if self.problem_type == 'multiclass':
            split_pred = pred_raw.argmax(axis=1).detach().reshape(num_instances, -1).cpu().numpy().round()
        else:
            split_pred = torch.sigmoid(pred_raw).detach().reshape(num_instances, -1).cpu().numpy().round()
        split_true = labels.reshape(num_instances, -1).cpu().numpy()

        accuracy = accuracy_score(split_pred, split_true)
        values = precision_recall_fscore_support(split_pred, split_true, average='micro')
        prec, recall, f1, _ = values
        loss = self.criterion(pred_raw, labels)
        split_loss = np.mean(loss.data.mean().tolist())
        return {'{}_accuracy'.format(split): accuracy,
            '{}_precision'.format(split): prec,
            '{}_recall'.format(split): recall,
            '{}_f1'.format(split): f1,
            '{}_loss'.format(split): split_loss}

    def before_training(self):
        pass

    def after_training_interrupted(self):
        pass

    def after_training(self):
        if self.early_stopping is not None and self._test_data is not None:
            self.model = torch.load(self.early_stopping.file_path, pickle_module=dill).eval()
            print('Loading the best model to evaluate on the test set.')
            self._test_metrics = self.compute_metrics('test', *self._test_data)
            print('Best model results on the test set: \n')
            for metric, value in self._test_metrics.items():
                print('{}: {:.3f}'.format(metric, value))

    def before_epoch(self):
        self._epoch_start = time()
        self.model = self.model.train()

    def after_epoch(self):
        self._epoch_end = time()
        self._epoch_time = self._epoch_end - self._epoch_start

        # Basic information display
        epoch = self._current_epoch
        epoch_time = self._epoch_time
        current_loss = np.mean(self._epoch_losses)
        tqdm.write('Epoch {} took {:.2f} seconds, with a total running loss of {:.3f}.'.format(epoch, epoch_time, current_loss))

        # If there is a validation split, compute history, otherwise just track loss
        metrics = {'loss': current_loss, 'time': epoch_time}
        self._history.append(metrics)
        if self._validation_data is not None:
            for (k, v) in self.compute_metrics('valid', *self._validation_data).items():
                metrics[k] = v

            # Display all the metrics if it's a suitable epoch
            if self._current_epoch % self.display_epochs == 0 or self._current_epoch == self._num_epochs:
                tqdm.write(' - '.join(['{}: {:.3f}'.format(k, v) for k, v in metrics.items()]))

            display_metrics = {k: v for k, v in metrics.items() if k in self.epoch_metrics}
            self._epoch_bar.set_postfix(**display_metrics)

            # If there is an early stopping callback, log and track
            early_stopping = self.early_stopping
            if early_stopping is not None and early_stopping(metrics[early_stopping.metric], self.model):
                tqdm.write('Early stopping at epoch {}/{} with a best validation score of {:.3f}.'.format(self._current_epoch, self._num_epochs, self.early_stopping.best_metric))
                raise InterruptTraining

    def before_batch(self, **kwargs):
        running_loss = np.mean(self._epoch_losses)
        self._batches_bar.set_postfix(loss=running_loss)

    def after_batch(self, **kwargs):
        pass
