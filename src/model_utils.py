import torch
import torch.nn as nn
import torch.optim as optim

from models import EdgeInferenceModel, NegativeSamplingModel
from learning import GraphNetworkTrainer
from batching import graph_random_walks, negative_sampling_generator, negative_sampling_batcher
from embedders import SimpleStructuralEmbedder, GatedStructuralEmbedder
from structural import StructuralMapper


def make_structural_model(G, options, device):
    mapper = StructuralMapper(G, distance=options.encoding_distance, use_distances=options.use_distance_labels)
    if options.model_type == 'simple':
        model = SimpleStructuralEmbedder(options.vector_length, 
                                         mapper, 
                                         options.counts_transform,
                                         device=device)
    elif options.model_type == 'gated':
        model = GatedStructuralEmbedder(options.vector_length, 
                                        options.gates_length, 
                                        options.gates_steps, 
                                        mapper, 
                                        options.transform_output, 
                                        options.counts_function, 
                                        options.aggregator_function,
                                        options.counts_transform, 
                                        device=device)
    else:
        raise ValueError('Unknown model type "{}".'.format(options.model_type))
    return mapper, model


def lambda_batch_iterator(G, neg_sampling_parameters, training_options, device):
    batch_gen = graph_random_walks(G, neg_sampling_parameters.random_walk_length, training_options.batch_size)
    ns_gen = negative_sampling_generator(G, 
                                         batch_gen, 
                                         neg_sampling_parameters.window_size, 
                                         neg_sampling_parameters.negatives_per_positive)
    pair_labels = negative_sampling_batcher(ns_gen, training_options.batch_size)
    all_batches = ((pair, torch.Tensor(label).to(device).reshape(-1, 1)) for pair, label in pair_labels)
    return all_batches


def train_negative_sampling(G, model, neg_sampling_parameters, training_options, device):
    training_model = NegativeSamplingModel(model).to(device)
    batch_iterator_fn = lambda: lambda_batch_iterator(G, neg_sampling_parameters, training_options, device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(training_model.parameters())
    trainer = GraphNetworkTrainer(training_model,
                                  optimizer, 
                                  criterion, 
                                  display_epochs=training_options.display_epochs, 
                                  problem_type='unsupervised')
    trainer.fit(batch_iterator_fn, G, training_options.epochs)
    return model

