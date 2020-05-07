import torch
import torch.nn as nn
import torch.optim as optim

from models import EdgeInferenceModel, NegativeSamplingModel
from learning import GraphNetworkTrainer
from batching import graph_random_walks, negative_sampling_generator, negative_sampling_batcher
from embedders import SimpleStructuralEmbedder, GatedStructuralEmbedder
from aggregators import SamplingAggregator
from structural import StructuralMapper


def make_aggregation_model(embedding_model, embedding_size, options, device):
    model = embedding_model
    if options is None:
        return model, embedding_size
    zipped_params = zip(options.node_dropouts,
                        options.output_sizes,
                        options.activations,
                        options.aggregations,
                        options.include_nodes,
                        options.nodes_to_sample,
                        options.num_attention_heads,
                        options.attention_aggregators,
                        options.attention_outputs_per_heads,
                        options.attention_dropouts)
    current_features = embedding_size
    for (dropout, output_size, activation, agg_fn, include_node, nodes_to_sample, 
         att_heads, att_agg, att_outputs_per_head, att_dropout) in zipped_params:
        model = SamplingAggregator(model, 
                                   current_features, 
                                   aggregation=agg_fn,
                                   hidden_units=0,
                                   num_hidden=0,
                                   output_units=output_size,
                                   activation=activation,
                                   node_dropout=dropout,
                                   num_attention_heads=att_heads,
                                   nodes_to_sample=nodes_to_sample,
                                   include_node=include_node,
                                   attention_aggregator=att_agg,
                                   attention_outputs_by_head=att_outputs_per_head,
                                   attention_dropout=att_dropout,
                                   peeking_units=0,
                                   number_of_peeks=0,
                                   device=device).to(device)
        current_features = max(1, att_heads if att_outputs_per_head else 1) * output_size + (current_features * include_node)

    return model, current_features


def make_structural_model(G, options, device):
    mapper = StructuralMapper(G, distance=options.encoding_distance, use_distances=options.use_distance_labels, device=device)
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
                                        options.counts_function, 
                                        options.aggregator_function,
                                        options.counts_transform, 
                                        device=device)
    else:
        raise ValueError('Unknown model type "{}".'.format(options.model_type))
    return mapper, model


def lambda_batch_iterator(G, neg_sampling_parameters, training_options, device):
    batch_gen = graph_random_walks(G, neg_sampling_parameters.random_walk_length, training_options.batch_size, training_options.batch_samples_fn)
    ns_gen = negative_sampling_generator(G, 
                                         batch_gen, 
                                         neg_sampling_parameters.window_size, 
                                         neg_sampling_parameters.negatives_per_positive,
                                         neg_sampling_parameters.minimum_negative_distance)
    pair_labels = negative_sampling_batcher(ns_gen, training_options.batch_size)
    all_batches = ((pair, torch.FloatTensor(label).to(device).reshape(-1, 1)) for pair, label in pair_labels)
    return all_batches


def train_negative_sampling(G, model, neg_sampling_parameters, training_options, device):
    training_model = NegativeSamplingModel(model).to(device)
    batch_iterator_fn = lambda: lambda_batch_iterator(G, neg_sampling_parameters, training_options, device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(training_model.parameters(), lr=training_options.learning_rate, weight_decay=training_options.weight_decay)
    trainer = GraphNetworkTrainer(training_model,
                                  optimizer, 
                                  criterion, 
                                  display_epochs=training_options.display_epochs, 
                                  problem_type='unsupervised')
    trainer.fit(batch_iterator_fn, G, training_options.epochs)
    return model

