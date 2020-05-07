import json
import argparse
import numpy as np

import torch

from sklearn.cluster import KMeans

from graph import load_graph
from models import NegativeSamplingModel
from learning import set_seed
from model_utils import make_structural_model, train_negative_sampling
from experiment_utils import create_experiment_params


DEFAULT_DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
LINK_PREDICTION_OUTPUTS = 1
EXPERIMENT_ID = 1
NUM_CLUSTERS = 9

KEY = 'Karate' 
GRAPH_PATH = 'data/{}/{}.edgelist'.format(KEY, KEY)
OUTPUT_PATH = 'output/{}-result.jsonl'.format(KEY)
CLUSTERS_OUTPUT_PATH = 'output/{}.clusters.tsv'.format(KEY)
EXPERIMENT_PATH = 'configs/clusteringAtK2.json'

def clustering_experiment(graph_path, 
                          model_options=None,
                          training_options=None,
                          cluster_kwargs={'n_clusters': NUM_CLUSTERS},
                          device=DEFAULT_DEVICE,
                          experiment_id=EXPERIMENT_ID,
                          use_seed=True):
    seed = None
    if use_seed:
        seed = experiment_id
        set_seed(seed)
    G = load_graph(graph_path)

    # Prepare the components and train the embedding model
    mapper, model = make_structural_model(G, model_options, device)
    trained_model = train_negative_sampling(G, model, model_options.neg_sampling_parameters, training_options, device)

    # Train the clustering model and compute the per-node cluster
    node_ids = {i: node['id'] for i, node in enumerate(G.vs)}
    embeddings = trained_model(G.vs, G).detach().numpy()
    clustering_model = KMeans(**cluster_kwargs).fit(embeddings)
    node_clusters = clustering_model.predict(embeddings)
    node_clusters = {node_ids[i]: int(value) for i, value in enumerate(node_clusters)}
    return clustering_model, node_clusters


def parse_arguments():
    main_args = argparse.ArgumentParser()
    main_args.add_argument('-n', '--num-clusters', help='Number of experiment attempts to run per experiment configuration.', type=int, default=NUM_CLUSTERS)
    main_args.add_argument('-g', '--graph-path', help='Path to the graph edgelist file to be used.', type=str, default=GRAPH_PATH)
    main_args.add_argument('-o', '--output-path', help='Path to store the clustering results, as a tab separated file containing node ids and clusters.', type=str, default=CLUSTERS_OUTPUT_PATH)
    main_args.add_argument('-e', '--experiment-config', help='Path to the experiment configuration json specifying training and model parameters.', type=str, default=EXPERIMENT_PATH)
    return main_args.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    num_clusters = args.num_clusters
    graph_path = args.graph_path
    output_path = args.output_path

    experiment_as_dict = {}
    with open(args.experiment_config, 'r') as f:
        experiment_as_dict = json.load(f)

    model_config, training_config = create_experiment_params(experiment_as_dict)
    model, clusters = clustering_experiment(graph_path, 
                                            model_config, 
                                            training_config, 
                                            cluster_kwargs={'n_clusters': num_clusters})

    with open(output_path, 'w') as f:
        for node_id, cluster in clusters.items():
            f.write('{}\t{}\n'.format(node_id, cluster))




