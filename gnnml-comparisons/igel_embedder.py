import torch as T
import igraph as ig
from structural import StructuralMapper
from embedders import SimpleStructuralEmbedder
from parameters import IGELParameters, NegativeSamplingParameters, TrainingParameters
from model_utils import train_negative_sampling
from learning import set_seed


TRAINING_OPTIONS = TrainingParameters(batch_size=10000, learning_rate=0.1, weight_decay=0.0, epochs=2, display_epochs=1, batch_samples_fn='uniform', problem_type='unsupervised')
NEGATIVE_SAMPLING_OPTIONS = NegativeSamplingParameters(random_walk_length=15, window_size=3, negatives_per_positive=10, minimum_negative_distance=1)

DEBUG = False
if DEBUG:
    TRAINING_OPTIONS = TrainingParameters(batch_size=10000, learning_rate=0.1, weight_decay=0.0, epochs=1, display_epochs=1, batch_samples_fn='uniform', problem_type='unsupervised')
    NEGATIVE_SAMPLING_OPTIONS = NegativeSamplingParameters(random_walk_length=1, window_size=1, negatives_per_positive=1, minimum_negative_distance=1)

def get_unsupervised_embeddings(G, 
                                distance, 
                                seed=0,
                                vector_length=5,
                                train_opts=TRAINING_OPTIONS):
    return get_unsupervised_embedder(G, distance, seed, vector_length, train_opts)(G.vs, G).cpu().detach().numpy()


def get_unsupervised_embedder(G, 
                              distance, 
                              seed=0,
                              vector_length=5,
                              train_opts=TRAINING_OPTIONS):
    set_seed(seed)
    nsp = NEGATIVE_SAMPLING_OPTIONS
    device = T.device('cuda') if T.cuda.is_available() else T.device('cpu')

    mapper = StructuralMapper(G, distance, use_distances=True, num_workers=1, device=device)
    model = SimpleStructuralEmbedder(vector_length, mapper, counts_transform='log', device=device)
    trained_model = train_negative_sampling(G, model, nsp, train_opts, device)
    return trained_model
