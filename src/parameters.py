from batching import batch_dictionary_mapping


class IGELParameters():
    def __init__(self,
                 model_type='simple',
                 vector_length=128,
                 encoding_distance=2,
                 use_distance_labels=True,
                 gates_length=0,
                 gates_steps=0,
                 counts_function='concat_both',
                 aggregator_function='mean',
                 counts_transform='log',
                 neg_sampling_parameters=None):
        self.model_type = model_type
        self.vector_length = vector_length
        self.encoding_distance = encoding_distance
        self.use_distance_labels = use_distance_labels
        self.gates_length = gates_length
        self.gates_steps = gates_steps
        self.counts_function = counts_function
        self.aggregator_function = aggregator_function
        self.neg_sampling_parameters = neg_sampling_parameters
        self.counts_transform = counts_transform


class TrainingParameters():
    def __init__(self,
                 batch_size=512,
                 learning_rate=0.001,
                 weight_decay=0.0,
                 epochs=1,
                 display_epochs=1,
                 batch_samples_fn='uniform',
                 problem_type='unsupervised'):
        if batch_samples_fn not in batch_dictionary_mapping:
            raise ValueError('Unknown batch sampling function "{}"!'.format(batch_samples_fn))
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.display_epochs = display_epochs
        self.batch_samples_fn = batch_dictionary_mapping[batch_samples_fn]
        self.problem_type = problem_type


class NegativeSamplingParameters():
    def __init__(self,
                 random_walk_length=80,
                 window_size=10,
                 negatives_per_positive=10):
        self.random_walk_length = random_walk_length
        self.window_size = window_size
        self.negatives_per_positive = negatives_per_positive
