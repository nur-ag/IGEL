import json
import hashlib
from itertools import product


def hash_dictionary(dictionary):
	as_string = json.dumps(dictionary)
	hashed = hashlib.md5(as_string.encode())
	return hashed.hexdigest()


def generate_experiment_tuples(experiment_dict):
	all_sets = [[(i, v) for v in values] 
				for (i, values) in sorted(experiment_dict.items())]
	return product(*all_sets)


def tuple_to_dictionary(experiment_tuple):
	return {key: value for (key, value) in experiment_tuple}
