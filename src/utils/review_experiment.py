import json
import fileinput

import numpy as np


def read_result(result_line):
    the_dict = json.loads(result_line)
    return the_dict


def expand_data(experiment):
    results = experiment['results']
    experiment['min'] = np.min(results)
    experiment['max'] = np.max(results)
    experiment['avg'] = np.mean(results)
    experiment['std'] = np.std(results)


total_experiment = []
for line in fileinput.input():
    experiment = read_result(line)
    total_experiment.append(experiment)

for experiment in total_experiment:
    expand_data(experiment)

for experiment in sorted(total_experiment, key=lambda x: x['avg']):
    json.dumps(experiment)
