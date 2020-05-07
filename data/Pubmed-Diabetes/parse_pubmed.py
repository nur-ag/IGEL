import os
import json
import numpy as np
import random

GLOBAL_SEED = 1337
TRAINING_NODES_PER_CLASS = 3286 # 20
VALIDATION_NODES = 4929 # 500
TEST_NODES = 4930 # 1000
DATASET_PATH = '{}/data'.format(os.path.dirname(os.path.realpath(__file__)))

def prepare_splits(class_mapping, 
                   num_train_nodes_per_class=TRAINING_NODES_PER_CLASS, 
                   num_valid_nodes=VALIDATION_NODES, 
                   num_test_nodes=TEST_NODES,
                   random_seed=GLOBAL_SEED):
    random.seed(random_seed)
    labeled_node_set = {n for n in class_mapping}
    inverse_class_mapping = {}
    for (i, c) in class_mapping.items():
        if c not in inverse_class_mapping:
            inverse_class_mapping[c] = []
        inverse_class_mapping[c].append(i)

    training_nodes = set()
    for (_, node_list) in inverse_class_mapping.items():
        random.shuffle(node_list)
        for node_id in node_list[:num_train_nodes_per_class]:
            training_nodes.add(node_id)

    leftover_nodes = sorted({node for node in labeled_node_set if node not in training_nodes})
    random.shuffle(leftover_nodes)
    training_nodes = sorted(training_nodes)
    validation_nodes = sorted(leftover_nodes[:num_valid_nodes])
    test_nodes = sorted(leftover_nodes[num_valid_nodes:][:num_test_nodes])
    return {'train': training_nodes, 
            'valid': validation_nodes,
            'test': test_nodes}

def parse_edge(edge_line):
    parts = edge_line.split('\t')
    return parts[1], parts[3]

def remap_node(node_label, mapping_dict):
    node = int(node_label.split(':')[1])
    node_index = mapping_dict.get(node, None)
    if node_index is None:
        new_label = len(mapping_dict)
        mapping_dict[node] = new_label
        return new_label
    return node_index

def parse_feature(feature_line, feature_mapper):
    feature_data = feature_line.split('\t')[:-1]
    paper_id = feature_data[0]
    label = int(feature_data[1].split('=')[1])
    features = [value.split('=') for value in feature_data[2:]]
    features_map = {key: float(value) for (key, value) in features}
    return paper_id, label, [features_map.get(key, default_value) for (key, default_value) in sorted(feature_mapper.items())]

def parse_feature_defaults(defaults_line):
    defaults_values = defaults_line.split('\t')
    feature_values = defaults_values[1:-1]
    feature_tokens = [v.split(':') for v in feature_values]
    return {tokens[1]: float(tokens[2]) for tokens in feature_tokens}

def parse_graph(file_handler):
    mapping_dict = {}
    edgelist_data = []
    for i, line in enumerate(file_handler):
        if i > 1:
            src, dst = parse_edge(line)
            src_mapped = remap_node(src, mapping_dict)
            dst_mapped = remap_node(dst, mapping_dict)
            edgelist_data.append((src_mapped, dst_mapped))
    return mapping_dict, edgelist_data

def parse_attributes(file_handler, mapping_dict):
    feature_mapper = {}
    label_mapper = {}
    feature_defaults = {}
    for i, line in enumerate(file_handler):
        if i == 0:
            continue
        elif i == 1:
            feature_defaults = parse_feature_defaults(line)
        else:
            paper_id, label, feature_vector = parse_feature(line, feature_defaults)
            clean_id = mapping_dict[int(paper_id)]
            label_mapper[clean_id] = label
            feature_mapper[clean_id] = feature_vector
    return label_mapper, feature_mapper

def parse_pubmed(dataset_path=DATASET_PATH):
    node_mapping, edges = parse_graph(open('{}/Pubmed-Diabetes.DIRECTED.cites.tab'.format(dataset_path), 'r'))
    label_mapper, feature_mapper = parse_attributes(open('{}/Pubmed-Diabetes.NODE.paper.tab'.format(dataset_path), 'r'), node_mapping)

    with open('{}/pubmed-id.json'.format(dataset_path), 'w') as f:
        f.write(json.dumps(node_mapping))

    with open('{}/pubmed.edgelist'.format(dataset_path), 'w') as f:
        edges = '\n'.join('\t'.join(map(str, t)) for t in edges)
        f.write(edges)

    with open('{}/pubmed-class_map.json'.format(dataset_path), 'w') as f:
        f.write(json.dumps(label_mapper))

    with open('{}/pubmed-splits.json'.format(dataset_path), 'w') as f:
        splits = prepare_splits(label_mapper)
        f.write(json.dumps(splits))

    attribute_matrix = np.asarray([vector for (key, vector) in sorted(feature_mapper.items())]).astype(np.float16)
    np.save('{}/pubmed-feats.npy'.format(dataset_path), attribute_matrix)

def main():
    parse_pubmed()

if __name__ == '__main__':
    main()
