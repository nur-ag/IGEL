import os
import json
import numpy as np
import random

GLOBAL_SEED = 1337
TRAINING_NODES_PER_CLASS = 200 # 20
VALIDATION_NODES = 654 # 500
TEST_NODES = 654 # 1000
DATASET_PATH = '{}'.format(os.path.dirname(os.path.realpath(__file__)))

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
    src, dst = edge_line.split('\t')
    return src, dst

def remap_node(node_label, mapping_dict):
    node = int(node_label)
    node_index = mapping_dict.get(node, None)
    if node_index is None:
        new_label = len(mapping_dict)
        mapping_dict[node] = new_label
        return new_label
    return node_index

def parse_feature(feature_line):
    feature_data = feature_line.split('\t')
    paper_id = int(feature_data[0])
    features = [float(value) for value in feature_data[1:-1]]
    label = feature_data[-1].strip()
    return paper_id, label, features

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
    feature_mapping = {}
    for i, line in enumerate(file_handler):
        paper_id, label, feature_vector = parse_feature(line)
        if label not in feature_mapping:
            feature_mapping[label] = len(feature_mapping)
        label_id = feature_mapping[label]
        clean_id = mapping_dict[int(paper_id)]
        label_mapper[clean_id] = label_id
        feature_mapper[clean_id] = feature_vector
    return label_mapper, feature_mapper, feature_mapping

def parse_cora(dataset_path=DATASET_PATH):
    node_mapping, edges = parse_graph(open('{}/cora.cites'.format(dataset_path), 'r'))
    label_mapper, feature_mapper, label_mapping = parse_attributes(open('{}/cora.content'.format(dataset_path), 'r'), node_mapping)

    with open('{}/cora.edgelist'.format(dataset_path), 'w') as f:
        edges = '\n'.join('\t'.join(map(str, t)) for t in edges)
        f.write(edges)

    with open('{}/cora-class_map.json'.format(dataset_path), 'w') as f:
        f.write(json.dumps(label_mapper))

    with open('{}/cora-classes_map.json'.format(dataset_path), 'w') as f:
        f.write(json.dumps(label_mapping))

    with open('{}/cora-splits.json'.format(dataset_path), 'w') as f:
        splits = prepare_splits(label_mapper)
        f.write(json.dumps(splits))

    attribute_matrix = np.asarray([vector for (key, vector) in sorted(feature_mapper.items())]).astype(np.float16)
    np.save('{}/cora-feats.npy'.format(dataset_path), attribute_matrix)

def main():
    parse_cora()

if __name__ == '__main__':
    main()
