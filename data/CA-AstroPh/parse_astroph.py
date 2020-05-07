import os
import json
import numpy as np

DATASET_PATH = '{}'.format(os.path.dirname(os.path.realpath(__file__)))

def parse_edge(edge_line):
    parts = edge_line.split(' ')
    return parts[0], parts[1]

def remap_node(node_label, mapping_dict):
    node = int(node_label)
    node_index = mapping_dict.get(node, None)
    if node_index is None:
        new_label = len(mapping_dict)
        mapping_dict[node] = new_label
        return new_label
    return node_index

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

def parse_astroph(dataset_path=DATASET_PATH):
    node_mapping, edges = parse_graph(open('{}/CA-AstroPH.edgelist'.format(dataset_path), 'r'))

    with open('{}/astroph-id.json'.format(dataset_path), 'w') as f:
        f.write(json.dumps(node_mapping))

    with open('{}/astroph.edgelist'.format(dataset_path), 'w') as f:
        edges = '\n'.join('\t'.join(map(str, t)) for t in edges)
        f.write(edges)

def main():
    parse_astroph()

if __name__ == '__main__':
    main()
