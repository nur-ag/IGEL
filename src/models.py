import torch
import torch.nn as nn


class NodeClassificationModel(nn.Module):
    def __init__(self, graph_model, graph_outs, num_labels):
        super(NodeClassificationModel, self).__init__()
        self.graph_model = graph_model
        self.out = nn.Linear(graph_outs, num_labels)

    def forward(self, node_seq, G):
        g_out = self.graph_model(node_seq, G)
        out = self.out(g_out)
        return out

class NegativeSamplingModel(nn.Module):
    def __init__(self, graph_model):
        super(NegativeSamplingModel, self).__init__()
        self.graph_model = graph_model
        self.out = nn.Linear(1, 1)

    def forward(self, indices_tuple, G):
        sources, targets = indices_tuple
        all_nodes = sorted({n for n in sources + targets})
        nodes_mapping = {n: i for i, n in enumerate(all_nodes)}
        as_node_seq = G.vs[all_nodes]
        tensor = self.graph_model(as_node_seq, G)
        source_indices = [nodes_mapping[n] for n in sources]
        target_indices = [nodes_mapping[n] for n in targets]
        sources_tensor = tensor[source_indices, :]
        targets_tensor = tensor[target_indices, :]
        product_tensor = (sources_tensor * targets_tensor).sum(axis=1)
        out = self.out(product_tensor.reshape(-1, 1))
        return out
