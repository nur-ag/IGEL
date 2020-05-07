import torch
import torch.nn as nn


class NodeInferenceModel(nn.Module):
    def __init__(self, graph_model, graph_outs, num_outs, hidden_size=64, depth=1, activation=nn.ReLU):
        super(NodeInferenceModel, self).__init__()
        self.graph_model = graph_model
        layers = []
        for i in range(depth):
            input_size = graph_outs if i == 0 else hidden_size
            output_size = hidden_size if i < depth - 1 else num_outs
            layers.append(nn.Linear(input_size, output_size))
            if i < depth - 1:
                layers.append(activation())
        self.out = nn.Sequential(*layers) if layers else None

    def forward(self, node_seq, G):
        g_out = self.graph_model(node_seq, G)
        if self.out is not None:
            return self.out(g_out)
        return g_out


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


class EdgeInferenceModel(nn.Module):
    def __init__(self, graph_embedder, vector_size, num_outputs, edge_fn, edge_model=None):
        super(EdgeInferenceModel, self).__init__()
        self.graph_embedder = graph_embedder
        self.out = nn.Linear(vector_size * (2 if edge_fn == 'concat' else 1), num_outputs)
        self.edge_fn = edge_fn
        self.edge_model = edge_model

    def compute_edge_fn(self, src_tensor, dst_tensor):
        if self.edge_fn == 'concat':
            return torch.cat([src_tensor, dst_tensor], axis=1)
        if self.edge_fn == 'add':
            return src_tensor + dst_tensor
        if self.edge_fn == 'mul':
            return src_tensor * dst_tensor
        if self.edge_fn == 'mean':
            return (src_tensor + dst_tensor) / 2
        if self.edge_fn == 'max':
            src_tensor = src_tensor.reshape(1, *src_tensor.shape)
            dst_tensor = dst_tensor.reshape(1, *dst_tensor.shape)
            return torch.cat([src_tensor, dst_tensor], axis=0).max(axis=0).values
        if self.edge_fn == 'weighted_l1':
            return (src_tensor - dst_tensor).abs()
        if self.edge_fn == 'weighted_l2':
            return (src_tensor - dst_tensor) ** 2
        if self.edge_fn == 'model' and self.edge_model is not None:
            return self.edge_model(torch.cat([src_tensor, dst_tensor], axis=1))
        raise ValueError('Invalid edge function "{}". Check the name and if using "model" ensure that your model is not None.'.format(self.edge_fn))

    def forward(self, indices_tuple, G):
        src, dst = indices_tuple
        all_nodes = sorted({n for n in src + dst})
        nodes_mapping = {n: i for i, n in enumerate(all_nodes)}
        as_node_seq = G.vs[all_nodes]
        tensor = self.graph_embedder(as_node_seq, G)
        src_indices = [nodes_mapping[n] for n in src]
        dst_indices = [nodes_mapping[n] for n in dst]
        src_tensor = tensor[src_indices, :]
        dst_tensor = tensor[dst_indices, :]
        classifier_tensor = self.compute_edge_fn(src_tensor, dst_tensor)
        out = self.out(classifier_tensor)
        return out
