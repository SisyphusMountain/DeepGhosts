import torch.nn as nn


class NodeEmbedding(nn.Module):
    def __init__(self, node_in_features, d_model):
        super().__init__()
        self.node_embedding = nn.Linear(node_in_features, d_model)
    def forward(self, x, edge_index=None, edge_attr=None, parenthood=None, batch=None):
        return self.node_embedding(x)