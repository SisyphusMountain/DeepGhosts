import torch.nn as nn


class LinearUnembedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear = nn.Linear(d_model, 1)
    
    def forward(self, x, edge_index=None, edge_attr=None, parenthood=None, batch=None):
        return self.linear(x)