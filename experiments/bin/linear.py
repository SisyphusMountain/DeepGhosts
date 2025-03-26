import torch.nn as nn
import torch.nn.functional as F

class LinearModel(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.layer = nn.Linear(in_features=in_features,
                               out_features=out_features,)
    def forward(self,x, edge_index=None, edge_attr=None, parenthood=None, batch=None):
        """We only use x, but it is convenient for the forward functions of all the models
        to have the same signature"""
        # Only positive values make sense
        return self.layer(x)