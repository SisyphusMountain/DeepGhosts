import torch.nn as nn


class MLP(nn.Module):
    """Simple MLP, with ReLU activation and dropout"""
    def __init__(self, in_features, hidden_layers: list, out_features, dropout=0.0):
        super().__init__()
        self.non_linearity = nn.ReLU()
        modules = []
        modules.append(nn.Linear(in_features=in_features, out_features=hidden_layers[0]))
        modules.append(nn.Dropout(p=dropout))
        modules.append(self.non_linearity)
        for index in range(len(hidden_layers)-1):
            modules.append(nn.Linear(in_features=hidden_layers[index],
                                     out_features=hidden_layers[index+1]))
            modules.append(nn.Dropout(p=dropout))
            modules.append(self.non_linearity)
        modules.append(nn.Linear(in_features=hidden_layers[-1],
                                 out_features=out_features))
        self.block = nn.Sequential(*modules)
    def forward(self, x, edge_index=None, edge_attr=None, parenthood=None, batch=None):
        return self.block(x)