import torch
import torch.nn as nn
import torch_geometric
from blocks import VanillaTransformerBlock

class GCN(nn.Module):
    def __init__(self,
                 in_channels,
                out_channels,
                aggr="add"):
        super().__init__()
        self.layer = torch_geometric.nn.conv.GCNConv(in_channels=in_channels,
                                                     out_channels=out_channels,
                                                     aggr=aggr)
    def forward(self, x, edge_index=None, edge_attr=None, parenthood=None, batch=None):
        return self.layer(x, edge_index=edge_index, edge_weight=edge_attr)


class TransformerMPNN(nn.Module):
        def __init__(self,
                d_model,
                n_heads,
                mlp_expansion_factor,
                aggr="add"
                ):
            super().__init__()
            self.transformer_block = VanillaTransformerBlock(d_model=d_model,
                                                            n_heads=n_heads,
                                                            mlp_expansion_factor=mlp_expansion_factor,)
            self.message_passing = GCN(in_channels=d_model,
                                                    out_channels=d_model,
                                                    aggr=aggr)

            self.aggregation = torch.nn.Linear(2*d_model, d_model)
        def forward(self, x, edge_index=None, edge_attr=None, parenthood=None, batch=None):
            x = x + self.aggregation(torch.cat([self.message_passing(x, edge_index=edge_index, edge_attr=edge_attr,
                                                                                parenthood=parenthood, batch=batch),
                                                                                self.transformer_block(x, edge_index=None, edge_attr=None, parenthood=None, batch=batch)], dim = -1))
            return x
        
class TransformerMPNNParenthood(nn.Module):
        def __init__(self,
                d_model,
                n_heads,
                mlp_expansion_factor,
                dropout):
            
            super().__init__()
            self.transformer_block = VanillaTransformerBlock(d_model=d_model,
                                                            n_heads=n_heads,
                                                            mlp_expansion_factor=mlp_expansion_factor,
                                                            )
            self.message_passing = GCN(in_channels=d_model,
                                                    out_channels=d_model,
                                                    )
            self.parenthood_passing = GCN(in_channels=d_model,
                                                    out_channels=d_model,
                                                    )
            self.aggregation = torch.nn.Linear(3*d_model, d_model)


        def forward(self, x, edge_index=None, edge_attr=None, parenthood=None, batch=None):
            x = x + self.aggregation(torch.cat([self.message_passing(x, edge_index=edge_index, edge_attr=edge_attr, parenthood=parenthood, batch=batch),
                                                            self.transformer_block(x, edge_index=None, edge_attr=None, parenthood=None, batch=None),
                                                            self.parenthood_passing(x, edge_index = parenthood, edge_attr=None, parenthood = None, batch = None)], dim = -1))
            return x