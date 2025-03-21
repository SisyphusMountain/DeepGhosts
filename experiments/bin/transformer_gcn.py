import torch.nn as nn
import torch.nn.functional as F
from unembedding import LinearUnembedding
from embedding import NodeEmbedding
from mpnn import TransformerMPNN


class TransformerGCN(nn.Module):
    def __init__(self,
                node_in_features,
                d_model,
                n_heads,
                mlp_expansion_factor,
                n_blocks,
                aggr,):
        super().__init__()
        self.embedding = NodeEmbedding(node_in_features=node_in_features,
                                       d_model=d_model)
        self.transformer_blocks = nn.ModuleList(
            [TransformerMPNN(d_model=d_model,
                                    n_heads=n_heads,
                                    mlp_expansion_factor=mlp_expansion_factor,
                                    aggr=aggr,)
                                    for _ in range(n_blocks)])
        self.unembedding = LinearUnembedding(d_model=d_model)
    def forward(self, x, edge_index, edge_attr, parenthood, batch):
        x = self.embedding(x)
        for module in self.transformer_blocks:
            x = module(x=x, edge_index=edge_index, edge_attr=edge_attr, parenthood=parenthood, batch=batch)
        x = self.unembedding(x=x, edge_index=edge_index, edge_attr=edge_attr, parenthood=parenthood, batch=batch)
        return x