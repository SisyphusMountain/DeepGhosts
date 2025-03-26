import torch.nn as nn
from blocks import VanillaTransformerBlock
from embedding import NodeEmbedding
from unembedding import LinearUnembedding


class VanillaTransformer(nn.Module):
    def __init__(self,
                node_in_features,
                d_model,
                n_heads,
                mlp_expansion_factor,
                n_blocks,):
        super().__init__()
        self.embedding = NodeEmbedding(node_in_features=node_in_features,
                                       d_model=d_model)
        self.transformer_blocks = nn.ModuleList(
            [VanillaTransformerBlock(d_model=d_model,
                                    n_heads=n_heads,
                                    mlp_expansion_factor=mlp_expansion_factor,)
                                    for _ in range(n_blocks)])
        self.final_layer_norm = nn.LayerNorm(d_model)
        self.unembedding = LinearUnembedding(d_model=d_model)
    def forward(self, x, edge_index, edge_attr, parenthood, batch):
        x = self.embedding(x)
        for module in self.transformer_blocks:
            x = module(x=x, edge_index=edge_index, edge_attr=edge_attr, parenthood=parenthood, batch=batch)
        x = self.final_layer_norm(x)
        x = self.unembedding(x=x, edge_index=edge_index, edge_attr=edge_attr, parenthood=parenthood, batch=batch)
        return x