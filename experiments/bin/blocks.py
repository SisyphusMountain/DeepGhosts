import torch.nn as nn
from attention import VanillaAttention


class VanillaTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, mlp_expansion_factor):
        super().__init__()

        self.attention = VanillaAttention(d_model=d_model,
                                            n_heads=n_heads,)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model*mlp_expansion_factor),
            nn.ReLU(),
            nn.Linear(d_model*mlp_expansion_factor, d_model),
        )
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)
    def forward(self, x, edge_index=None, edge_attr=None, parenthood=None, batch=None):
        x = x + self.attention(x = self.layer_norm_1(x), batch = batch)
        x = x + self.mlp(self.layer_norm_2(x))
        return x