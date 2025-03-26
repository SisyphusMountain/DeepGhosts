import torch.nn as nn
import torch
from torch_geometric.utils import to_dense_batch
import einops

class VanillaAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.n_heads = n_heads
        self.scaling = (d_model//n_heads)**(-0.5)
        self.final_linear = nn.Linear(d_model, d_model)
    def forward(self, x, edge_index=None, edge_attr=None, parenthood=None, batch=None):
        x, mask = to_dense_batch(x, batch) # x has shape [batch_size, num_nodes, d_model]
        attn_mask = mask.unsqueeze(1) * mask.unsqueeze(2)
        float_attn_mask = ((~attn_mask)*(-1.0e9)).unsqueeze(1)
        keys = self.W_K(x)
        queries = self.W_Q(x)
        values = self.W_V(x)
        keys = einops.rearrange(keys, "b n (h x) -> b n h x", h = self.n_heads)
        queries = einops.rearrange(queries, "b n (h x) -> b n h x", h = self.n_heads)
        values = einops.rearrange(values, "b n (h x) -> b n h x", h = self.n_heads)
        attn_coefficients = torch.einsum("bihx, bjhx  -> bhij",keys, queries)*self.scaling
        attn_coefficients += float_attn_mask
        softmaxed_attn_coefficients = torch.softmax(attn_coefficients, dim = -1)
        computed_values = torch.einsum("bhij, bjhx-> bihx", softmaxed_attn_coefficients, values)
        concatenated_values = einops.rearrange(computed_values, "b i h x ->b i (h x)").contiguous()
        out_dense = self.final_linear(concatenated_values)
        out = out_dense[mask]
        return out