import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import einops
from torch_geometric.utils import to_dense_batch


class LinearModel(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.layer = nn.Linear(in_features=in_features,
                               out_features=out_features,)
    def forward(self,x, edge_index=None, edge_attr=None, parenthood=None, batch=None):
        """We only use x, but it is convenient for the forward functions of all the models
        to have the same signature"""
        return self.layer(x)

class MLP(nn.Module):
    """Simple MLP, without dropout and with ReLU activation"""
    def __init__(self, in_features, hidden_layers: list, out_features):
        super().__init__()
        self.non_linearity = nn.ReLU()
        modules = []
        modules.append(nn.Linear(in_features=in_features, out_features=hidden_layers[0]))
        modules.append(self.non_linearity)
        for index in range(len(hidden_layers)-1):
            modules.append(nn.Linear(in_features=hidden_layers[index],
                                     out_features=hidden_layers[index+1]))
            modules.append(self.non_linearity)
        modules.append(nn.Linear(in_features=hidden_layers[-1],
                                 out_features=out_features))
        self.block = nn.Sequential(*modules)
    def forward(self, x, edge_index=None, edge_attr=None, parenthood=None, batch=None):
        return self.block(x)

## ---------------- EMBEDDING/UNEMBEDDING ---------------- ##
class NodeEmbedding(nn.Module):
    def __init__(self, node_in_features, d_model):
        super().__init__()
        self.node_embedding = torch.nn.Linear(node_in_features, d_model)
    def forward(self, x, edge_index=None, edge_attr=None, parenthood=None, batch=None):
        return self.node_embedding(x)
    
class LinearUnembedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear = torch.nn.Linear(d_model, 1)
    
    def forward(self, x, edge_index=None, edge_attr=None, parenthood=None, batch=None):
        return self.linear(x)

## ---------------- ATTENTION ---------------- ##
class VanillaAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.W_K = torch.nn.Linear(d_model, d_model)
        self.W_Q = torch.nn.Linear(d_model, d_model)
        self.W_V = torch.nn.Linear(d_model, d_model)
        self.n_heads = n_heads
        self.scaling = (d_model//n_heads)**(-0.5)
        self.final_linear = torch.nn.Linear(d_model, d_model)
        self.attn_dropout = torch.nn.Dropout(dropout)
        self.resid_dropout = torch.nn.Dropout(dropout)
    def forward(self, x, edge_index=None, edge_attr=None, parenthood=None, batch=None):
        x, mask = to_dense_batch(x, batch) # x has shape [batch_size, num_nodes, d_model]
        keys = self.W_K(x)
        queries = self.W_Q(x)
        values = self.W_V(x)
        keys = einops.rearrange(keys, "b n (h x) -> b n h x", h = self.n_heads)
        queries = einops.rearrange(queries, "b n (h x) -> b n h x", h = self.n_heads)
        values = einops.rearrange(values, "b n (h x) -> b n h x", h = self.n_heads)
        attn_coefficents = torch.einsum("bihx, bjhx  -> bhij",keys, queries)*self.scaling
        softmaxed_attn_coefficients = torch.softmax(attn_coefficents, dim = -1)
        softmaxed_attn_coefficients = self.attn_dropout(softmaxed_attn_coefficients)
        computed_values = torch.einsum("bhij, bjhx-> bihx", softmaxed_attn_coefficients, values)
        concatenated_values = einops.rearrange(computed_values, "b i h x ->b i (h x)").contiguous()
        out_dense = self.final_linear(concatenated_values)
        out = out_dense[mask]
        return self.resid_dropout(out)
## ---------------- MESSAGE PASSING ---------------- ##
class GCN(nn.Module):
    def __init__(self,
                 in_channels,
                out_channels,
                dropout=0.1,
                aggr="add"):
        super().__init__()
        self.layer = torch_geometric.nn.conv.GCNConv(in_channels=in_channels,
                                                     out_channels=out_channels)
    def forward(self, x, edge_index=None, edge_attr=None, parenthood=None, batch=None):
        return self.layer(x, edge_index=edge_index, edge_weight=edge_attr)

class GAT(nn.Module):
    def __init__(self,
                 in_channels,
                 heads,
                 out_channels,
                 concat=False,
                 edge_dim = None,
                 aggr="add",
                 dropout=0.1):
        super().__init__()
        if edge_dim is None:
            self.layer = torch_geometric.nn.conv.GATv2Conv(in_channels=in_channels,
                                                        out_channels=out_channels,
                                                        heads=heads,
                                                        concat=concat,
                                                        aggr=aggr,
                                                        dropout=dropout)
        else:
            self.layer = torch_geometric.nn.conv.GATv2Conv(in_channels=in_channels,
                                                     out_channels=out_channels,
                                                     edge_dim = edge_dim,
                                                     heads=heads,
                                                     concat=concat,
                                                     aggr=aggr,
                                                     dropout=dropout)
    def forward(self, x, edge_index, edge_attr, parenthood=None, batch=None):
        return self.layer(x=x, edge_index=edge_index, edge_attr=edge_attr)

class TransformerConv(nn.Module):
    def __init__(self,
                in_channels,
                heads,
                edge_dim,
                out_channels,
                concat=False,
                aggr="add",
                dropout=0.1):
        super().__init__()
        self.layer = torch_geometric.nn.conv.TransformerConv(in_channels=in_channels,
                                                        out_channels=out_channels,
                                                        heads=heads,
                                                        edge_dim = edge_dim,
                                                        concat=concat,
                                                        aggr=aggr,
                                                        dropout=dropout)
    def forward(self, x, edge_index, edge_attr, parenthood=None, batch=None):
        return self.layer(x, edge_index, edge_attr)
## ---------------- BLOCKS ---------------- ##
class VanillaTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, mlp_expansion_factor, dropout, fast_attention=False):
        super().__init__()

        self.attention = VanillaAttention(d_model=d_model,
                                            n_heads=n_heads,
                                            dropout=dropout)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model*mlp_expansion_factor),
            torch.nn.ReLU(),
            torch.nn.Linear(d_model*mlp_expansion_factor, d_model),
        )
        self.layer_norm_1 = torch.nn.LayerNorm(d_model)
        self.layer_norm_2 = torch.nn.LayerNorm(d_model)
        self.dropout = torch.nn.Dropout(dropout)
    def forward(self, x, edge_index=None, edge_attr=None, parenthood=None, batch=None):
        x = self.layer_norm_1(x)
        x = x + self.attention(x = x, batch = batch)
        x = x + self.dropout(self.mlp(self.layer_norm_2(x)))
        return x

class TransformerMPNN(nn.Module):
        def __init__(self,
                d_model,
                n_heads,
                mlp_expansion_factor,
                dropout):
            
            super().__init__()
            self.aggregation = torch.nn.Linear(2*d_model, d_model)

            self.transformer_block = VanillaTransformerBlock(d_model=d_model,
                                                            n_heads=n_heads,
                                                            mlp_expansion_factor=mlp_expansion_factor,
                                                            dropout=dropout)
            self.message_passing = GCN(in_channels=d_model,
                                                    out_channels=d_model,
                                                    dropout=0.1)
            self.layer_norm = torch.nn.LayerNorm(d_model)

        def forward(self, x, edge_index=None, edge_attr=None, parenthood=None, batch=None):
            x = self.layer_norm(self.aggregation(torch.cat([self.message_passing(x, edge_index=edge_index, edge_attr=edge_attr,
                                                                                parenthood=parenthood, batch=batch),
                                                                                self.transformer_block(x, edge_index=None, edge_attr=None, parenthood=None, batch=None)], dim = -1)))+x
            return x

class TransformerMPNNGAT(nn.Module):
        def __init__(self,
                d_model,
                n_heads,
                mlp_expansion_factor,
                dropout,
                edge_dim = None):
            
            super().__init__()
            self.aggregation = torch.nn.Linear(2*d_model, d_model)

            self.transformer_block = VanillaTransformerBlock(d_model=d_model,
                                                            n_heads=n_heads,
                                                            mlp_expansion_factor=mlp_expansion_factor,
                                                            dropout=dropout)
            if edge_dim is None:
                self.message_passing = GAT(in_channels=d_model,
                                                        heads=n_heads,
                                                        out_channels=d_model,
                                                        dropout=0.1)
            else:
                self.message_passing = GAT(in_channels=d_model,
                                                        heads=n_heads,
                                                        edge_dim=edge_dim,
                                                        out_channels=d_model,
                                                        dropout=0.1)
            self.layer_norm = torch.nn.LayerNorm(d_model)

        def forward(self, x, edge_index=None, edge_attr=None, parenthood=None, batch=None):
            x = self.layer_norm(self.aggregation(torch.cat([self.message_passing(x, edge_index=edge_index, edge_attr=edge_attr, parenthood=parenthood, batch=batch),self.transformer_block(x, edge_index=None, edge_attr=None, parenthood=None, batch=None)], dim = -1)))+x
            return x

class TransformerMPNNTransformerConv(nn.Module):
        def __init__(self,
                d_model,
                n_heads,
                mlp_expansion_factor,
                dropout,
                edge_dim = None):
            
            super().__init__()
            self.aggregation = torch.nn.Linear(2*d_model, d_model)

            self.transformer_block = VanillaTransformerBlock(d_model=d_model,
                                                            n_heads=n_heads,
                                                            mlp_expansion_factor=mlp_expansion_factor,
                                                            dropout=dropout)
            self.message_passing = TransformerConv(in_channels=d_model,
                                                    heads=n_heads,
                                                    edge_dim=edge_dim,
                                                    out_channels=d_model,
                                                    dropout=dropout,
                                                    concat=False)
            self.layer_norm = torch.nn.LayerNorm(d_model)

        def forward(self, x, edge_index=None, edge_attr=None, parenthood=None, batch=None):
            x = self.layer_norm(self.aggregation(torch.cat([self.message_passing(x, edge_index=edge_index, edge_attr=edge_attr, parenthood=parenthood, batch=batch),self.transformer_block(x, edge_index=None, edge_attr=None, parenthood=None, batch=None)], dim = -1)))+x
            return x

class TransformerMPNNParenthood(nn.Module):
        def __init__(self,
                d_model,
                n_heads,
                mlp_expansion_factor,
                dropout):
            
            super().__init__()
            self.ln1 = torch.nn.LayerNorm(d_model)
            self.ln2 = torch.nn.LayerNorm(d_model)
            self.ln3 = torch.nn.LayerNorm(d_model)
            self.aggregation = torch.nn.Linear(3*d_model, d_model)
            self.transformer_block = VanillaTransformerBlock(d_model=d_model,
                                                            n_heads=n_heads,
                                                            mlp_expansion_factor=mlp_expansion_factor,
                                                            dropout=dropout)
            self.message_passing = GCN(in_channels=d_model,
                                                    out_channels=d_model,
                                                    dropout=dropout)
            self.parenthood_passing = GAT(in_channels=d_model,
                                                    heads=n_heads,
                                                    out_channels=d_model,
                                                    dropout=dropout)

            self.layer_norm = torch.nn.LayerNorm(d_model)

        def forward(self, x, edge_index=None, edge_attr=None, parenthood=None, batch=None):
            x = self.layer_norm(self.aggregation(torch.cat([self.ln1(self.message_passing(x, edge_index=edge_index, edge_attr=edge_attr, parenthood=parenthood, batch=batch)),
                                                            self.ln2(self.transformer_block(x, edge_index=None, edge_attr=None, parenthood=None, batch=None)),
                                                            self.ln3(self.parenthood_passing(x, edge_index = parenthood, edge_attr=None, parenthood = None, batch = None))], dim = -1)))
            return x
## ---------------- MODEL ---------------- ##

class VanillaTransformer(nn.Module):
    def __init__(self,
                node_in_features,
                d_model,
                n_heads,
                mlp_expansion_factor,
                n_blocks,
                dropout):
        super().__init__()
        self.embedding = NodeEmbedding(node_in_features=node_in_features,
                                       d_model=d_model)
        self.transformer_blocks = nn.ModuleList(
            [VanillaTransformerBlock(d_model=d_model,
                                    n_heads=n_heads,
                                    mlp_expansion_factor=mlp_expansion_factor,
                                    dropout=dropout)
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

class TransformerGCN(nn.Module):
    def __init__(self,
                node_in_features,
                d_model,
                n_heads,
                mlp_expansion_factor,
                n_blocks,
                dropout):
        super().__init__()
        self.embedding = NodeEmbedding(node_in_features=node_in_features,
                                       d_model=d_model)
        self.transformer_blocks = nn.ModuleList(
            [TransformerMPNN(d_model=d_model,
                                    n_heads=n_heads,
                                    mlp_expansion_factor=mlp_expansion_factor,
                                    dropout=dropout)
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
    
class TransformerGAT(nn.Module):
    def __init__(self,
                node_in_features,
                d_model,
                n_heads,
                mlp_expansion_factor,
                n_blocks,
                dropout,
                edge_dim):
        super().__init__()
        self.embedding = NodeEmbedding(node_in_features=node_in_features,
                                       d_model=d_model)
        self.transformer_blocks = nn.ModuleList(
            [TransformerMPNNGAT(d_model=d_model,
                                    n_heads=n_heads,
                                    mlp_expansion_factor=mlp_expansion_factor,
                                    edge_dim = edge_dim,
                                    dropout=dropout)
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

class TransformerTransformer(nn.Module):
    def __init__(self,
                node_in_features,
                d_model,
                n_heads,
                mlp_expansion_factor,
                n_blocks,
                dropout):
        super().__init__()
        self.embedding = NodeEmbedding(node_in_features=node_in_features,
                                       d_model=d_model)
        self.transformer_blocks = nn.ModuleList(
            [TransformerMPNNTransformerConv(d_model=d_model,
                                    n_heads=n_heads,
                                    mlp_expansion_factor=mlp_expansion_factor,
                                    dropout=dropout)
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
    
class TransformerParenthood(nn.Module):
    def __init__(self,
                node_in_features,
                d_model,
                n_heads,
                mlp_expansion_factor,
                n_blocks,
                dropout):
        super().__init__()
        self.embedding = NodeEmbedding(node_in_features=node_in_features,
                                       d_model=d_model)
        self.transformer_blocks = nn.ModuleList(
            [TransformerMPNNParenthood(d_model=d_model,
                                    n_heads=n_heads,
                                    mlp_expansion_factor=mlp_expansion_factor,
                                    dropout=dropout)
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
