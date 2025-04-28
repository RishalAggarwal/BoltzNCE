import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Tuple
from torch import Tensor

@torch.jit.script
def softmax_dropout(input, dropout_prob: float, is_training: bool):
    return F.dropout(F.softmax(input, -1), dropout_prob, is_training)


@torch.jit.script
def gaussian(x, mean, std):
    pi = 3.14159
    a = (2*pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)


class SelfMultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        scaling_factor=1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.num_heads = num_heads
        self.dropout = dropout

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = (self.head_dim * scaling_factor) ** -0.5

        self.in_proj: Callable[[Tensor], Tensor] = nn.Linear(
            embed_dim, embed_dim * 3, bias=bias
        )
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(
        self,
        query: Tensor,
        attn_bias: Tensor = None,
    ) -> Tensor:
        n_node, n_graph, embed_dim = query.size()
        #(n_graph, n_node, embed_dim)
        q, k, v = self.in_proj(query).chunk(3, dim=-1)

        _shape = (-1, n_graph * self.num_heads, self.head_dim)
        #(n_graph*num_heads, n_node,head_dim)
        q = q.contiguous().view(_shape).transpose(0, 1) * self.scaling
        k = k.contiguous().view(_shape).transpose(0, 1)
        v = v.contiguous().view(_shape).transpose(0, 1) 

        # (n_graph * num_heads, n_node,n_node))
        attn_weights = torch.bmm(q, k.transpose(1, 2)) + attn_bias
        attn_probs = softmax_dropout(attn_weights, self.dropout, self.training)

        attn = torch.bmm(attn_probs, v)
        attn = attn.transpose(0, 1).contiguous().view(n_node, n_graph, embed_dim)
        attn = self.out_proj(attn)
        return attn


class Graphormer3DEncoderLayer(nn.Module):
    """
    Implements a Graphormer-3D Encoder Layer.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        num_kernel: int = 50,
        attention_heads: int = 32,
    ) -> None:
        super().__init__()

        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout

        self.dropout = dropout
        self.activation_dropout = activation_dropout

        self.self_attn = SelfMultiheadAttention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
        )
        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = nn.LayerNorm(self.embedding_dim)
        self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)
        self.final_layer_norm = nn.LayerNorm(self.embedding_dim)
        self.bias_proj = NonLinear(num_kernel, attention_heads)

    def forward(
        self,
        x: Tensor,
        gbf_feature: Tensor = None,
        padding_mask: Tensor = None,
    ):
        residual = x
        n_node, n_graph, embed_dim = x.size()
        x = self.self_attn_layer_norm(x)
        attn_bias = self.bias_proj(gbf_feature).permute(0, 3, 1, 2)
        attn_bias.masked_fill_(padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
        attn_bias = attn_bias.reshape(-1, n_node, n_node)
        x = self.self_attn(
            query=x,
            attn_bias=attn_bias,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x

        residual = x
        x = self.final_layer_norm(x)
        x = F.silu(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        return x,attn_bias


class GaussianLayer(nn.Module):
    def __init__(self, K=128, edge_types=1024):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        self.mul = nn.Embedding(edge_types, 1)
        self.bias = nn.Embedding(edge_types, 1)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, x, edge_types):
        mul = self.mul(edge_types)
        bias = self.bias(edge_types)
        x = mul * x.unsqueeze(-1) + bias
        x = x.expand(-1, -1, -1, self.K)
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-5
        return gaussian(x.float(), mean, std).type_as(self.means.weight)
    
class RBF(nn.Module):
    def __init__(self, K, edge_types):
        super().__init__()
        self.K = K
        self.means = nn.parameter.Parameter(torch.empty(K))
        self.temps = nn.parameter.Parameter(torch.empty(K))
        self.mul: Callable[..., Tensor] = nn.Embedding(edge_types, 1)
        self.bias: Callable[..., Tensor] = nn.Embedding(edge_types, 1)
        nn.init.uniform_(self.means, 0, 3)
        nn.init.uniform_(self.temps, 0.1, 10)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, x: Tensor, edge_types):
        mul = self.mul(edge_types)
        bias = self.bias(edge_types)
        x = mul * x.unsqueeze(-1) + bias
        mean = self.means.float()
        temp = self.temps.float().abs()
        return ((x - mean).square() * (-temp)).exp().type_as(self.means)


class NonLinear(nn.Module):
    def __init__(self, input, output_size, hidden=None):
        super(NonLinear, self).__init__()
        if hidden is None:
            hidden = input
        self.layer1 = nn.Linear(input, hidden)
        self.layer2 = nn.Linear(hidden, output_size)

    def forward(self, x):
        x = F.silu(self.layer1(x))
        x = self.layer2(x)
        return x


class NodeTaskHead(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        coord_range: float = 1.0,
    ):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim)
        self.coord_range = coord_range
        self.embed_dim = embed_dim
        self.q_proj: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, embed_dim)
        self.k_proj: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, embed_dim)
        self.v_proj: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, embed_dim)
        self.num_heads = num_heads
        self.scaling = (embed_dim // num_heads) ** -0.5
        self.force_proj1: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, 1)
        self.force_proj2: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, 1)
        self.force_proj3: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, 1)

    def forward(
        self,
        query: Tensor,
        attn_bias: Tensor,
        delta_pos: Tensor,
    ) -> Tensor:
        query = self.ln(query)
        bsz, n_node, _ = query.size()
        q = (
            self.q_proj(query).view(bsz, n_node, self.num_heads, -1).transpose(1, 2)
            * self.scaling
        )
        k = self.k_proj(query).view(bsz, n_node, self.num_heads, -1).transpose(1, 2) 
        v = self.v_proj(query).view(bsz, n_node, self.num_heads, -1).transpose(1, 2) 
        attn = q @ k.transpose(-1, -2)  # [bsz, head, n, n]
        attn_probs = softmax_dropout(
            attn.view(-1, n_node, n_node) + attn_bias, 0.1, self.training
        ).view(bsz, self.num_heads, n_node, n_node)
        rot_attn_probs = attn_probs.unsqueeze(-1) * delta_pos.unsqueeze(1).type_as(
            attn_probs
        )  # [bsz, head, n, n, 3]
        rot_attn_probs = rot_attn_probs.permute(0, 1, 4, 2, 3)
        x = rot_attn_probs @ v.unsqueeze(2)  # [bsz, head , 3, n, d]
        x = x.permute(0, 3, 2, 1, 4).contiguous().view(bsz, n_node, 3, -1)
        f1 = self.force_proj1(x[:, :, 0, :]).view(bsz, n_node, 1)
        f2 = self.force_proj2(x[:, :, 1, :]).view(bsz, n_node, 1)
        f3 = self.force_proj3(x[:, :, 2, :]).view(bsz, n_node, 1)
        cur_force = torch.cat([f1, f2, f3], dim=-1).float()
        cur_force = F.tanh(cur_force) * self.coord_range
        return cur_force

class Graphormer3D(nn.Module):
    def __init__(self, num_features=21,num_layers=6,embed_dim=512,ffn_embed_dim=512, attention_heads=32, dropout=0.1, attention_dropout=0.1, activation_dropout=0.1, num_kernel=50, input_dropout=0.1,blocks=3):
        super().__init__()
        self.atom_types = num_features+1 #0 is padding index
        self.blocks = blocks
        self.edge_types = self.atom_types * self.atom_types
        self.atom_encoder = nn.Embedding(self.atom_types, embed_dim, padding_idx=0)
        self.time_encoder = NonLinear(1, embed_dim)
        #self.tag_encoder = nn.Embedding(3, args['embed_dim'])
        self.input_dropout = input_dropout
        self.layers = nn.ModuleList([
            Graphormer3DEncoderLayer(
                embed_dim,
                ffn_embed_dim,
                num_attention_heads=attention_heads,
                dropout=dropout,
                attention_dropout=attention_dropout,
                activation_dropout=activation_dropout,
                num_kernel=num_kernel,
                attention_heads=attention_heads,
            ) for _ in range(num_layers)
        ])
        self.bias_updates=nn.ModuleList([
            NodeTaskHead(
                embed_dim,
                attention_heads,
            ) for _ in range(num_layers)
        ])
        self.final_ln = nn.LayerNorm(embed_dim)
        self.energy_proj = NonLinear(embed_dim, 1)
        #self.energy_agg_factor = nn.Embedding(3, 1)
        #nn.init.normal_(self.energy_agg_factor.weight, 0, 0.01)

        self.gbf = GaussianLayer(num_kernel, self.edge_types)
        #self.bias_proj = NonLinear(num_kernel, attention_heads)
        self.edge_proj = nn.Linear(num_kernel, embed_dim)

    def forward(self, atoms, pos,time):
        padding_mask = atoms.eq(0)
        real_mask = ~padding_mask
        n_graph, n_node = atoms.size()
        delta_pos = pos.unsqueeze(1) - pos.unsqueeze(2)
        delta_pos = delta_pos + 1e-5
        dist = delta_pos.norm(dim=-1)
        delta_pos = delta_pos/dist.unsqueeze(-1)
        edge_type = atoms.view(n_graph, n_node, 1) * self.atom_types + atoms.view(n_graph, 1, n_node)
        gbf_feature = self.gbf(dist, edge_type)
        edge_features = gbf_feature.masked_fill(padding_mask.unsqueeze(1).unsqueeze(-1), 0.0)

        node_features = self.atom_encoder(atoms) + self.edge_proj(edge_features.sum(dim=-2)) + self.time_encoder(time)
        output = F.dropout(node_features, p=self.input_dropout, training=self.training).transpose(0, 1)

        '''graph_attn_bias = self.bias_proj(gbf_feature).permute(0, 3, 1, 2)
        graph_attn_bias.masked_fill_(padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
        graph_attn_bias = graph_attn_bias.reshape(-1, n_node, n_node)'''

        for _ in range(self.blocks):
            for i in range(len(self.layers)):
                output,graph_attn_bias = self.layers[i](output, gbf_feature, padding_mask)
                '''output_for_force = output.transpose(0, 1)
                #graph_attn_bias_for_force = graph_attn_bias * real_mask.unsqueeze(1).unsqueeze(2)
                cur_force = self.bias_updates[i](output_for_force, graph_attn_bias, delta_pos)
                pos = pos + cur_force
                delta_pos = pos.unsqueeze(1) - pos.unsqueeze(2)
                delta_pos = delta_pos + 1e-5
                dist = delta_pos.norm(dim=-1)
                delta_pos = delta_pos/dist.unsqueeze(-1)
                gbf_feature = self.gbf(dist, edge_type)'''
                '''graph_attn_bias = self.bias_proj(gbf_feature).permute(0, 3, 1, 2)
                graph_attn_bias.masked_fill_(padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
                graph_attn_bias = graph_attn_bias.reshape(-1, n_node, n_node)'''


        output = self.final_ln(output).transpose(0, 1)
        eng_output = F.dropout(output, p=0.1, training=self.training)
        eng_output = (self.energy_proj(eng_output)).flatten(-2)
        eng_output = (eng_output*real_mask).sum(dim=-1).view(-1,1)
        return eng_output,padding_mask
