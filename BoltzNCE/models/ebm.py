import torch
import torch.nn as nn
from .gvp import GVPConv
from .swish import SwishBeta
import dgl
from.graphormer import Graphormer3D


class GVP_EBM(torch.nn.Module):
    def __init__(self, num_features=21, num_layers=8, n_hidden=64,n_vec=16,n_message_gvps=1,n_update_gvps=1,num_particles=22,use_dst_feats=False,vector_gating=True):
        super(GVP_EBM, self).__init__()
        self.n_vec_channels=n_vec
        self.initial_embedding = torch.nn.Sequential(torch.nn.Linear(num_features+1, n_hidden),nn.SiLU())
        self.convs=torch.nn.ModuleList([GVPConv(scalar_size=n_hidden,vector_size=n_vec,n_message_gvps=n_message_gvps,n_update_gvps=n_update_gvps,use_dst_feats=use_dst_feats,vector_gating=vector_gating,coords_range=10,scalar_activation=SwishBeta) for _ in range(num_layers)]) 
        self.output = torch.nn.Sequential(torch.nn.Linear(n_hidden, n_hidden,bias=True), 
                                          nn.SiLU(),
                                          torch.nn.Linear(n_hidden, 1,bias=True))
        self.num_particles=num_particles

    def forward(self,t,data,return_logprob=False, require_grad=False):
        x_init = data.ndata['x'].clone()
        v_init = torch.zeros((data.num_nodes(), self.n_vec_channels, 3), device='cuda')
        torch_grad=False
        if self.training or require_grad:
            torch_grad=True
            t.requires_grad_(True)
            x_init.requires_grad_(True)
            v_init.requires_grad_(True)
        with torch.set_grad_enabled(torch_grad):
            ts=t.repeat_interleave(self.num_particles)
            ts=ts.view(-1,1)
            z_init = torch.cat([data.ndata['h'],ts],dim=1)
            zs = self.initial_embedding(z_init)
            hs,vs,xs=self.convs[0](data,zs,x_init,v_init)
            for i in range(1,len(self.convs)):
                hs,vs,xs=self.convs[i](data,hs,xs,vs)
            data.ndata['h_out'] = hs
            energy = dgl.mean_nodes(data, 'h_out')
            energy=self.output(energy)
            if return_logprob:
                return energy
            position_grad= torch.autograd.grad(energy.sum(), x_init, create_graph=True)[0]
            return position_grad, energy
        

class graphormer_EBM(torch.nn.Module):
    def __init__(self,num_particles, **args):
        super(graphormer_EBM, self).__init__()  
        self.num_particles=num_particles
        self.graphormer=Graphormer3D(**args)

    def forward(self,t,data:dgl.DGLGraph,return_logprob=False, require_grad=False):
        x_init = data.ndata['x'].clone()
        torch_grad=False
        if self.training or require_grad:
            torch_grad=True
            t.requires_grad_(True)
            x_init.requires_grad_(True)
        ts=t.repeat_interleave(data.batch_num_nodes())
        ts=ts.view(-1,1)
        padded_feats, padded_pos,padded_ts = self.graph_to_padded_sequenece(data,x_init,ts)
        with torch.set_grad_enabled(torch_grad):
            energy,padding_mask = self.graphormer(padded_feats, padded_pos, padded_ts)
            if return_logprob:
                return energy
            position_grad = torch.autograd.grad(energy.sum(), x_init, create_graph=True)[0]
            return position_grad, energy

    def graph_to_padded_sequenece(self,data:dgl.DGLGraph,x_init,ts):
        batch_size = data.batch_size
        node_feats=torch.argmax(data.ndata['h'],dim=1)
        node_feats =node_feats+1 #0 index is reserved for padding
        num_nodes_per_graph = data.batch_num_nodes()
        max_nodes = num_nodes_per_graph.max().item()

        padded_feats = torch.zeros(batch_size, max_nodes, device=node_feats.device,dtype=torch.long)
        padded_pos = torch.zeros(batch_size, max_nodes, 3, device=node_feats.device,dtype=torch.float)
        padded_ts = torch.ones(batch_size, max_nodes, 1, device=node_feats.device,dtype=torch.float) * -1 # -1 is reserved for padding

        node_offsets = torch.zeros_like(num_nodes_per_graph)
        node_offsets[1:] = num_nodes_per_graph[:-1].cumsum(dim=0)
        node_ids=torch.arange(len(node_feats), device=node_feats.device)
        graph_ids = torch.arange(len(num_nodes_per_graph), device=data.device)
        graph_ids = graph_ids.repeat_interleave(num_nodes_per_graph)
        node_pos_in_graph = node_ids - node_offsets[graph_ids]
        padded_feats[graph_ids, node_pos_in_graph] = node_feats
        padded_pos[graph_ids, node_pos_in_graph] = x_init
        padded_ts[graph_ids, node_pos_in_graph] = ts
        return padded_feats, padded_pos, padded_ts

