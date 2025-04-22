import torch
import torch.nn as nn
from .gvp import GVPConv
from .swish import SwishBeta
import dgl


class GVP_EBM(torch.nn.Module):
    def __init__(self, n_features=21, num_layers=8, n_hidden=64,n_vec=16,n_message_gvps=1,n_update_gvps=1,num_particles=22,use_dst_feats=False,vector_gating=True):
        super(GVP_EBM, self).__init__()
        self.n_vec_channels=n_vec
        self.initial_embedding = torch.nn.Sequential(torch.nn.Linear(n_features+1, n_hidden),nn.SiLU())
        self.convs=torch.nn.ModuleList([GVPConv(scalar_size=n_hidden,vector_size=n_vec,n_message_gvps=n_message_gvps,n_update_gvps=n_update_gvps,use_dst_feats=use_dst_feats,vector_gating=vector_gating,coords_range=10,scalar_activation=SwishBeta) for _ in range(num_layers)]) 
        self.output = torch.nn.Linear(n_hidden, 1,bias=True)
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
        
