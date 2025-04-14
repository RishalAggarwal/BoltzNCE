import torch
import torch.nn as nn
import torch.nn.functional as F
from .gvp import GVPConv, NodePositionUpdate
from .swish import SwishBeta


class GVP_vector_field(torch.nn.Module):
    def __init__(self, n_features=21, n_layers=5, n_hidden=64,n_vec=16,n_message_gvps=1,n_update_gvps=1,n_coord_gvps=1,num_particles=22, use_dst_feats=False,vector_gating=True):
        super(GVP_vector_field, self).__init__()
        self.n_vec_channels=n_vec
        self.initial_embedding = torch.nn.Sequential(torch.nn.Linear(n_features+1, n_hidden),nn.SiLU())
        self.convs=torch.nn.ModuleList([GVPConv(scalar_size=n_hidden,vector_size=n_vec,n_message_gvps=n_message_gvps,n_update_gvps=n_update_gvps,use_dst_feats=use_dst_feats,vector_gating=vector_gating,coords_range=10,scalar_activation=SwishBeta) for _ in range(n_layers)]) 
        self.position_updater=NodePositionUpdate(n_scalars=n_hidden,n_vec_channels=n_vec,n_gvps=n_coord_gvps)
        self.num_particles=num_particles

    def forward(self,t,data):
        x_init = data.ndata['x'].clone()
        v_init = torch.zeros((data.num_nodes(), self.n_vec_channels, 3), device='cuda')
        ts=t.repeat_interleave(self.num_particles)
        ts=ts.view(-1,1)
        z_init = torch.cat([data.ndata['h'],ts],dim=1)
        zs = self.initial_embedding(z_init)
        hs,vs,xs=self.convs[0](data,zs,x_init,v_init)
        for i in range(1,len(self.convs)):
            hs,vs,xs=self.convs[i](data,hs,xs,vs)
        vector_field=self.position_updater(hs,vs)
        return vector_field