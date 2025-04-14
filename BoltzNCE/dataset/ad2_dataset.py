import torch
import dgl
from bgmol.datasets import AImplicitUnconstrained
import mdtraj as md
from scipy.stats import vonmises
from bgflow.utils import remove_mean
import numpy as np
import os

def get_alanine_traj():
    download=False
    if not os.path.exists("AImplicitUnconstrained/traj0.h5"):
        download=True
    dataset = AImplicitUnconstrained(read=True,download=download)
    ala_traj = md.Trajectory(dataset.xyz, dataset.system.mdtraj_topology)
    return ala_traj

def get_alanine_dataset():
    n_particles = 22
    n_dimensions = 3
    dim = n_particles * n_dimensions
    scaling = 10
    data_smaller = torch.from_numpy(np.load("../data/AD2_relaxed_weighted.npy")).float()/10
    data_smaller = remove_mean(data_smaller, n_particles, n_dimensions).reshape(-1, dim) * scaling
    return data_smaller

def get_alanine_features():
    ala_traj=get_alanine_traj()
    atom_dict = {"H": 0, "C":1, "N":2, "O":3}
    atom_types = []
    for atom_name in ala_traj.topology.atoms:
        atom_types.append(atom_name.name[0])
    atom_types = np.array([atom_dict[atom_type] for atom_type in atom_types])
    atom_types[[4,6,8,14,16]] = np.arange(4, 9)
    atom_types_train = np.arange(22)
    atom_types_train[[1, 2, 3]] = 2
    atom_types_train[[19, 20, 21]] = 20
    atom_types_train[[11, 12, 13]] = 12
    h_initial = torch.nn.functional.one_hot(torch.tensor(atom_types_train))
    return atom_types, h_initial

def get_alanine_types_dataset_dataloaders(dataset=None,batch_size=512,shuffle=True,num_workers=8,scaling=1.0):
    if dataset is None:
        dataset = get_alanine_dataset()
    atom_types, h_initial = get_alanine_features()
    dataset = alanine_dataset(dataset,h_initial,scaling)
    dataloader = dgl.dataloading.GraphDataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return atom_types,h_initial,dataset,dataloader


class alanine_dataset(dgl.data.DGLDataset):
    def __init__(self, dataset,h_initial,scaling=1.0):
        self.dataset = dataset
        original_shape = dataset.shape
        self.n_particles = 22
        self.n_dimensions = 3
        self.dim = self.n_particles * self.n_dimensions
        self.dataset = remove_mean(self.dataset, self.n_particles, self.n_dimensions).reshape(-1, self.dim)
        self.dataset = self.dataset/scaling
        self.dataset = self.dataset.reshape(original_shape)
        self.n_samples = len(dataset)
        self.h_intial = h_initial.cpu()
        self.nodes=torch.arange(self.n_particles)
        self.edges=torch.cartesian_prod(self.nodes,self.nodes)
        self.edges=self.edges[self.edges[:,0]!=self.edges[:,1]].transpose(0,1)
        self.graphs=[]
        for i in range(self.n_samples):
            g = dgl.graph((self.edges[0].cpu(),self.edges[1].cpu()),num_nodes=self.n_particles)
            g.ndata['x'] = torch.empty((self.n_particles, self.n_dimensions))
            g.ndata['h'] = self.h_intial
            self.graphs.append(g)
    def __len__(self):
        return self.n_samples
    def __getitem__(self, idx):
        self.graphs[idx].ndata['x'] = self.dataset[idx].reshape(self.n_particles, self.n_dimensions)
        return self.graphs[idx]