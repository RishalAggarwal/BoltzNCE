import MDAnalysis
from MDAnalysis.analysis.align import AlignTraj
import mdtraj as md
import os
import torch
import dgl
import numpy as np

class AlaninesysGraphDataset(dgl.data.DGLDataset):
    def __init__(self, data_path, split='AAAAAA',coords=None):
        super(AlaninesysGraphDataset, self).__init__(name='alaninesys_graph_dataset')
        self.topology, self.h_initial = alaninesys_featurizer(data_path, split)
        data_path = data_path + split
        u=MDAnalysis.Universe(data_path +'/'+split+'.prmtop', [data_path + '/' + split.lower()+'_1.nc'])  
        if coords is not None:
            self.coords = np.load(data_path + '/' + coords)
            self.coords = torch.from_numpy(self.coords)
        else:   
            coords =[]
            for ts in u.trajectory:
                coords.append(u.select_atoms('all').positions)
            coords = np.array(coords)
            #take first 300 ns
            self.coords = torch.from_numpy(coords[:30000])
        #mean center the coordinates
        self.coords = self.coords - self.coords.mean(dim=1, keepdim=True)
        
        self.n_particles = self.h_initial.shape[0]
        self.nodes=torch.arange(self.h_initial.shape[0])
        self.edges=torch.cartesian_prod(self.nodes,self.nodes)
        self.edges=self.edges[self.edges[:,0]!=self.edges[:,1]].transpose(0,1)

    def __len__(self):
        return len(self.coords)
    
    def __getitem__(self, idx):
        g = dgl.graph((self.edges[0].cpu(),self.edges[1].cpu()),num_nodes=self.n_particles)
        g.ndata['h'] = self.h_initial
        g.ndata['x'] = self.coords[idx]
        return g


def get_alaninesys_dataset(data_path=None,batch_size=512,shuffle=True,num_workers=8,kabsch=False,split="train",biased=False,coords=None):
    dataset = AlaninesysGraphDataset(data_path=data_path, split=split,coords=coords)
    dataloader = dgl.dataloading.GraphDataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

def alaninesys_featurizer(data_path, split='AAAAAA'):
    data_path = data_path + split
    u=MDAnalysis.Universe(data_path +'/'+split+'.prmtop', [data_path + '/' + split.lower()+'_1.nc']) 
    topology = md.load_prmtop(data_path + '/' + split+'.prmtop')
    atom_types = []
    amino_idx = []
    amino_types = []
    for i, amino in enumerate(topology.residues):
        for atom_name in amino.atoms:
            amino_idx.append(i)
            amino_types.append(amino.name)
            if atom_name.name[0] == "H" and atom_name.name[-1] in ("1", "2", "3"):
                atom_name.name = atom_name.name[:-1]
            if atom_name.name[:2] == "OE" or atom_name.name[:2] == "OD":
                atom_name.name = atom_name.name[:-1]
            atom_types.append(atom_name.name)
    amino_idx_one_hot = torch.nn.functional.one_hot(
                torch.tensor(amino_idx), num_classes=max(amino_idx) + 1
            )
    unique_atoms = sorted(set(atom_types))
    atom_to_index = {atom: i for i, atom in enumerate(unique_atoms)}

    # Step 2: Convert to indices
    indices = torch.tensor([atom_to_index[atom] for atom in atom_types])

    # Step 3: One-hot encode
    atom_type_one_hot = torch.nn.functional.one_hot(indices, num_classes=len(unique_atoms)) 
    final_one_hot = torch.cat((amino_idx_one_hot, atom_type_one_hot), dim=1)
    h_initial = final_one_hot
    return topology,h_initial
