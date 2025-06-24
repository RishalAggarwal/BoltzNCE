import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import dgl
import mdtraj as md
import tqdm
from torch.utils.data import Dataset, DataLoader
from bgflow import MeanFreeNormalDistribution
from .aa2_dataset import aa2_featurizer





# --- Dataset class with x1 as prior (noise), x0 as true data ---
class AA2GraphDataset(dgl.data.DGLDataset):
    def __init__(self, data_path, split="train", max_atom_number=51):
        self.max_atoms = max_atom_number

        # featurize pdbs
        directory = f"/{split}"
        self.peptides,self.atom_types_dict, self.h_dict = aa2_featurizer(data_path, directory)

        # load the numpy trajectories
        arr = np.load(os.path.join(data_path, f"all_{split}.npy"), allow_pickle=True).item()
        self.data = {pep: arr[pep] for pep in self.peptides}


        # build (pep, frame) index list
        n_frames = len(next(iter(self.data.values())))
        self.samples = [(pep, frame) for pep in self.peptides for frame in range(n_frames)]

        # per-peptide prior distributions
        '''self.priors = {}
        for pep in self.peptides:
            n_atoms = self.h_dict[pep].shape[0]
            prior = MeanFreeNormalDistribution(
                n_atoms * 3,
                n_atoms,
                two_event_dims=False
            )
            self.priors[pep] = prior

        # full-graph edges (no self-loops)
        N = self.max_atoms
        mask = ~torch.eye(N, dtype=torch.bool)
        src, dst = torch.nonzero(mask, as_tuple=True)
        self.edge_index = (src, dst)'''



    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pep, frame = self.samples[idx]
        n_real = self.h_dict[pep].shape[0]

        # h features (pad to max_atoms)
        h = self.h_dict[pep]
        '''if n_real < self.max_atoms:
            pad = torch.zeros(self.max_atoms - n_real, h.size(1))
            h = torch.cat([h, pad], dim=0)'''

        # x0: true coordinates
        x_true = torch.from_numpy(self.data[pep][frame]).float()
        x_true = x_true.view(-1, 3)  # ensure shape is (n_atoms, 3)
        '''pad_len = (self.max_atoms - n_real) * 3
        x_true = torch.nn.functional.pad(x_true, (0, pad_len)).view(self.max_atoms, 3)'''

        # x1: prior noise
        x_noise = torch.randn_like(x_true)  # random noise in [0,1]
        # x_noise = resample_noise()
        #x_noise = torch.nn.functional.pad(x_noise, (0, pad_len)).view(self.max_atoms, 3)

        nodes=torch.arange(n_real)
        edges=torch.cartesian_prod(nodes,nodes)
        edges=edges[edges[:,0]!=edges[:,1]].transpose(0,1)

        # build graph
        g = dgl.graph((edges[0].cpu(),edges[1].cpu()), num_nodes=n_real)
        g.ndata["h"]  = h
        g.ndata["x0"] = x_true
        g.ndata["x1"] = x_noise

        # node & edge masks
        '''node_mask = torch.zeros(self.max_atoms, dtype=torch.bool)
        node_mask[:n_real] = True
        g.ndata["mask"] = node_mask
        src, dst = g.edges()
        g.edata["mask"] = node_mask[src] & node_mask[dst]'''

        return g
# --- 3) Tests ---
'''def run_tests(ds):
    print("Running initial tests...")
    # single graph
    g0 = ds[0]
    assert g0.num_nodes() == ds.max_atoms
    assert g0.num_edges() == ds.max_atoms * (ds.max_atoms - 1)
    for key in ("h", "x0", "x1", "mask"):
        assert key in g0.ndata, f"Missing node feature {key}"
    assert g0.ndata["x0"].shape == (ds.max_atoms, 3)
    # DataLoader batching test
    loader_test = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0,
                             collate_fn=lambda batch: dgl.batch([g for g,_,_ in batch]))
    bg = next(iter(loader_test))
    assert bg.num_nodes() == 4 * ds.max_atoms
    print("✔️ All tests passed.\n")'''


# --- 4) Main execution ---
def main():
    data_path = "../../data/2AA-1-large"
    ds = AA2GraphDataset(data_path, split="train", max_atom_number=51)

    # run the tests
    #run_tests(ds)

    # fetch first example via DataLoader
    loader = dgl.dataloading.GraphDataLoader(ds, batch_size=512, shuffle=True, num_workers=8)
    for  batch in tqdm.tqdm(loader):
        continue
        # visualize the first graph in the batch
def get_ad2_dataloader(data_path=None,batch_size=512,shuffle=True,num_workers=8):
    dataset = AA2GraphDataset(data_path=data_path, split="train", max_atom_number=51)
    dataloader = dgl.dataloading.GraphDataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

if __name__ == "__main__":
    main()