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
from scipy.stats import vonmises





# --- Dataset class with x1 as prior (noise), x0 as true data ---
class AA2GraphDataset(dgl.data.DGLDataset):
    def __init__(self, data_path, split="train", max_atom_number=51,kabsch=False,biased=False):
        self.max_atoms = max_atom_number
        self.kabsch= kabsch

        # featurize pdbs
        self.directory = f"/{split}"
        feature_dir=self.directory
        if not os.path.exists(os.path.join(data_path, feature_dir)):
            feature_dir = 'train'  # fallback to train directory if split not found
        self.peptides,self.atom_types_dict, self.h_dict = aa2_featurizer(data_path, feature_dir)

        # load the numpy trajectories
        arr = np.load(os.path.join(data_path, f"all_{split}.npy"), allow_pickle=True).item()
        self.data = {pep: arr[pep] for pep in self.peptides}
        if biased:
            kappa = 10.0
            for pep in self.peptides:
                pdb_path=os.path.join(data_path,feature_dir,f"{pep}-traj-state0.pdb")
                topology=md.load_topology(pdb_path)
                n_atoms = self.h_dict[pep].shape[0]
                coords= self.data[pep].reshape(-1, n_atoms, 3)
                pep_traj = md.Trajectory(coords, topology)
                phi = md.compute_phi(pep_traj)[1].flatten()
                weights=150*vonmises.pdf(phi-1., kappa)+1
                weighted_idx=np.random.choice(np.arange(len(self.data[pep])), len(self.data[pep]), p=weights/weights.sum(), replace=True)
                self.data[pep] = self.data[pep][weighted_idx]
        self.samples = [(pep, frame) for pep in self.peptides for frame in range(len(self.data[pep]))] 
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

    #taken from https://hunterheidenreich.com/posts/kabsch_algorithm/, thank you Hunter Heidenreich!
    def kabsch_torch(self,P, Q):
        """
        Computes the optimal rotation and translation to align two sets of points (P -> Q),
        and their RMSD.
        :param P: A Nx3 matrix of points
        :param Q: A Nx3 matrix of points
        :return: A tuple containing the optimal rotation matrix, the optimal
                translation vector, and the RMSD.
        """
        assert P.shape == Q.shape, "Matrix dimensions must match"

        # Compute centroids
        centroid_P = torch.mean(P, dim=0)
        centroid_Q = torch.mean(Q, dim=0)

        # Center the points
        p = P - centroid_P
        q = Q - centroid_Q

        # Compute the covariance matrix
        H = torch.matmul(p.transpose(0, 1), q)

        # SVD
        U, S, Vt = torch.linalg.svd(H)

        # Validate right-handed coordinate system
        if torch.det(torch.matmul(Vt.transpose(0, 1), U.transpose(0, 1))) < 0.0:
            Vt[:, -1] *= -1.0

        # Optimal rotation
        R = torch.matmul(Vt.transpose(0, 1), U.transpose(0, 1))

        # RMSD
        #rmsd = torch.sqrt(torch.sum(torch.square(torch.matmul(p, R.transpose(0, 1)) - q)) / P.shape[0])

        return R


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
        if 'gen' not in self.directory:
            x_true = torch.from_numpy(self.data[pep][frame]).float()
        else:
            x_true = self.data[pep][frame].float()
            x_true = x_true*30 # convert to Angstroms
        x_true = x_true.view(-1, 3)  # ensure shape is (n_atoms, 3)
        #mean center the coordinates
        x_true = x_true - x_true.mean(dim=0, keepdim=True)
        '''pad_len = (self.max_atoms - n_real) * 3
        x_true = torch.nn.functional.pad(x_true, (0, pad_len)).view(self.max_atoms, 3)'''

        # x1: prior noise
        x_noise = torch.randn_like(x_true)  # random noise in [0,1]
        # x_noise = resample_noise()
        #x_noise = torch.nn.functional.pad(x_noise, (0, pad_len)).view(self.max_atoms, 3)

        if self.kabsch:
            # Kabsch algorithm to align x_true and x_noise
            R = self.kabsch_torch(x_true, x_noise)
            x_true = torch.matmul(x_true, R.transpose(0, 1))

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
def get_aa2_dataloader(data_path=None,batch_size=512,shuffle=True,num_workers=8,kabsch=False,split="train",biased=False):
    dataset = AA2GraphDataset(data_path=data_path, split=split, max_atom_number=51,kabsch=kabsch, biased=biased)
    dataloader = dgl.dataloading.GraphDataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

if __name__ == "__main__":
    main()