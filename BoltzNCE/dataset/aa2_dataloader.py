import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import dgl
import mdtraj as md

from torch.utils.data import Dataset, DataLoader
from bgflow import MeanFreeNormalDistribution

# reuse your featurizer
from aa2_dataset import aa2_featurizer  





# --- Dataset class with x1 as prior (noise), x0 as true data ---
class AA2GraphDataset(Dataset):
    def __init__(self, data_path, split="train", max_atom_number=51, device=None):
        self.max_atoms = max_atom_number
        self.device = device or torch.device("cpu")

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
        self.priors = {}
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
        self.edge_index = (src, dst)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pep, frame = self.samples[idx]
        n_real = self.h_dict[pep].shape[0]

        # h features (pad to max_atoms)
        h = self.h_dict[pep]
        if n_real < self.max_atoms:
            pad = torch.zeros(self.max_atoms - n_real, h.size(1))
            h = torch.cat([h, pad], dim=0)

        # x0: true coordinates
        x_true = torch.from_numpy(self.data[pep][frame]).float()
        pad_len = (self.max_atoms - n_real) * 3
        x_true = torch.nn.functional.pad(x_true, (0, pad_len)).view(self.max_atoms, 3)

        # x1: prior noise
        x_noise = self.priors[pep].sample(1).reshape(-1)
        # x_noise = resample_noise()
        x_noise = torch.nn.functional.pad(x_noise, (0, pad_len)).view(self.max_atoms, 3)

        # build graph
        g = dgl.graph(self.edge_index, num_nodes=self.max_atoms)
        g.ndata["h"]  = h
        g.ndata["x0"] = x_true
        g.ndata["x1"] = x_noise

        # node & edge masks
        node_mask = torch.zeros(self.max_atoms, dtype=torch.bool)
        node_mask[:n_real] = True
        g.ndata["mask"] = node_mask
        src, dst = g.edges()
        g.edata["mask"] = node_mask[src] & node_mask[dst]

        return g,pep,n_real
# --- 3) Tests ---
def run_tests(ds):
    print("Running initial tests...")
    # single graph
    g0, pep0, n0 = ds[0]
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
    print("✔️ All tests passed.\n")


def compare_dataloader_to_raw(data_path, split="train", max_atoms=51, n_tests=10):
    ds = AA2GraphDataset(data_path, split, max_atom_number = max_atoms)
    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: batch  # return list of (g, pep, n)
    )
    mismatches = []
    for i, batch in enumerate(loader):
        if i >= n_tests:
            break
        g, pep, n_real = batch[0]
        # raw coords from .npy
        raw_flat = ds.data[pep][i]  # assume frame index = i
        raw_coords = raw_flat.reshape(n_real, 3)
        # loader x0 coords
        loader_coords = g.ndata["x0"][:n_real].numpy()
        # compare
        if not np.allclose(raw_coords, loader_coords, atol=1e-6):
            diff = np.max(np.abs(raw_coords - loader_coords))
            mismatches.append((i, pep, diff))
    if not mismatches:
        print(f"All {n_tests} samples match exactly between raw .npy and DataLoader x0.")
    else:
        print("Mismatches found:")
        for idx, pep, d in mismatches:
            print(f" Sample {idx} (pep {pep}): max abs diff = {d:.3e}")
# --- 4) Main execution ---
def main():
    data_path = "../../data/2AA-1-large"
    ds = AA2GraphDataset(data_path, split="train", max_atom_number=51)

    # run the tests
    run_tests(ds)
    compare_dataloader_to_raw(data_path, split="train", max_atoms=51, n_tests=10)

    # fetch first example via DataLoader
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0,
                        collate_fn=lambda batch: batch)
    batch = next(iter(loader))
    g0, pep0, n0 = batch[0]

    # raw coords from numpy
    raw_flat   = ds.data[pep0][0]
    raw_coords = raw_flat.reshape(n0, 3)

    # loader coords from x0
    loaded_coords = g0.ndata["x0"][:n0].cpu().numpy()
    noisy_coords = g0.ndata["x1"][:n0].cpu().numpy()

    # write PDBs
    topo = md.load_topology(f"{data_path}/train/{pep0}-traj-state0.pdb")
    traj_raw    = md.Trajectory(raw_coords[None], topo)
    traj_loaded = md.Trajectory(loaded_coords[None], topo)

    out_dir = "./"
    os.makedirs(out_dir, exist_ok=True)
    raw_pdb    = os.path.join(out_dir, "raw_first.pdb")
    loaded_pdb = os.path.join(out_dir, "loaded_first.pdb")
    traj_raw.save_pdb(raw_pdb)
    traj_loaded.save_pdb(loaded_pdb)

    # visualize and save comparison PNG
    fig = plt.figure(figsize=(8, 4))
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.scatter(raw_coords[:,0], raw_coords[:,1], raw_coords[:,2], alpha=0.7)
    ax1.set_title("Raw coords")
    ax2 = fig.add_subplot(122, projection="3d")
    ax2.scatter(loaded_coords[:,0], loaded_coords[:,1], loaded_coords[:,2], alpha=0.7)
    ax2.set_title("Loaded x0 coords")
    plt.tight_layout()
    png_path = os.path.join(out_dir, "loaded_vs_data_comparison.png")
    plt.savefig(png_path)
    
    fig = plt.figure(figsize=(8, 4))
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.scatter(loaded_coords[:,0], loaded_coords[:,1], loaded_coords[:,2], alpha=0.7,label="loaded")
    
    ax1.set_title("Raw coords")
    ax2 = fig.add_subplot(122, projection="3d")
    ax1.scatter(noisy_coords[:,0], noisy_coords[:,1], noisy_coords[:,2], alpha=0.1,label = "noisy")
    ax2.set_title("Loaded x0 coords")
    plt.tight_layout()
    plt.legend()
    png_path = os.path.join(out_dir, "loaded_vs_sampled_noise_comparison.png")
    plt.savefig(png_path)
    print(f"Saved comparison plot to: {png_path}")
    print(f"Raw coords PDB:         {raw_pdb}")
    print(f"Loaded coords PDB:      {loaded_pdb}")

if __name__ == "__main__":
    main()