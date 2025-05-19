import time
import tqdm
import numpy as np
import torch
import dgl
import os
from bgflow.utils import remove_mean
from bgflow.utils import distances_from_vectors
from bgflow.utils import distance_vectors, as_numpy
from bgflow.bg import sampling_efficiency,unormalized_nll,effective_sample_size
from bgflow import XTBEnergy, XTBBridge
from models.interpolant import Interpolant
from models.ebm import GVP_EBM
from dataset.ad2_dataset import get_alanine_atom_types,get_alanine_implicit_dataset,get_alanine_features
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import mdtraj as md
import argparse
import wandb
from utils.arguments import get_args
from utils.utils import load_models
import ot


energies_holdout_file = "../data/AD2_relaxed_weighted_all_energies.npy"
scaling=10
atom_types_xtb = get_alanine_atom_types()
temperature = 300
number_dict = {0: 1, 1:6, 2:7, 3:8}
numbers = np.array([number_dict[atom_type] for atom_type in atom_types_xtb])
target_xtb = XTBEnergy(
    XTBBridge(numbers=numbers, temperature=temperature, solvent="water"),
    two_event_dims=False
)
# energies_np = as_numpy(target_xtb.energy(torch.from_numpy(samples)/scaling))
energy_offset = 34600

    # if we already computed it, just load
if os.path.exists(energies_holdout_file):
    energies_data_holdout = np.load(energies_holdout_file)
    # if you need it back as a torch tensor:
    energies_data_holdout = torch.from_numpy(energies_data_holdout)
else:
    # otherwise compute it...
    data_holdout_xtb = torch.from_numpy(
        np.load("../data/AD2_relaxed_weighted.npy")
    ).reshape(-1, 66)
    energies_data_holdout = target_xtb.energy(data_holdout_xtb/ scaling)
    energies_data_holdout += energy_offset
    energies_data_holdout = energies_data_holdout.detach().cpu().numpy()
    np.save(energies_holdout_file, energies_data_holdout)
