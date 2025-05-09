import os
import torch
import numpy as np
import mdtraj as md
import tqdm

n_dimensions = 3


def get_aa2_dataset():
    data_path='data/2AA-1-large'
    train_directory =  "/train"
    val_directory =  "/val"
    max_atom_number = 51
    train_peptides,train_atom_types_dict, train_h_dict = aa2_featurizer(data_path,train_directory)
    val_peptides,val_atom_types_dict, val_h_dict = aa2_featurizer(data_path,val_directory)
    data = np.load(data_path + "/all_train.npy", allow_pickle=True).item()
    data_val = np.load(data_path + "/all_val.npy", allow_pickle=True).item()
    n_data = len(data[train_peptides[0]])
    n_random = n_data // 10

def aa2_featurizer(data_path,directory):
    peptides=[]
    for file in os.listdir(data_path+directory):
        filename = os.fsdecode(file)
        if filename.endswith(".pdb"):
            peptides.append(filename[:2])
    max_atom_number = 0
    atom_dict = {"H": 0, "C": 1, "N": 2, "O": 3, "S": 4}
    scaling = 30
    topologies = {}
    atom_types_dict = {}
    h_dict = {}
    n_encodings = 76
    atom_types_ecoding = np.load(
        data_path + "/atom_types_ecoding.npy", allow_pickle=True
    ).item()

    amino_dict = {
        "ALA": 0,
        "ARG": 1,
        "ASN": 2,
        "ASP": 3,
        "CYS": 4,
        "GLN": 5,
        "GLU": 6,
        "GLY": 7,
        "HIS": 8,
        "ILE": 9,
        "LEU": 10,
        "LYS": 11,
        "MET": 12,
        "PHE": 13,
        "PRO": 14,
        "SER": 15,
        "THR": 16,
        "TRP": 17,
        "TYR": 18,
        "VAL": 19,
    }
    for peptide in tqdm.tqdm(peptides):

        topologies[peptide] = md.load_topology(
            data_path + directory+f"/{peptide}-traj-state0.pdb"
        )
        n_atoms = len(list(topologies[peptide].atoms))
        atom_types = []
        amino_idx = []
        amino_types = []
        for i, amino in enumerate(topologies[peptide].residues):

            for atom_name in amino.atoms:
                amino_idx.append(i)
                amino_types.append(amino_dict[amino.name])
                if atom_name.name[0] == "H" and atom_name.name[-1] in ("1", "2", "3"):
                    if amino_dict[amino.name] in (8, 13, 17, 18) and atom_name.name[:2] in (
                        "HE",
                        "HD",
                        "HZ",
                        "HH",
                    ):
                        pass
                    else:
                        atom_name.name = atom_name.name[:-1]
                if atom_name.name[:2] == "OE" or atom_name.name[:2] == "OD":
                    atom_name.name = atom_name.name[:-1]
                atom_types.append(atom_name.name)
        atom_types_dict[peptide] = np.array(
            [atom_types_ecoding[atom_type] for atom_type in atom_types]
        )
        atom_onehot = torch.nn.functional.one_hot(
            torch.tensor(atom_types_dict[peptide]), num_classes=len(atom_types_ecoding)
        )
        amino_idx_onehot = torch.nn.functional.one_hot(
            torch.tensor(amino_idx), num_classes=2
        )
        amino_types_onehot = torch.nn.functional.one_hot(
            torch.tensor(amino_types), num_classes=20
        )

        h_dict[peptide] = torch.cat(
            [amino_idx_onehot, amino_types_onehot, atom_onehot], dim=1
        )
    return peptides,atom_types_dict,h_dict

