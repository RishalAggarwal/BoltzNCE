import numpy as np
import torch
import MDAnalysis 
import parmed
import mdtraj as md
import argparse
import wandb
import sys
import openmm
from openmm.unit import *
from openmm.app import GBn2, HCT
sys.path.append("./BoltzNCE/")
from bgflow.utils import as_numpy
from BoltzNCE.utils.arguments import get_args
from BoltzNCE.utils.utils import load_models
from BoltzNCE.utils.tbg_utils import create_adjacency_list, find_chirality_centers, compute_chirality_sign, check_symmetry_change
from BoltzNCE.dataset.alsys_dataloader import alaninesys_featurizer
import networkx.algorithms.isomorphism as iso
import networkx as nx
from bgflow import OpenMMBridge, OpenMMEnergy
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from infer_aa2 import gen_samples,get_potential_logp, align_topology, align_samples,fix_chirality,update_interpolant_args
from infer_aa2 import plot_fes,phi_to_grid,compute_free_energy_difference,calc_energy_w2,calc_torsion_w2,calculate_w2_distances
from bgflow.bg import sampling_efficiency


def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, default=None)
    p.add_argument("--divergence", action="store_true",default=True)
    p.add_argument("--no-divergence", action="store_false", dest="divergence")
    p.add_argument('--n_samples', type=int, default=500)
    p.add_argument('--n_sample_batches', type=int, default=200)
    p.add_argument('--wandb_inference_name', type=str, default=None)
    p.add_argument('--save_generated',action='store_true', default=False)
    p.add_argument('--save_prefix',type=str, default='./generated/')
    p.add_argument("--rtol", type=float, default=1e-4,
                   help="relative tolerance for ODE/SDE solver")
    p.add_argument("--atol", type=float, default=1e-4,
                   help="absolute tolerance for ODE/SDE solver")
    p.add_argument("--tmin",    type=float, default=0.0,
                   help="endpoint of the integration time span")
    p.add_argument('--integration_method', type=str, default='dopri5')
    #p.add_argument('--data_path',type=str,default="data/",required=False)
    #p.add_argument('--split',type=str,default="AAAA",required=False)
    args=p.parse_args()

    return args,p

def plot_energy_histograms(classical_target_energies, classical_model_energies, log_w_np,prefix=''):
    plt.figure(figsize=(16,9))
    range_limits = (classical_target_energies.min()-10,classical_target_energies.max()+100)
    plt.hist(classical_target_energies, bins=100, alpha=0.5, range=range_limits,density=True, label="MD")
    plt.hist(classical_model_energies, bins=100,alpha=0.5, range=range_limits,density=True, label="Boltz Emulator")
    plt.hist(classical_model_energies, bins=100, range=range_limits,density=True, label="Boltz Emulator reweighted", weights=np.exp(log_w_np), histtype='step', linewidth=5)
    plt.legend(fontsize=30)
    plt.xlabel("Energy  / $k_B T$", fontsize=45)  
    plt.yticks(fontsize=25)
    plt.xticks(fontsize=25)

    plt.title(f"Classical energy distribution", fontsize=45)
    wandb.log({prefix+"Classical energy distribution": wandb.Image(plt)})
    plt.close()

def process_gen_samples(samples_np,dlogf_np,scaling,topology,adj_list,atom_types, args, prefix=''):
    traj_samples_aligned=md.Trajectory(samples_np.reshape(-1, dim//3, 3), topology=topology)
    #aligned_samples, aligned_idxs = align_samples(samples_np, adj_list, dim, atom_types, scaling)
    #traj_samples_aligned = md.Trajectory(samples_np, topology=topology)
    model_samples = torch.from_numpy(traj_samples_aligned.xyz)
    data_path = args['data_path']  + args['split']
    u=MDAnalysis.Universe(data_path+'/'+args['split']+'.prmtop', [data_path + '/' + args['split'].lower()+'_19.nc'])  
    coords =[]
    for ts in u.trajectory:
        coords.append(u.select_atoms('all').positions)
    coords = np.array(coords)
    coords = torch.from_numpy(coords)
    #mean center the coordinates
    coords = coords - coords.mean(dim=1, keepdim=True)
    coords = coords.numpy()
    data=coords
    model_samples,symmetry_change=fix_chirality(model_samples, adj_list, atom_types, data, dim)
    traj_samples=md.Trajectory(as_numpy(model_samples)[~symmetry_change], topology=topology)
    return model_samples,data,symmetry_change,traj_samples

def plot_ramachandrans(traj_samples,plot_name=''):
    phis= md.compute_phi(traj_samples)
    psis= md.compute_psi(traj_samples)
    for i in range(phis[1].shape[1]):
        fig, ax = plt.subplots(figsize=(9, 9))
        plot_range = [-np.pi, np.pi]
        h, x_bins, y_bins, im = ax.hist2d(phis[1][:,i], psis[1][:,i], 100, norm=LogNorm(), range=[plot_range,plot_range],rasterized=True)
        ax.set_xlabel(r"$\varphi$", fontsize=45)
        ax.set_ylabel(r"$\psi$", fontsize=45)
        ax.set_title(plot_name, fontsize=45)
        ax.xaxis.set_tick_params(labelsize=25)
        ax.yaxis.set_tick_params(labelsize=25)
        ax.set_yticks([])
        wandb.log({plot_name + r"$\varphi$"+ str(i+1) +" Ramachandran plot": wandb.Image(plt)})
        plt.close(fig)

def compute_metrics(data,dlogf_np, topology, model_samples, n_atoms, n_dimensions,  symmetry_change, traj_samples,args,prefix='',threshold=0):
    plot_ramachandrans(traj_samples, plot_name=prefix+'Emulator')
    traj_samples_data = md.Trajectory(data.reshape(-1, n_atoms, 3), topology=topology)
    plot_ramachandrans(traj_samples_data, plot_name=prefix+'MD')
    data_path = args['data_path'] + args['split']
    prmtop = openmm.app.AmberPrmtopFile(data_path + '/' + args['split']+'.prmtop')
    forcefield = openmm.app.ForceField('amber14/protein.ff15ipq.xml', 'implicit/gbn2.xml')
    system = forcefield.createSystem(prmtop.topology, nonbondedMethod=openmm.app.NoCutoff, constraints=openmm.app.HBonds)
    integrator = openmm.LangevinMiddleIntegrator(300*openmm.unit.kelvin, 1/openmm.unit.picosecond, 2*openmm.unit.femtosecond)
    openmm_energy = OpenMMEnergy(bridge=OpenMMBridge(system, integrator, platform_name="CUDA"))
    classical_model_energies = as_numpy(openmm_energy.energy(model_samples.reshape(-1, dim)[~symmetry_change]/10))
    classical_target_energies = as_numpy(openmm_energy.energy(torch.from_numpy(data).reshape(-1, dim)/10))
    idxs = np.arange(len(model_samples))[~symmetry_change]
    log_w_np = -classical_model_energies - as_numpy(dlogf_np.reshape(-1,1)[idxs])
    log_w_torch=torch.tensor(log_w_np)
    log_w_torch=log_w_torch - torch.logsumexp(log_w_torch,dim=(0,1))
    log_w_np=log_w_torch.numpy()
    if threshold>0:
        threshold_index=len(classical_model_energies) - int(threshold*len(classical_model_energies)/100)
        classical_model_energies=classical_model_energies[np.argsort(log_w_np,axis=0)[:,0],:]
        classical_model_energies=classical_model_energies[:threshold_index,:]
        model_samples=model_samples[np.argsort(log_w_np,axis=0)[:,0],:]
        traj_samples=md.Trajectory(as_numpy(model_samples[~symmetry_change])[:threshold_index,:], topology=topology)
        log_w_np=np.sort(log_w_np,axis=0)
        log_w_np=log_w_np[:threshold_index]
    wandb.log({prefix + "Sampling efficiency Mean": sampling_efficiency(torch.from_numpy(log_w_np)).item()})
    plot_energy_histograms(classical_target_energies, classical_model_energies, log_w_np, prefix=prefix)
    phis_data= md.compute_phi(traj_samples_data)[1]
    phis= md.compute_phi(traj_samples)[1]
    for i in range(phis.shape[1]):
        compute_free_energy_difference(phis[:,i], log_w_np, prefix=prefix+'phi '+str(i+1))
        compute_free_energy_difference(phis_data[:,i], np.zeros((phis_data[:,i].shape[0],1)), prefix='MD phi '+str(i+1))
        grid_left_data, fes_left_data, grid_right_data, fes_right_data = phi_to_grid(phis_data[:,i].flatten())
        grid_left, fes_left, grid_right, fes_right = phi_to_grid(phis[:,i].flatten())
        grid_left_weighted, fes_left_weighted, grid_right_weighted, fes_right_weighted = phi_to_grid(phis[:,i].flatten(), weights=np.exp(log_w_np))
        plt.figure(figsize=(16,9))
        plt.plot(np.hstack([grid_left_data, grid_right_data]), np.hstack([fes_left_data, fes_right_data]), linewidth=5, label="MD")
        plt.plot(np.hstack([grid_left, grid_right]), np.hstack([fes_left, fes_right]), linewidth=5, linestyle="--", label="Emulator")
        plt.plot(np.hstack([grid_left_weighted, grid_right_weighted]), np.hstack([fes_left_weighted, fes_right_weighted]), linewidth=5, linestyle="--", label="Emulator reweighted")
        plt.legend(fontsize=30)
        plt.title(r"Free energy projection $\varphi$ "+str(i+1), fontsize=45)
        plt.xlabel(r"$\varphi$", fontsize=45)
        plt.ylabel("Free energy / $k_B T$", fontsize=45)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        wandb.log({prefix + "Free energy projection phi "+str(i+1): wandb.Image(plt)})
        plt.close()

if __name__ == "__main__":
    args,p=parse_arguments()
    args=get_args(args,p)
    wandb.init(project=args['wandb_project'], name=args['wandb_inference_name'])
    wandb.config.update(args)
    scaling=30.0
    args['data_path']=args['dataloader']['data_path']
    args['split']=args['dataloader']['split']

    topology,h_initial=alaninesys_featurizer(args['data_path'], split=args['split'])
    adj_list = torch.from_numpy(np.array([(b.atom1.index, b.atom2.index) for b in topology.bonds], dtype=np.int32))
    atom_dict = {"C": 0, "H":1, "N":2, "O":3, "S":4}
    #["C", "H", "N", "O", "S"]
    atom_types = []
    for atom_name in topology.atoms:
        atom_types.append(atom_name.name[0])
    atom_types = torch.from_numpy(np.array([atom_dict[atom_type] for atom_type in atom_types]))
    dim = h_initial.shape[0] * 3
    n_particles = h_initial.shape[0]
    args['dim'] = dim
    update_interpolant_args(args)
    
    potential_model, vector_field, interpolant_obj = load_models(args,h_initial=h_initial,potential=args['model_type']=='potential')
    interpolant_obj.integration_method=args['integration_method']
    if potential_model is not None:
        potential_model.eval()
    vector_field.eval()

    integral_type = 'ode'
    if args['divergence']==True:
        integral_type='ode_divergence'
    print("########## generating initial samples")
    samples_np,dlogf_np=gen_samples(n_samples=args['n_samples'],n_sample_batches=args['n_sample_batches'],interpolant_obj=interpolant_obj,integral_type=integral_type)
    if potential_model is not None:
        dlogf_np= get_potential_logp(samples_np, interpolant_obj)

    model_samples,data,symmetry_change,traj_samples = process_gen_samples(samples_np, dlogf_np, scaling, topology, adj_list, atom_types, args)
    compute_metrics(data,dlogf_np, topology, model_samples, n_particles, dim, symmetry_change, traj_samples,args,prefix='',threshold=0)

    if args['save_generated']:
        numpy_dict={
            'samples': as_numpy(model_samples),
            'dlogf': dlogf_np,
        }
        np.savez(args['save_prefix'] + '_numpy_dict.npz', **numpy_dict)