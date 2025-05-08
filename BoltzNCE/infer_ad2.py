import time
import tqdm
import numpy as np
import torch
import dgl
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

num_particles = 22
n_dimensions = 3
dim = num_particles * n_dimensions


def parse_arguments():
    p =argparse.ArgumentParser()
    p.add_argument('--config', type=str, default=None)
    p.add_argument('--MCMC',type=bool, default=False)
    p.add_argument('--MCMC_steps',type=int, default=500)
    p.add_argument("--divergence", action="store_true",default=True)
    p.add_argument("--no-divergence", action="store_false", dest="divergence")
    p.add_argument("--SDE", action="store_true", default=False)
    p.add_argument('--weight_threshold', type=float, default=0.2)
    p.add_argument('--n_samples', type=int, default=500)
    p.add_argument('--n_sample_batches', type=int, default=200)
    p.add_argument('--wandb_inference_name', type=str, default=None)
    args=p.parse_args()
    return args,p

def gen_samples(n_samples,n_sample_batches,interpolant_obj: Interpolant,integral_type='ode',ess_threshold=0.75,n_timesteps=1000):
    
    time_start = time.time()
    
    latent_np = np.empty(shape=(0))
    samples_np = np.empty(shape=(0))
    log_w_np = np.empty(shape=(0))
    dlogp_all=[]
    #dlogf_all=[]
    #energies_np = np.empty(shape=(0))
    distances_x_np = np.empty(shape=(0))

    if integral_type=='Jarzynski':
        samples= interpolant_obj.Jarzynski_integral(n_samples*n_sample_batches,n_timesteps=n_timesteps,ess_threshold=ess_threshold)
        samples_np=samples.detach().cpu().numpy()
    else:
        for i in tqdm.tqdm(range(n_sample_batches)):
            with torch.no_grad():
                if integral_type=='sde':
                    samples= interpolant_obj.sde_integral(n_samples,n_timesteps=n_timesteps)
                elif integral_type=='ode':
                    samples=interpolant_obj.ode_integral(n_samples)
                elif integral_type=='PC':
                    samples=interpolant_obj.PC_integral(n_samples)
                elif integral_type=='ode_divergence':
                    samples,logp_samples=interpolant_obj.ode_divergence_integral(n_samples)
                    dlogp_all.append(logp_samples.cpu().detach().numpy())

                else:
                    raise ValueError("integral_type not recognized")
                
                #dlogf=interpolant_obj.log_prob_forward(samples)
                #latent = latent[0]
                #log_weights = bg.log_weights_given_latent(samples, latent, dlogp).detach().cpu().numpy()
                samples_np = np.append(samples_np, samples.detach().cpu().numpy())
                distances_x = distances_from_vectors(distance_vectors(samples.view(-1, num_particles, n_dimensions))).detach().cpu().numpy().reshape(-1)
                distances_x_np = np.append(distances_x_np, distances_x)
        
                #log_w_np = np.append(log_w_np, log_weights)
                #dlogf_all.append(dlogf.cpu().detach())
                #energies = target.energy(samples/scaling).detach().cpu().numpy()
                #energies_np = np.append(energies_np, energies)
    if len(dlogp_all)>0:
        dlogp_all = np.concatenate(dlogp_all, axis=0)
    samples_np = samples_np.reshape(-1, dim)
    samples_np = samples_np * interpolant_obj.scaling
    time_end = time.time()
    print("Sampling took {} seconds".format(time_end - time_start))
    wandb.log({"Sampling time": time_end - time_start})
    return samples_np,dlogp_all

def get_energies(samples,energies_holdout=True):
    scaling=10
    atom_types_xtb = get_alanine_atom_types()
    temperature = 300
    number_dict = {0: 1, 1:6, 2:7, 3:8}
    numbers = np.array([number_dict[atom_type] for atom_type in atom_types_xtb])
    target_xtb = XTBEnergy(
        XTBBridge(numbers=numbers, temperature=temperature, solvent="water"),
        two_event_dims=False
    )
    energies_np = as_numpy(target_xtb.energy(torch.from_numpy(samples)/scaling))
    energy_offset = 34600
    energies_np += energy_offset
    energies_data_holdout=None
    if energies_holdout:
        data_holdout_xtb = torch.from_numpy(np.load("../data/AD2_relaxed_holdout.npy")).reshape(-1, 66)
        energies_data_holdout = target_xtb.energy(data_holdout_xtb[::9]/scaling)
        energies_data_holdout += energy_offset
    #TODO implement jensen-shannon divergence for the energies
    return energies_np,energies_data_holdout

def get_potential_logp(model: Interpolant,samples):
    dlogf_all=[]
    samples_torch=torch.from_numpy(samples).float().cuda()
    samples_torch = samples_torch/model.scaling
    batch_size=1000
    i=0
    while (i+batch_size)<len(samples_torch):
        samples_prob=samples_torch[i:i+batch_size]
        dlogf=model.log_prob_forward(samples_prob)
        dlogf_all.append(dlogf.cpu().detach())
        i=i+batch_size
    samples_prob=samples_torch[i:len(samples_torch)]
    dlogf=model.log_prob_forward(samples_prob)
    dlogf_all.append(dlogf.cpu().detach())
    dlogf_all=torch.cat(dlogf_all,dim=0)
    dlogf_all=dlogf_all-torch.logsumexp(dlogf_all,dim=(0,1))
    wandb.log({"NLL_mean": -dlogf_all.mean()})
    wandb.log({"NLL_std": -dlogf_all.std()})
    dlogf_np=dlogf_all.cpu().detach().numpy()
    return dlogf_np

def compute_nll(model: Interpolant,samples_torch):
    samples_torch = samples_torch/model.scaling
    batch_size=200
    i=0
    nll_all=[]
    while (i+batch_size)<len(samples_torch):
        samples_prob=samples_torch[i:i+batch_size]
        nll=model.NLL_integral(samples_prob)
        nll_all.append(nll.cpu().detach())
        i=i+batch_size
    samples_prob=samples_torch[i:len(samples_torch)]
    nll=model.NLL_integral(samples_prob)
    nll_all.append(nll.cpu().detach())
    nll_all=torch.cat(nll_all,dim=0)
    wandb.log({"vector_NLL_mean": -nll_all.mean()})
    wandb.log({"vector_NLL_std": -nll_all.std()})
    nll_np=nll_all.cpu().detach().numpy()
    return nll_np

def get_importance_weights(dlogf_np,energies_np):
    energies_torch=torch.tensor(-energies_np)
    energies_w=energies_torch.numpy()
    #energies_torch=energies_torch - torch.logsumexp(energies_torch,dim=(0,1))
    #energies_w=energies_torch.numpy()
    log_w_np = as_numpy(energies_w).reshape(-1,1) - dlogf_np.reshape(-1,1)
    wandb.log({"Sampling efficiency Mean": sampling_efficiency(torch.from_numpy(log_w_np)).item()})
    log_w_torch=torch.tensor(log_w_np)
    log_w_torch=log_w_torch - torch.logsumexp(log_w_torch,dim=(0,1))
    log_w_np=log_w_torch.numpy()
    return log_w_np

def plot_energy_distributions(energies_data_holdout,samples_np,energies_np,log_w_np,weight_threshold=0,prefix=''):
    threshold_index=len(energies_np) - int(weight_threshold*len(energies_np)/100)
    energies_np=energies_np[np.argsort(log_w_np,axis=0)[:,0],:]
    energies_np=energies_np[:threshold_index,:]
    samples_np=samples_np[np.argsort(log_w_np,axis=0)[:,0],:]
    samples_np=samples_np[:threshold_index,:]
    #log_w_np=log_w_np[np.argsort(log_w_np,axis=0)[:,0],:]
    log_w_np=np.sort(log_w_np,axis=0)
    log_w_np=log_w_np[:threshold_index]
    log_w_torch=torch.tensor(log_w_np)
    log_w_torch=log_w_torch - torch.logsumexp(log_w_torch,dim=(0,1))
    log_w_np=log_w_torch.numpy()

    plt.figure(figsize=(16,9))

    plt.hist(as_numpy(energies_data_holdout), bins=100, alpha=0.5, density=True, label="MD relaxed samples", range=(-120, 0))
    #plt.hist(energies_np, bins=100,range=(-34710, -34610) ,alpha=0.5, density=True, label="BG");
    plt.hist(energies_np, bins=100,range=(-120, 0) ,alpha=0.5, density=True, label="Boltzmann Emulator")
    plt.hist(energies_np, bins=50, density=True,  range=(-120, 0), alpha=0.4, histtype='step', linewidth=4,
            color="r", label="BoltzNCE weighted samples", weights=np.exp(log_w_np))
    plt.xticks(fontsize=25) 
    plt.yticks(fontsize=25)
    plt.legend(fontsize=30)
    plt.xlabel(r"Energy / $k_B T$", fontsize=45)  
    plt.title("Energy distribution", fontsize=45)      
    wandb.log({prefix + "Energy distribution": wandb.Image(plt)})
    return samples_np,energies_np,log_w_np


def get_ramachandran_and_free_energy(samples_np,energies_np,log_w_np,prefix=''):
    atom_types = get_alanine_atom_types()
    atom_types[[4,6,8,14,16]] = np.arange(4, 9)
    # carbon atoms
    carbon_pos = np.where(atom_types==1)[0]
    carbon_samples_np = samples_np.reshape(-1, 22, 3)[:, carbon_pos]
    carbon_distances = np.linalg.norm(samples_np.reshape(-1, 22, 3)[:, [8]] - carbon_samples_np, axis=-1)
    # likely index of c beta atom
    cb_idx = np.where(carbon_distances==carbon_distances.min(1, keepdims=True))


    def determine_chirality_batch(cartesian_coords_batch):
        # Convert Cartesian coordinates to numpy array
        coords_batch = np.array(cartesian_coords_batch)

        # Check if the shape of the array is (n, 4, 3), where n is the number of chirality centers
        if coords_batch.shape[-2:] != (4, 3):
            raise ValueError("Input should be a batch of four 3D Cartesian coordinates")

        # Calculate the vectors from the chirality centers to the four connected atoms
        vectors_batch = coords_batch - coords_batch[:, 0:1, :]
        #print(vectors_batch)
        # Calculate the normal vectors of the planes formed by the three vectors for each chirality center
        normal_vectors_batch = np.cross(vectors_batch[:, 1, :], vectors_batch[:, 2, :])
        #print(normal_vectors_batch)
        # Calculate the dot products of the normal vectors and the vectors from the chirality centers to the fourth atoms
        dot_products_batch = np.einsum('...i,...i->...', normal_vectors_batch, vectors_batch[:, 3, :])
        #print(dot_products_batch)
        # Determine the chirality labels based on the signs of the dot products
        chirality_labels_batch = np.where(dot_products_batch > .000, 'L', 'D')

        return chirality_labels_batch


    back_bone_samples = samples_np.reshape(-1, 22, 3)[:, np.array([8,6,14])]
    cb_samples = samples_np.reshape(-1, 22, 3)[cb_idx[0], carbon_pos[cb_idx[1]]][:, None, :]
    chirality = determine_chirality_batch(np.concatenate([back_bone_samples, cb_samples], axis=1))
    samples_np_mapped = samples_np.copy()
    samples_np_mapped[chirality=="D"] *= -1
    dataset=get_alanine_implicit_dataset()
    traj_samples3 = md.Trajectory(samples_np_mapped[energies_np.flatten()<-75].reshape(-1, 22, 3), topology=dataset.system.mdtraj_topology)

    phi_indices, psi_indices = [4, 6, 8, 14], [6, 8, 14, 16]
    angles = md.compute_dihedrals(traj_samples3, [phi_indices, psi_indices])


    plot_range = [-np.pi, np.pi]



    fig, ax = plt.subplots(figsize=(11, 9))

    h, x_bins, y_bins, im = ax.hist2d(angles[:,0], angles[:,1], 100, norm=LogNorm(), range=[plot_range,plot_range],rasterized=True)
    ticks = np.array([np.exp(-6)*h.max(), np.exp(-4)*h.max(),np.exp(-2)*h.max(), h.max()])
    ax.set_xlabel(r"$\varphi$", fontsize=45)
    ax.set_title("Ramachandran plot", fontsize=45)
    ax.set_ylabel(r"$\psi$", fontsize=45)
    ax.xaxis.set_tick_params(labelsize=25)
    ax.yaxis.set_tick_params(labelsize=25)
    cbar = fig.colorbar(im, ticks=ticks)
    cbar.ax.set_yticklabels(np.abs(-np.log(ticks/h.max())), fontsize=25)
    cbar.ax.invert_yaxis()
    cbar.ax.set_ylabel(r"Free energy / $k_B T$", fontsize=35)
    wandb.log({prefix + "Ramachandran plot": wandb.Image(plt)})

    def plot_fes(
    samples: np.ndarray,
    bw_method:  None,
    weights: None,
    get_DeltaF: bool = True,
    kBT: float = 1.0,
    ):
        from scipy.stats import gaussian_kde
        bw_method =0.18# "scott"
        grid = np.linspace(samples.min(), samples.max(), 100)
        samples=samples[weights!=0]
        weights=weights[weights!=0]
        fes = -kBT * gaussian_kde(samples, bw_method, weights).logpdf(grid)
        fes -= fes.min()
        #plt.plot(grid, fes)

        return grid, fes

    
    traj_samples4 = md.Trajectory(samples_np_mapped.reshape(-1, 22, 3), topology=dataset.system.mdtraj_topology)

    phi = md.compute_phi(traj_samples4)[1].flatten()
    phi_right = phi.copy()
    phi_left = phi.copy()
    phi_right[phi<0] += 2*np.pi
    phi_left[phi>np.pi/2] -= 2*np.pi

    npz=np.load("../data/ad2_umbrella_sampling_means.npz")
    xs=npz["xs"]
    f_i_mean=npz["f_i_mean"]
    f_i_std=npz["f_i_std"]
            
    #plot_fes(phi,bw_method=None, weights=np.exp(log_w_np),get_DeltaF=False)
    grid_left, fes_left = plot_fes(phi_left, bw_method=None, weights=np.exp(log_w_np[:,0]), get_DeltaF=False)
    grid_right, fes_right = plot_fes(phi_right, bw_method=None, weights=np.exp(log_w_np[:,0]), get_DeltaF=False)

    
    middle = 1.1
    idx_left = (grid_left>=-np.pi)&(grid_left<middle)
    grid_left = grid_left[idx_left]
    fes_left = fes_left[idx_left]
    idx_right = (grid_right<=np.pi)&(grid_right>middle)
    grid_right = grid_right[idx_right]
    fes_right = fes_right[idx_right]


    plt.figure(figsize=(16,9))
    plt.plot(xs, f_i_mean, linewidth=5)
    plt.fill_between(xs, f_i_mean - f_i_std, f_i_mean + f_i_std, alpha=0.2)
    plt.plot(np.hstack([grid_left, grid_right]), np.hstack([fes_left, fes_right]), linewidth=5, linestyle="--")
    plt.xlabel(r"$\varphi$", fontsize=45)
    plt.ylabel(r"Free energy / $k_B T$", fontsize=45)
    wandb.log({prefix + 'Free energy profile': wandb.Image(plt)})
    left = 0.
    right = 2

    hist, edges = np.histogram(phi, bins=100, density=True,weights=np.exp(log_w_np[:,0]))
    centers = 0.5*(edges[1:] + edges[:-1])
    centers_pos = (centers > left) & (centers < right)
    
    free_energy_difference = -np.log(hist[centers_pos].sum()/
    hist[~centers_pos].sum())
    wandb.log({"Free energy difference": free_energy_difference})
    print('predicted free energy difference: ', free_energy_difference)


if __name__== "__main__":
    args,p=parse_arguments()
    args=get_args(args,p)

    wandb.init(project=args['wandb_project'], name=args['wandb_inference_name'])
    wandb.config.update(args)

    h_initial = get_alanine_features()
    potential_model, vector_field, interpolant_obj = load_models(args,h_initial)
    potential_model.eval()
    vector_field.eval()
    integral_type='ode'
    if args['divergence']==True:
        integral_type='ode_divergence'
    if args['SDE']:
        integral_type='sde'
        args['divergence']=False

    if args['model_type']=='vector_field':
        #nll_samples=torch.from_numpy(np.load("../data/AD2_relaxed_holdout.npy")).reshape(-1, 66).float()[::100]
        #nll_samples=remove_mean(nll_samples,n_particles=num_particles,n_dimensions=n_dimensions)
        #nll_np=compute_nll(interpolant_obj,nll_samples)
        samples_np,dlogf_np=gen_samples(n_samples=args['n_samples'],n_sample_batches=args['n_sample_batches'],interpolant_obj=interpolant_obj,integral_type=integral_type,n_timesteps=1000)
        if args['divergence']:
            wandb.log({"NLL_mean": -dlogf_np.mean()})
            wandb.log({"NLL_std": -dlogf_np.std()})
        energies_np,energies_data_holdout=get_energies(samples_np)
        log_w_np=np.zeros((len(samples_np),1))
        if args['divergence']:
            log_w_np=get_importance_weights(dlogf_np,energies_np)
            
        samples_np,energies_np,log_w_np=plot_energy_distributions(energies_data_holdout,samples_np,energies_np,log_w_np,weight_threshold=0)
        get_ramachandran_and_free_energy(samples_np,energies_np,log_w_np)

    elif args['model_type']=='potential':
        samples_np,_=gen_samples(n_samples=args['n_samples'],n_sample_batches=args['n_sample_batches'],interpolant_obj=interpolant_obj,integral_type=integral_type,n_timesteps=1000)
        if args['MCMC']:
            time_start = time.time()
            samples_np=interpolant_obj.simulate(samples_np,steps=args['MCMC_steps'],step_size=2e-4,simulation_fn=interpolant_obj.MALA_steps)
            time_end=time.time()
            wandb.log({"MCMC time": time_end - time_start})
            print(time_end - time_start)
        energies_np,energies_data_holdout=get_energies(samples_np)
        dlogf_np=get_potential_logp(interpolant_obj,samples_np)
        log_w_np=get_importance_weights(dlogf_np,energies_np)
        samples_np_05,energies_np_05,log_w_np_05=plot_energy_distributions(energies_data_holdout,samples_np,energies_np,log_w_np,weight_threshold=args['weight_threshold'])
        get_ramachandran_and_free_energy(samples_np_05,energies_np_05,log_w_np_05)
    else:
        raise ValueError("Model type not recognized")


    