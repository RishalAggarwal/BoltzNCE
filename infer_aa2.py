#inference script heavily inspired by work of Leon Klein and his transferable Boltzmann Generators codebase

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
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import mdtraj as md
import argparse
import wandb
import einops
import sys

from bgflow import MeanFreeNormalDistribution, OpenMMBridge, OpenMMEnergy
sys.path.append('./BoltzNCE/')
from BoltzNCE.models.interpolant import Interpolant
from BoltzNCE.models.ebm import GVP_EBM
from BoltzNCE.utils.utils import load_models
from BoltzNCE.utils.arguments import get_args
from BoltzNCE.dataset.aa2_dataloader import get_aa2_dataloader
from BoltzNCE.dataset.aa2_dataset import aa2_featurizer
from BoltzNCE.utils.tbg_utils import create_adjacency_list, find_chirality_centers, compute_chirality_sign, check_symmetry_change
import networkx.algorithms.isomorphism as iso
import networkx as nx
import scipy
import signal
import deeptime as dt
import openmm
from train_aa2 import train_potential
from BoltzNCE.dataset.aa2_single_dataloader import get_aa2_single_dataloader


def parse_arguments():
    p =argparse.ArgumentParser()
    p.add_argument('--config', type=str, default=None)
    p.add_argument('--peptide', type=str, default='AA')
    p.add_argument("--divergence", action="store_true",default=True)
    p.add_argument("--no-divergence", action="store_false", dest="divergence")
    p.add_argument('--compute_metrics', dest='compute_metrics', action='store_true',default = True)
    p.add_argument('--no-compute_metrics', dest='compute_metrics', action='store_false')
    p.add_argument('--n_epochs', type=int, default=100)

    '''p.add_argument("--NLL", action="store_true",default=True)
    p.add_argument("--no-NLL", action="store_false", dest="NLL")'''
    '''p.add_argument("--SDE", action="store_true", default=False)
    p.add_argument('--weight_threshold', type=float, default=0.2)'''
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
    p.add_argument('--data_path',type=str,default="data/2AA-1-huge",required=False)
    p.add_argument('--data_directory',type=str,default="/test",required=False)
    args=p.parse_args()
    
    return args,p


def gen_samples(n_samples,n_sample_batches,interpolant_obj: Interpolant,integral_type='ode'):
    time_start = time.time()
    
    latent_np = np.empty(shape=(0))
    samples_np = np.empty(shape=(0))
    log_w_np = np.empty(shape=(0))
    dlogp_all=[]
    #dlogf_all=[]
    #energies_np = np.empty(shape=(0))
    distances_x_np = np.empty(shape=(0))
    interpolant_placeholder=interpolant_obj.interpolant_type
    interpolant_obj.interpolant_type= interpolant_obj.integration_interpolant

    for i in tqdm.tqdm(range(n_sample_batches)):
        if integral_type=='ode':
            samples=interpolant_obj.ode_integral(n_samples)
        elif integral_type=='ode_divergence':
            samples,logp_samples=interpolant_obj.ode_divergence_integral(n_samples)
            dlogp_all.append(logp_samples.cpu().detach().numpy())
        else:
            raise ValueError("integral_type not recognized")
                
        #dlogf=interpolant_obj.log_prob_forward(samples)
        #latent = latent[0]
        #log_weights = bg.log_weights_given_latent(samples, latent, dlogp).detach().cpu().numpy()
        samples_np = np.append(samples_np, samples.detach().cpu().numpy())
        n_dimensions = 3
        num_particles = interpolant_obj.dim // n_dimensions
        distances_x = distances_from_vectors(distance_vectors(samples.view(-1, num_particles, n_dimensions))).detach().cpu().numpy().reshape(-1)
        distances_x_np = np.append(distances_x_np, distances_x)

        #log_w_np = np.append(log_w_np, log_weights)
        #dlogf_all.append(dlogf.cpu().detach())
        #energies = target.energy(samples/scaling).detach().cpu().numpy()
        #energies_np = np.append(energies_np, energies)
    if len(dlogp_all)>0:
        dlogp_all = np.concatenate(dlogp_all, axis=0)
    else:
        #uniform weights when no divergence
        dlogp_all=np.zeros((samples_np.shape[0], 1), dtype=np.float32)
    samples_np = samples_np.reshape(-1, interpolant_obj.dim)
    samples_np = samples_np * interpolant_obj.scaling
    time_end = time.time()
    print("Sampling took {} seconds".format(time_end - time_start))
    wandb.log({"Sampling time": time_end - time_start})
    interpolant_obj.interpolant_type=interpolant_placeholder
    return samples_np,dlogp_all

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
    dlogf_np=dlogf_all.cpu().detach().numpy()
    return dlogf_np

def compute_metrics(npz, dlogf_np, scaling, topology, model_samples, n_atoms, n_dimensions, aligned_idxs, symmetry_change, pdb_path, traj_samples,prefix=''):  
    plot_ramachandran(traj_samples, plot_name=prefix+'TBG')
    data=remove_mean(npz['positions'][npz['step']%10000==0].reshape(-1, n_atoms*n_dimensions)[:190000], n_atoms, n_dimensions)*scaling
    traj_samples_data = md.Trajectory(data.reshape(-1, dim//3, 3)/scaling, topology=topology)
    plot_ramachandran(traj_samples_data, plot_name=prefix+'MD')
    pdb = openmm.app.PDBFile(pdb_path)
    forcefield = openmm.app.ForceField("amber14-all.xml", "implicit/obc1.xml")

    system = forcefield.createSystem(pdb.topology, nonbondedMethod=openmm.app.CutoffNonPeriodic,
            nonbondedCutoff=2.0*openmm.unit.nanometer, constraints=None)
    integrator = openmm.LangevinMiddleIntegrator(310*openmm.unit.kelvin, 0.3/openmm.unit.picosecond, 0.5*openmm.unit.femtosecond)
    openmm_energy = OpenMMEnergy(bridge=OpenMMBridge(system, integrator, platform_name="CUDA"))
    classical_model_energies = as_numpy(openmm_energy.energy(model_samples.reshape(-1, dim)[~symmetry_change]))
    classical_target_energies = as_numpy(openmm_energy.energy(torch.from_numpy(data)[::10].reshape(-1, dim)/scaling))
    idxs = np.array(aligned_idxs)[~symmetry_change]
    log_w_np = -classical_model_energies - as_numpy(dlogf_np.reshape(-1,1)[idxs])
    log_w_torch=torch.tensor(log_w_np)
    log_w_torch=log_w_torch - torch.logsumexp(log_w_torch,dim=(0,1))
    log_w_np=log_w_torch.numpy()
    wandb.log({prefix + "Sampling efficiency Mean": sampling_efficiency(torch.from_numpy(log_w_np)).item()})
    plot_energy_histograms(classical_target_energies, classical_model_energies, log_w_np, prefix=prefix)
    grid_left_data, fes_left_data, grid_right_data, fes_right_data = phi_to_grid(md.compute_phi(traj_samples_data)[1].flatten())
    grid_left, fes_left, grid_right, fes_right = phi_to_grid(md.compute_phi(traj_samples)[1].flatten())
    grid_left_weighted, fes_left_weighted, grid_right_weighted, fes_right_weighted = phi_to_grid(md.compute_phi(traj_samples)[1].flatten(), weights=np.exp(log_w_np))
    plt.figure(figsize=(16,9))
    plt.plot(np.hstack([grid_left_data, grid_right_data]), np.hstack([fes_left_data, fes_right_data]), linewidth=5, label="MD")
    plt.plot(np.hstack([grid_left, grid_right]), np.hstack([fes_left, fes_right]), linewidth=5,linestyle="--", label="TBG")
    plt.plot(np.hstack([grid_left_weighted, grid_right_weighted]), np.hstack([fes_left_weighted, fes_right_weighted]), linewidth=5, linestyle="--", label="TBG reweighted")
    plt.legend(fontsize=30)
    plt.title(r"Free energy projection $\varphi$", fontsize=45)
    plt.xlabel(r"$\varphi$", fontsize=45)
    plt.ylabel("Free energy / $k_B T$", fontsize=45)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    wandb.log({prefix + "Free energy projection": wandb.Image(plt)})
    plt.close()
    tica_model = run_tica(traj_samples_data, lagtime=100)
    features = tica_features(traj_samples_data)
    tics = tica_model.transform(features)
    feat_model = tica_features(traj_samples)
    tics_model = tica_model.transform(feat_model)
    fig, ax = plt.subplots(figsize=(10,10))
    ax = plot_tic01(ax, tics, f"MD", tics_lims=tics)
    wandb.log({prefix+ "TICA MD": wandb.Image(fig)})
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10,10))
    ax = plot_tic01(ax, tics_model, f"TBG", tics_lims=tics)
    wandb.log({prefix+ "TICA TBG": wandb.Image(fig)})
    plt.close(fig)
    #free energy projection along first TICA component
    grid_data, fes_data = plot_fes(tics[:,0], bw_method=None, weights=None, get_DeltaF=False)
    grid, fes = plot_fes(tics_model[:,0], bw_method=None, weights=None, get_DeltaF=False)
    grid_weighted,fes_weighted = plot_fes(tics_model[:,0], bw_method=None, weights=np.exp(log_w_np), get_DeltaF=False)
    plt.figure(figsize=(16,9))
    plt.plot(grid_data, fes_data, linewidth=5, label="MD")
    plt.plot(grid, fes, linewidth=5, linestyle="--", label="TBG")
    plt.plot(grid_weighted, fes_weighted, linewidth=5, linestyle="--", label="TBG reweighted")
    plt.title(r"Free energy projection TICA", fontsize=45)
    plt.xlabel("TIC0", fontsize=45)
    plt.ylabel("Free energy / $k_B T$", fontsize=45)
    plt.legend(fontsize=30)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    wandb.log({prefix + "Free energy projection TICA0": wandb.Image(plt)})
    plt.close()


def align_topology(sample, reference, scaling, atom_types):
    sample = sample.reshape(-1, 3)
    all_dists = scipy.spatial.distance.cdist(sample, sample)
    adj_list_computed = create_adjacency_list(all_dists/scaling, atom_types)
    G_reference = nx.Graph(reference)
    G_sample = nx.Graph(adj_list_computed)
    # not same number of nodes
    if len(G_sample.nodes) != len(G_reference.nodes):
        return sample, False
    for i, atom_type in enumerate(atom_types):
        G_reference.nodes[i]['type']=atom_type
        G_sample.nodes[i]['type']=atom_type
        
    nm = iso.categorical_node_match("type", -1)
    GM = iso.GraphMatcher(G_reference, G_sample, node_match=nm)
    is_isomorphic = GM.is_isomorphic()
    # True
    GM.mapping
    initial_idx = list(GM.mapping.keys())
    final_idx = list(GM.mapping.values())
    sample[initial_idx] = sample[final_idx]
    return sample, is_isomorphic

def align_samples(samples_np, adj_list, dim,atom_types, scaling):
    def handler(signum, frame):
        raise TimeoutError("Function call took too long")

    aligned_samples = []
    aligned_idxs = []
    #for i, sample in enumerate(samples_np[(energies_np.flatten() < -52800)].reshape(-1,dim//3, 3)):
    for i, sample in tqdm.tqdm(enumerate(samples_np.reshape(-1,dim//3, 3))):   
            # Set a timer for 5 seconds
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(5)  # Timeout set to 5 seconds

        try:
            # Call your function here
            aligned_sample, is_isomorphic = align_topology(sample, as_numpy(adj_list).tolist(), scaling, atom_types)
            if is_isomorphic:
                aligned_samples.append(aligned_sample)
                aligned_idxs.append(i)
        except TimeoutError: 
            print("Skipping iteration, function call took too long")
            continue  # Skip to the next iteration
        finally:
            signal.alarm(0)

    aligned_samples = np.array(aligned_samples)
    print(f"Correct configuration rate {len(aligned_samples)/len(samples_np)}")
    wandb.log({"Correct configuration rate": len(aligned_samples)/len(samples_np)})
    return aligned_samples, aligned_idxs

def plot_ramachandran(traj_samples, plot_name=''):
    fig, ax = plt.subplots(figsize=(9, 9))
    plot_range = [-np.pi, np.pi]
    phis = md.compute_phi(traj_samples)[1].flatten()
    psis = md.compute_psi(traj_samples)[1].flatten()
    h, x_bins, y_bins, im = ax.hist2d(phis, psis, 100, norm=LogNorm(), range=[plot_range,plot_range],rasterized=True)
    ticks = np.array([np.exp(-6)*h.max(), np.exp(-4.0)*h.max(),np.exp(-2)*h.max(), h.max()])
    ax.set_xlabel(r"$\varphi$", fontsize=45)
    ax.set_title(plot_name, fontsize=45)
    ax.xaxis.set_tick_params(labelsize=25)
    ax.yaxis.set_tick_params(labelsize=25)
    ax.set_yticks([])
    wandb.log({plot_name + " Ramachandran plot": wandb.Image(plt)})
    plt.close(fig)

def fix_chirality(samples, adj_list, atom_types, data, dim):
    chirality_centers = find_chirality_centers(adj_list, atom_types)
    if len(chirality_centers) == 0:
        print("No chirality centers found, skipping chirality check")
        symmetry_change = np.zeros(len(samples), dtype=bool)
        return samples, symmetry_change
    reference_signs = compute_chirality_sign(torch.from_numpy(data.reshape(-1, dim//3, 3))[[1]], chirality_centers)
    symmetry_change = check_symmetry_change(samples, chirality_centers, reference_signs)
    samples[symmetry_change] *=-1
    symmetry_change = check_symmetry_change(samples, chirality_centers, reference_signs)
    print(f"Correct symmetry rate {(~symmetry_change).sum()/len(samples)}")
    wandb.log({"Correct symmetry rate": (~symmetry_change).sum()/len(samples)})
    return samples, symmetry_change

def plot_energy_histograms(classical_target_energies, classical_model_energies, log_w_np,prefix=''):
    plt.figure(figsize=(16,9))
    range_limits = (classical_target_energies.min()-10,classical_target_energies.max()+100)
    plt.hist(classical_target_energies, bins=100, alpha=0.5, range=range_limits,density=True, label="MD")
    plt.hist(classical_model_energies, bins=100,alpha=0.5, range=range_limits,density=True, label="TBG + full")
    plt.hist(classical_model_energies, bins=100, range=range_limits,density=True, label="TBG + full weighted", weights=np.exp(log_w_np), histtype='step', linewidth=5)
    plt.legend(fontsize=30)
    plt.xlabel("Energy  / $k_B T$", fontsize=45)  
    plt.yticks(fontsize=25)
    plt.xticks(fontsize=25)

    plt.title(f"Classical energy distribution", fontsize=45)
    wandb.log({prefix+"Classical energy distribution": wandb.Image(plt)})
    plt.close()

def plot_fes(
    samples: np.ndarray,
    bw_method:  None,
    weights: None,
    get_DeltaF: bool = True,
    kBT: float = 1.0,
):
    from scipy.stats import gaussian_kde
    bw_method = 0.18
    grid = np.linspace(samples.min(), samples.max(), 100)
    if weights is not None:
        weights = weights[:,0]
    fes = -kBT * gaussian_kde(samples, bw_method, weights).logpdf(grid)
    fes -= fes.min()

    return grid, fes

def phi_to_grid(phis,weights=None):
    phi_right = phis.copy()
    phi_left = phis.copy()
    phi_right[phis<0] += 2*np.pi
    phi_left[phis>np.pi/2] -= 2*np.pi       

    #plot_fes(phi,bw_method=None, weights=np.exp(log_w_np),get_DeltaF=False)
    grid_left, fes_left = plot_fes(phi_left,bw_method=None, weights=weights, get_DeltaF=False)
    grid_right, fes_right = plot_fes(phi_right,bw_method=None, weights=weights, get_DeltaF=False)
    middle = 0#1.1
    idx_left = (grid_left>=-np.pi)&(grid_left<middle)
    grid_left_data  = grid_left[idx_left]
    fes_left_data  = fes_left[idx_left]
    idx_right = (grid_right<=np.pi)&(grid_right>middle)
    grid_right_data  = grid_right[idx_right]
    fes_right_data  = fes_right[idx_right]    
    return grid_left_data, fes_left_data, grid_right_data, fes_right_data

def distances(xyz):
    distance_matrix_ca = np.linalg.norm(
        xyz[:, None, :, :] - xyz[:, :, None, :],
        axis=-1
    )
    n_ca = distance_matrix_ca.shape[-1]
    m, n = np.triu_indices(n_ca, k=1)
    distances_ca = distance_matrix_ca[:, m, n]
    return distances_ca

def wrap(array):
    return (np.sin(array), np.cos(array))

def tica_features(
    trajectory,
    use_dihedrals=True,
    use_distances=True,
    selection="symbol == C or symbol == N or symbol == S"
):
    trajectory = trajectory.atom_slice(trajectory.top.select(selection))
    n_atoms = trajectory.xyz.shape[1]
    if use_dihedrals:
        _, phi = md.compute_phi(
            trajectory
        )
        _, psi = md.compute_psi(
            trajectory
        )
        _, omega= md.compute_omega(
            trajectory
        )
        dihedrals = np.concatenate([*wrap(phi), *wrap(psi), *wrap(omega)], axis=-1)
    if use_distances:
        ca_distances = distances(trajectory.xyz)
    if use_distances and use_dihedrals:
        return np.concatenate([ca_distances, dihedrals], axis=-1)
    elif use_distances:
        return ca_distances
    elif use_dihedrals:
        return ca_dihedrals
    else:
        return []
    
from matplotlib.colors import LogNorm

def plot_tic01(ax, tics, name, tics_lims, cmap='viridis'):
    _ = ax.hist2d(tics[:,0], tics[:,1], bins=100, norm=LogNorm(), cmap=cmap,rasterized=True)
    ax.set_xlabel("TIC0", fontsize=45)
    ax.set_ylabel("TIC1", fontsize=45)
    ax.set_ylim(tics_lims[:,1].min(),tics_lims[:,1].max())
    ax.set_xlim(tics_lims[:,0].min(),tics_lims[:,0].max())
    ax.set_xticks([])
    ax.set_yticks([])
    plt.title(f"{name}", fontsize=45)
    return ax
def run_tica(trajectory, lagtime=500, dim=40):
    ca_features = tica_features(trajectory)
    tica = dt.decomposition.TICA(dim=dim, lagtime=lagtime)
    koopman_estimator = dt.covariance.KoopmanWeightingEstimator(lagtime=lagtime)
    reweighting_model = koopman_estimator.fit(ca_features).fetch_model()
    tica_model = tica.fit(ca_features, reweighting_model).fetch_model()
    return tica_model

def update_interpolant_args(args):
    args['interpolant']['rtol']=args['rtol']
    args['interpolant']['atol']=args['atol']
    args['interpolant']['tmin']=args['tmin']
    args['interpolant']['dim'] = args['dim']
    args['interpolant']['num_particles'] = args['dim'] // 3
    return args

def process_gen_samples(samples_np, dlogf_np, scaling, topology, adj_list, atom_types, peptide, args,prefix=''):
    traj_samples = md.Trajectory(samples_np.reshape(-1, dim//3, 3)/scaling, topology=topology)
    aligned_samples, aligned_idxs = align_samples(samples_np, adj_list, dim, atom_types, scaling)
    traj_samples_aligned = md.Trajectory(aligned_samples/scaling, topology=topology)
    model_samples = torch.from_numpy(traj_samples_aligned.xyz)
    if args['data_directory']=='/test':
        npz=np.load(args['data_path'] + args['data_directory'] + f"/{peptide}-traj-arrays.npz")
        n_atoms = npz['positions'].shape[1]
        n_dimensions = 3
        data=remove_mean(npz['positions'][npz['step']%10000==0].reshape(-1, n_atoms*n_dimensions)[:190000], n_atoms, n_dimensions)*scaling
    else:
        data = np.load(args['data_path'] + "all_train.npy", allow_pickle=True).item()[peptide]
        n_atoms = int(data.shape[1]/3) # expand back to b n_atoms, dims
        n_dimensions = 3
        data = remove_mean(data.reshape(-1, n_atoms*n_dimensions), n_atoms, n_dimensions)*scaling
    model_samples,symmetry_change=fix_chirality(model_samples, adj_list, atom_types, data, dim)
    traj_samples=md.Trajectory(as_numpy(model_samples)[~symmetry_change], topology=topology)
    
    if args['compute_metrics'] == True:
        compute_metrics(npz, dlogf_np, scaling, topology, model_samples, n_atoms, n_dimensions, aligned_idxs, symmetry_change, pdb_path, traj_samples,prefix)
    return model_samples,npz,aligned_idxs, symmetry_change, traj_samples


if __name__== "__main__":
    args,p=parse_arguments()
    args=get_args(args,p)
    wandb.init(project=args['wandb_project'], name=args['wandb_inference_name'])
    wandb.config.update(args)
    scaling=30.0
    
    test_peptides,test_atom_types_dict, test_h_dict = aa2_featurizer(args['data_path'],args['data_directory'])
    peptide= args['peptide']
    pdb_path=args['data_path'] + args['data_directory'] + f"/{peptide}-traj-state0.pdb"
    topology=md.load_topology(pdb_path)

    atom_dict = {"C": 0, "H":1, "N":2, "O":3, "S":4}
    #["C", "H", "N", "O", "S"]
    atom_types = []
    for atom_name in topology.atoms:
        atom_types.append(atom_name.name[0])
    atom_types = torch.from_numpy(np.array([atom_dict[atom_type] for atom_type in atom_types]))
    backbone_idxs = topology.select("backbone")
    adj_list = torch.from_numpy(np.array([(b.atom1.index, b.atom2.index) for b in topology.bonds], dtype=np.int32))
    
    h_initial=test_h_dict[args['peptide']].cuda()
    dim = len(h_initial) * 3
    args['dim'] = dim
    update_interpolant_args(args)
    
    potential_model, vector_field, interpolant_obj = load_models(args,h_initial=h_initial,potential=args['model_type']=='potential')
    if potential_model is not None:
        potential_model.eval()
    vector_field.eval()
    integral_type = 'ode'
    if args['divergence']==True:
        integral_type='ode_divergence'
    print("########## generating initial samples")
    samples_np,dlogf_np=gen_samples(n_samples=args['n_samples'],n_sample_batches=args['n_sample_batches'],interpolant_obj=interpolant_obj,integral_type=integral_type)
    print(f"Generated {len(samples_np)} samples")
    if args['model_type']=='vector_field':
        model_samples,npz,aligned_idxs, symmetry_change, traj_samples =process_gen_samples(samples_np, dlogf_np, scaling, topology, adj_list, atom_types, peptide, args)
    elif args['model_type']=='potential':
        print("########## training and computing potential logp")
        optim_potential = torch.optim.Adam(potential_model.parameters(), lr=1e-4)
        scheduler_potential = torch.optim.lr_scheduler.ReduceLROnPlateau(optim_potential, mode='min', factor=0.5, patience=20, verbose=True,min_lr=1e-5)
        dlogf_np = get_potential_logp(interpolant_obj, samples_np)
        model_samples,npz,aligned_idxs, symmetry_change, traj_samples = process_gen_samples(samples_np, dlogf_np, scaling, topology, adj_list, atom_types, peptide, args,prefix=f"potential_Epoch_0_")
        samples_np =(model_samples.reshape(-1, dim)[~symmetry_change]*30).cpu().detach().numpy()
        n_atoms = samples_np.shape[1]//3 # expand back to b n_atoms, dims
        n_dimensions = 3
        samples_np =samples_np.reshape(-1, n_atoms*n_dimensions)
        aligned_idxs = np.arange(len(samples_np))
        symmetry_change = np.zeros(len(samples_np), dtype=bool)
        _, dataloader = get_aa2_single_dataloader(samples_np,h_initial,256,True,0,kabsch=True)
        for i in range(args['n_epochs']):
            print(f"########## Epoch {i+1}")
            potential_model.train()
            potential_model=train_potential(args, dataloader, interpolant_obj, potential_model, optim_potential, scheduler_potential, num_epochs=1, window_size=0.025, num_negatives=1, nce_weight=1.0,grad_norm=None)
            potential_model.eval()
            interpolant_obj.potential_function = potential_model
            if i==5:
                optim_potential.param_groups[0]['lr'] = 1e-4
            if i % 10 == 0:
                dlogf_np = get_potential_logp(interpolant_obj, samples_np)
                compute_metrics(npz, dlogf_np, scaling, topology, model_samples, n_atoms, n_dimensions, aligned_idxs, symmetry_change, pdb_path, traj_samples,prefix=f"potential_Epoch_{i+1}_")
    else:
        raise ValueError("model_type not recognized, should be either vector_field or potential")   
    if args['save_generated']:
        model_samples = einops.rearrange(model_samples,"b n d -> b (n d)")
        # print(model_samples.shape) #(1000, 114)
        # print(dlogf_np.shape) #(114000, 1)
        numpy_dict={
            'samples': samples_np,
            'dlogf': dlogf_np,
        }
        np.savez(args['save_prefix'] + args['peptide'] + '_numpy_dict.npz', **numpy_dict)   
        train_dict={args['peptide']: model_samples,}
        np.save(args['save_prefix'] + args['peptide'], train_dict)    
