import time
import tqdm
import numpy as np
import torch
import dgl
from bgflow.utils import remove_mean
from bgflow.utils import distances_from_vectors
from bgflow.utils import distance_vectors

num_particles = 22
n_dimensions = 3
dim = num_particles * n_dimensions


def gen_samples(n_samples,n_sample_batches,interpolant_obj,integral_type='ode',ess_threshold=0.75,n_timesteps=1000):
    
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
            
    
    latent_np = latent_np.reshape(-1, dim)
    samples_np = samples_np.reshape(-1, dim)
    samples_np = samples_np.reshape(-1, dim)
    time_end = time.time()
    print("Sampling took {} seconds".format(time_end - time_start))
    return samples_np