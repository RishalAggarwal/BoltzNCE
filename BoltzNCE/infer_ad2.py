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
    samples_np = samples_np * interpolant_obj.scaling
    time_end = time.time()
    print("Sampling took {} seconds".format(time_end - time_start))
    return samples_np

'''def get_energies_and_weights(model,samples):
    dlogf_all=[]
    samples_torch=torch.from_numpy(samples).float().cuda()
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
    energies_np = as_numpy(target_xtb.energy(torch.from_numpy(samples)/scaling))
    energy_offset = 34600
    energies_np += energy_offset
    energies_torch=torch.tensor(-energies_np)
    energies_torch=energies_torch - torch.logsumexp(energies_torch,dim=(0,1))
    energies_w=energies_torch.numpy()
    plt.scatter(energies_w[energies_np<0],dlogf_np[energies_np<0])
    plt.xlabel('Boltzmann Weight')
    plt.ylabel('Model Likelihood')
    log_w_np = as_numpy(energies_w).reshape(-1,1) - dlogf_np.reshape(-1,1)
    print("Sampling efficiency: ",sampling_efficiency(torch.from_numpy(log_w_np)).item())
    return dlogf_np,energies_np,log_w_np'''