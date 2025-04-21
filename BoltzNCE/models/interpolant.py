import torch
import torchdiffeq
import dgl
import numpy as np
import tqdm
import ot
from scipy.optimize import linear_sum_assignment


class Interpolant(torch.nn.Module):
    def __init__(self, h_initial=None,potential_function=None,num_particles=22,n_dimensions=3,dim=66,interpolant_type='linear',vector_field=None,scaling=1.0,ot=False):
        super().__init__()
        self.potential_function = potential_function
        self.vector_field=vector_field
        self.prior = torch.distributions.MultivariateNormal(torch.zeros(dim), torch.eye(dim))
        self.n_particles = num_particles
        self.n_dimensions = n_dimensions
        self.h_initial = h_initial.to('cuda')
        self.dim = dim
        self.nodes=torch.arange(self.n_particles)
        self.edges=torch.cartesian_prod(self.nodes,self.nodes)
        self.edges=self.edges[self.edges[:,0]!=self.edges[:,1]].transpose(0,1)
        self.interpolant_type=interpolant_type
        self.graph=None
        self.scaling=scaling
        self.ot=ot

    def alpha(self,t):
        if self.interpolant_type=='linear':
            return 1-t
        elif self.interpolant_type=='trig':
            return torch.cos(t*torch.pi/2)
        else:
            raise ValueError('interpolant type not supported')
        
    def sigma(self,t):
            if self.interpolant_type=='linear':
                return t
            elif self.interpolant_type=='trig':
                return torch.sin(t*torch.pi/2)
            else:
                raise ValueError('interpolant type not supported')
            
    def sigma_dot(self,t):
        if self.interpolant_type=='linear':
            return torch.ones_like(t)
        elif self.interpolant_type=='trig':
            return (torch.pi/2)*torch.cos(t*torch.pi/2)
        else:
            raise ValueError('interpolant type not supported')
    
    def alpha_dot(self,t):
        if self.interpolant_type=='linear':
            return -torch.ones_like(t)
        elif self.interpolant_type=='trig':
            return -(torch.pi/2)*torch.sin(t*torch.pi/2)
        else:
            raise ValueError('interpolant type not supported')
    
    def get_xt_and_vt(self,t,g):
        alpha_t=self.alpha(t).view(-1,1)
        sigma_t=self.sigma(t).view(-1,1)
        alpha_t_dot=self.alpha_dot(t).view(-1,1)
        sigma_t_dot=self.sigma_dot(t).view(-1,1)
        sigma_t_extended=sigma_t.repeat_interleave(self.n_particles)
        sigma_t_extended=sigma_t_extended.view(-1,1)
        alpha_t_extended=alpha_t.repeat_interleave(self.n_particles)
        alpha_t_extended=alpha_t_extended.view(-1,1)
        alpha_t_dot_extended=alpha_t_dot.repeat_interleave(self.n_particles)
        alpha_t_dot_extended=alpha_t_dot_extended.view(-1,1)
        sigma_t_dot_extended=sigma_t_dot.repeat_interleave(self.n_particles)
        sigma_t_dot_extended=sigma_t_dot_extended.view(-1,1)
        coords_prior = torch.randn_like(g.ndata['x'])
        coords_shape=g.ndata['x'].shape
        if self.ot:
            coords_sample=g.ndata['x']
            coords_prior=coords_prior.view(-1,self.dim)
            coords_sample=coords_sample.view(-1,self.dim)
            row_ind, col_ind = self.OT_coupling(coords_sample,coords_prior)
            coords_prior=coords_prior[col_ind]
            coords_prior=coords_prior.view(coords_shape)
            coords_sample=coords_sample.view(coords_shape)
        g.ndata['x1']=coords_prior
        g.ndata['xt'] = alpha_t_extended*g.ndata['x'] + sigma_t_extended*g.ndata['x1']
        g.ndata['v'] = alpha_t_dot_extended*g.ndata['x'] + sigma_t_dot_extended*g.ndata['x1']
        g.ndata['x']=g.ndata['xt']
        return g

    def OT_coupling(self,x,y):
        C=torch.cdist(x,y)
        C=C**2
        C=C/C.max()
        C=C.cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(C)
        return row_ind,col_ind
    
    def velocity_from_score(self,t,x,score):
        alpha_t=self.alpha(t)
        sigma_t=self.sigma(t)
        sigma_t_dot=self.sigma_dot(t)
        alpha_t_dot=self.alpha_dot(t)
        velocity= (1/alpha_t)*((alpha_t_dot*sigma_t - sigma_t_dot*alpha_t)*sigma_t*score + alpha_t_dot*x)
        return velocity

    def sde_w(self,t):
        return torch.sin(torch.pi*t)**2
        
    
    def log_prob_forward(self,samples,time=0.0,torch_grad=False):
        with torch.set_grad_enabled(torch_grad):
            self.setup_graph(samples.shape[0])
            t=torch.ones((samples.shape[0],1)).to('cuda')
            t=t*time
            self.graph.ndata['x']=samples.view(-1,self.n_dimensions)
            ll=self.potential_function(t,self.graph,return_logprob=True)
            return ll

    def score(self,t,x):
        coordinates=x.clone().detach()
        coordinates=coordinates.view(-1,self.n_dimensions)
        self.graph.ndata['x']=coordinates
        t_clone=t.clone().detach()
        t_clone=t_clone*torch.ones((self.graph.batch_size,1)).to('cuda')
        position_score,ll=self.potential_function(t_clone,self.graph)
        position_score=position_score.view(-1,self.dim)
        return position_score
    
    def ode_forward(self,t,x):
        if self.vector_field is not None:
            coordinates=x.clone().detach()
            coordinates=coordinates.view(-1,self.n_dimensions)
            self.graph.ndata['x']=coordinates
            t_clone=t.clone().detach()
            t_clone=t_clone*torch.ones((self.graph.batch_size,1)).to('cuda')
            velocity=self.vector_field(t_clone,self.graph)
            velocity=velocity.view(-1,self.dim)
        else:
            position_score=self.score(t,x)
            velocity=self.velocity_from_score(t,x,position_score)
        return velocity
    
    def ode_divergence_forward(self,t,x_log_prob_tuple):
        x=x_log_prob_tuple[0].clone().detach()
        log_prob=x_log_prob_tuple[1]
        with torch.set_grad_enabled(True):
            x.requires_grad=True
            self.graph.ndata['x']=x.view(-1,self.n_dimensions)
            t_clone=t.clone()
            t_clone=t_clone*torch.ones((self.graph.batch_size,1)).to('cuda')
            velocity=self.vector_field(t_clone,self.graph)
            velocity=velocity.view(-1,self.dim)
            #compute dlogp
            dlogp=torch.zeros((x.shape[0])).to(x.device)
            for i in range(self.dim):
                dlogp+= torch.autograd.grad(velocity[:,i], x,grad_outputs=torch.ones_like(velocity[:,i]),retain_graph=True)[0][:,i]
        return (velocity, -dlogp)


    def sde_forward(self,t,x):
        position_score=self.score(t,x)
        if self.vector_field is not None:
            velocity=self.ode_forward(t,x)
            velocity=velocity.view(-1,self.dim)
        else:
            velocity=self.velocity_from_score(t,x,position_score)
        velocity=self.velocity_from_score(t,x,position_score)
        sde_weight=self.sde_w(t).view(-1,1).to(x.device)
        drift = velocity - 0.5 * sde_weight * position_score
        noise = torch.sqrt(sde_weight)*torch.randn_like(velocity)
        return drift, noise
    

    @torch.no_grad()
    def sde_integral(self,n_samples,n_timesteps=200):
        self.setup_graph(n_samples)
        x_init=self.prior.sample((n_samples,)).to('cuda')
        self.graph.ndata['x']=x_init.view(-1,self.n_dimensions)
        timespan = torch.linspace(0.999,0.04,n_timesteps)
        samples=x_init
        dt=(timespan[1]-timespan[0]).view(-1,1).to(samples.device)
        for t in timespan:
            drift,noise=self.sde_forward(t,samples)
            samples=samples + drift * dt + noise * torch.sqrt(torch.abs(dt))
        return samples
    
    
    @torch.no_grad()
    def ode_integral(self,n_samples):
        self.setup_graph(n_samples)
        x_init=self.prior.sample((n_samples,)).to('cuda')
        self.graph.ndata['x']=x_init.view(-1,self.n_dimensions)
        tmax=0.999
        if self.vector_field is not None:
            tmax=1.0
        t = torch.linspace(tmax, 0., 100).to('cuda')
        x=torchdiffeq.odeint_adjoint(self.ode_forward, x_init, t, method='dopri5',atol=1e-5,rtol=1e-5,adjoint_params=())
        return x[-1]
    
    @torch.no_grad()
    def ode_divergence_integral(self,n_samples):
        self.setup_graph(n_samples)
        x_init=self.prior.sample((n_samples,))
        log_prob_init=self.prior.log_prob(x_init).view(-1,1)
        x_init=x_init.to('cuda')
        log_prob_init=log_prob_init.to('cuda')
        self.graph.ndata['x']=x_init.view(-1,self.n_dimensions)
        tmax=0.999
        if self.vector_field is not None:
            tmax=1.0
        t = torch.linspace(tmax, 0., 100).to('cuda')
        x,log_prob=torchdiffeq.odeint_adjoint(self.ode_divergence_forward, (x_init,log_prob_init), t, method='dopri5',atol=1e-5,rtol=1e-5,adjoint_params=())
        return x[-1],log_prob[-1]
    
    def setup_graph(self,batch_size):
        if (self.graph is not None) and (self.graph.batch_size==batch_size):
            return
        graphs=[]
        for i in range(batch_size):
            g=dgl.graph((self.edges[0].cpu(),self.edges[1].cpu()),num_nodes=self.n_particles).to('cuda')
            g.ndata['x'] = torch.empty((self.n_particles,self.n_dimensions)).cuda()
            g.ndata['h'] = self.h_initial
            graphs.append(g)
        self.graph=dgl.batch(graphs)

    def minimization_steps(self,samples,steps,step_size,time=0):
        for j in tqdm.tqdm(range(steps)):
            dlogf=self.log_prob_forward(samples,time,torch_grad=True)
            grad=torch.autograd.grad(dlogf.sum(),samples)[0]
            samples=samples+step_size*grad
            samples.grad=None
        return samples
    
    def langevin_steps(self,samples,steps,step_size,time=0):
        for j in range(steps):
            dlogf=self.log_prob_forward(samples,time,torch_grad=True)
            grad=torch.autograd.grad(dlogf,samples,grad_outputs=torch.ones_like(dlogf))[0]
            samples=samples+step_size*grad + np.sqrt(2*step_size)*torch.randn_like(samples)
            samples.grad=None
        return samples

    def MALA_steps(self,samples_simulate,steps,step_size,time=0.0):
        for j in range(steps):
            dlogf_current=self.log_prob_forward(samples_simulate,time,torch_grad=True)
            grad_current=torch.autograd.grad(dlogf_current,samples_simulate,grad_outputs=torch.ones_like(dlogf_current))[0]
            mean_proposal=samples_simulate+step_size*grad_current
            noise_proposal=np.sqrt(2*step_size)*torch.randn_like(samples_simulate)
            samples_proposal=mean_proposal + noise_proposal
            dlogf_proposal = self.log_prob_forward(samples_proposal,time,torch_grad=True)
            grad_proposal = torch.autograd.grad(dlogf_proposal,samples_proposal,grad_outputs=torch.ones_like(dlogf_proposal))[0]
            mean_reverse = samples_proposal + step_size*grad_proposal
            reverse_prob=(-1/(4*step_size))*((samples_simulate-mean_reverse)**2).sum(-1).view(-1,1)
            forward_prob = (-1/(4*step_size))*((noise_proposal)**2).sum(-1).view(-1,1)
            log_alpha= dlogf_proposal + reverse_prob - dlogf_current - forward_prob
            alpha=torch.clamp(torch.exp(log_alpha),max=1.0)[:,0]
            u_rand=torch.rand_like(alpha)
            with torch.set_grad_enabled(False):
                samples_simulate[u_rand<=alpha]=samples_proposal[u_rand<=alpha]
            samples_simulate.grad=None
            samples_proposal.grad=None
        return samples_simulate

    def simulate(self,samples,steps,simulation_fn,step_size=2e-4,time=0.0):
        samples_torch=samples
        samples_torch=samples_torch/self.scaling
        step_size=step_size/self.scaling
        if type(samples) != torch.Tensor:
            samples_torch=torch.from_numpy(samples).cuda().float()
        batchsize=500
        i=0
        while (i+batchsize)<len(samples_torch):
            samples_simulate=samples_torch[i:i+batchsize].clone().cuda()
            samples_simulate.requires_grad=True
            samples_simulate=simulation_fn(samples_simulate,steps,step_size,time)
            samples_torch[i:i+batchsize]=samples_simulate.detach().cuda()
            i+=batchsize
        self.setup_graph(len(samples_torch)-i)
        samples_simulate=samples_torch[i:].clone().cuda()
        samples_simulate.requires_grad=True
        samples_simulate=simulation_fn(samples_simulate,steps,step_size,time)
        samples_torch[i:]=samples_simulate.detach().cuda()
        samples_torch=samples_torch*self.scaling
        if type(samples) != torch.Tensor:
            samples_np=samples_torch.cpu().detach().numpy()
            return samples_np
        return samples_torch

    @torch.no_grad()
    def PC_integral(self,n_samples,n_timesteps=100):
        self.setup_graph(n_samples)
        x_init=self.prior.sample((n_samples,)).to('cuda')
        self.graph.ndata['x']=x_init.view(-1,self.n_dimensions)
        tmax=0.999
        if self.vector_field is not None:
            tmax=1.0
        timespan = torch.linspace(tmax, 0., n_timesteps+1).to('cuda')
        samples=x_init
        dt=(timespan[1]-timespan[0]).view(-1,1).to(samples.device)
        for t in timespan:
            velocity=self.ode_forward(t,samples)
            samples=samples+velocity*dt
            if t<0.5:
                num_corrector_steps=30
                with torch.set_grad_enabled(True):
                    samples=self.simulate(samples,num_corrector_steps,simulation_fn=self.MALA_steps,time=t)
        return samples

    @torch.no_grad()
    def Jarzynski_integral(self,n_samples,n_timesteps=100,ess_threshold=0.75):
        x_init=self.prior.sample((n_samples,))
        tmax=1-1/n_timesteps
        timespan = torch.linspace(tmax, 0., n_timesteps).to('cuda')
        samples=x_init
        log_weights=torch.zeros((n_samples,1)).to('cuda')
        log_prob_prior=self.prior.log_prob(x_init).view(-1,1)
        batch_size=500
        self.setup_graph(batch_size)
        log_prob_model=torch.empty((n_samples,1)).to('cuda')
        i=0
        '''while (i+batch_size) < n_samples:
            samples_simulate=samples[i:i+batch_size]
            log_prob_simulate=self.log_prob_forward(samples_simulate.cuda(),time=1.0,torch_grad=False)
            log_prob_model[i:i+batch_size]=log_prob_simulate
            i+=batch_size
        samples_simulate=samples[i:]
        log_prob_simulate=self.log_prob_forward(samples_simulate.cuda(),time=1.0,torch_grad=False)
        log_prob_model[i:]=log_prob_simulate
        log_prob_prior=log_prob_prior.to('cuda')
        initial_log_weights=log_prob_model-log_prob_prior
        initial_log_weights=initial_log_weights-torch.logsumexp(initial_log_weights, dim=(0, 1))
        initial_weights=torch.exp(initial_log_weights)
        initial_weights=initial_weights.cpu()
        ess= (initial_weights.sum())**2/(initial_weights**2).sum()
        ess=ess/n_samples
        print('initial ess:', ess)
        samples[torch.multinomial(initial_weights.view(-1),len(initial_weights),replacement=True)]
        #with torch.set_grad_enabled(True):
        #    samples=self.simulate(samples,step_size=1e-3,steps=20,simulation_fn=self.MALA_steps,time=1.0)'''
        self.setup_graph(batch_size)
        i=0
        dt=(timespan[1]-timespan[0]).view(-1,1).to('cuda')
        for t in tqdm.tqdm(timespan):
            #do one step of ode integration and calculate dlogw
            while (i+batch_size)<n_samples:
                samples_simulate=samples[i:i+batch_size].clone().cuda()
                log_weights_simulate=log_weights[i:i+batch_size]
                velocity=self.ode_forward(t,samples_simulate)
                samples_simulate=samples_simulate+velocity*dt
                if t==1:
                    current_energy_simulate=self.prior.log_prob(samples_simulate.detach().cpu()).cuda().view(-1,1)
                else:
                    current_energy_simulate=self.log_prob_forward(samples_simulate,t-dt,torch_grad=False)
                new_energy_simulate=self.log_prob_forward(samples_simulate,t,torch_grad=False)
                log_weights_simulate=log_weights_simulate - new_energy_simulate + current_energy_simulate
                samples[i:i+batch_size]=samples_simulate.detach().cpu()
                log_weights[i:i+batch_size]=log_weights_simulate
                i+=batch_size
            self.setup_graph(n_samples-i)
            samples_simulate=samples[i:].clone().cuda()
            log_weights_simulate=log_weights[i:]
            velocity=self.ode_forward(t,samples_simulate)
            samples_simulate=samples_simulate+velocity*dt
            if t==1:
                current_energy_simulate=self.prior.log_prob(samples_simulate.detach().cpu()).cuda().view(-1,1)
            else:
                current_energy_simulate=self.log_prob_forward(samples_simulate,t-dt,torch_grad=False)
            new_energy_simulate=self.log_prob_forward(samples_simulate,t,torch_grad=False)
            log_weights_simulate=log_weights_simulate - new_energy_simulate + current_energy_simulate
            samples[i:]=samples_simulate.detach().cpu()
            log_weights[i:]=log_weights_simulate
            log_weights=log_weights-torch.logsumexp(log_weights, dim=(0, 1))
            #check if ESS < ess_threshold
            weights=torch.exp(log_weights)
            ess= (weights.sum())**2/(weights**2).sum()
            ess=ess/n_samples
            print(ess)
            if ess<ess_threshold:
                #resample samples according to weights
                weights=weights.cpu()
                samples[torch.multinomial(weights.view(-1),len(weights),replacement=True)]
                #Do a few steps of MCMC to generate variance
                with torch.set_grad_enabled(True):
                    samples=self.simulate(samples,steps=20,simulation_fn=self.MALA_steps,time=t)
                #reinitialize log weights
                log_weights=torch.zeros((n_samples,1)).to('cuda')
        #resample one final time based on weights
        log_weights=log_weights-torch.logsumexp(log_weights, dim=(0, 1))
        log_weights=log_weights.cpu()
        weights=torch.exp(log_weights)
        samples[torch.multinomial(weights.view(-1),len(weights),replacement=True)]
        with torch.set_grad_enabled(True):
            samples=self.simulate(samples,steps=20,simulation_fn=self.MALA_steps,time=0)
        return samples