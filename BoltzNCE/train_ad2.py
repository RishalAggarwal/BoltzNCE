
import torch
from dataset.ad2_dataset import get_alanine_types_dataset_dataloaders,alanine_dataset
from models.ebm import GVP_EBM
from models.vector_field import GVP_vector_field
from models.interpolant import Interpolant
import tqdm
import dgl
import yaml
import argparse
import wandb
from infer_ad2 import gen_samples

#alanine globals
global_vars={}
global_vars['num_particles'] = 22
global_vars['n_dimensions'] = 3
global_vars['dim']=66

def parse_arguments():
    p =argparse.ArgumentParser()
    p.add_argument('--config', type=str, default=None)
    p.add_argument('--wandb', type=bool, default=False)
    p.add_argument('--wandb_project', type=str, default='BoltzNCE_alanine')
    p.add_argument('--wandb_name', type=str, default=None)
    p.add_argument('--num_particles',type=int, default=None)
    p.add_argument('--n_dimensions',type=int, default=None)
    p.add_argument('--dim',type=int, default=None)
    p.add_argument('--load_potential_checkpoint', type=str, default=None)
    p.add_argument('--load_vector_field_checkpoint', type=str, default=None)
    p.add_argument('--save_potential_checkpoint', type=str, default=None)
    p.add_argument('--save_vector_field_checkpoint', type=str, default=None)
    p.add_argument('--config_save_name', type=str, default='./saved_models/default_config.yaml')
    p.add_argument('--model_type', type=str, default='vector_field')

    dataloader_group=p.add_argument_group('dataloader')
    dataloader_group.add_argument('--num_workers', type=int, default=8)
    dataloader_group.add_argument('--batch_size', type=int, default=512)
    dataloader_group.add_argument('--shuffle', type=bool, default=True)

    training_group=p.add_argument_group('training')
    training_group.add_argument('--num_epochs', type=int, default=1000)
    training_group.add_argument('--window_size', type=float, default=0.025)

    optimizer_group=p.add_argument_group('optimizer')
    optimizer_group.add_argument('lr', type=float, default=1e-3)
    optimizer_group.add_argument('weight_decay', type=float, default=0.0)

    
    scheduler_group=p.add_argument_group('scheduler')
    scheduler_group.add_argument('factor', type=float, default=0.5)
    scheduler_group.add_argument('patience', type=int, default=20)
    scheduler_group.add_argument('verbose', type=bool, default=True)
    scheduler_group.add_argument('min_lr', type=float, default=1e-5)
    scheduler_group.add_argument('mode', type=str, default='min')

    potential_group=p.add_argument_group('potential_model')
    potential_group.add_argument('n_layers', type=int, default=8)
    potential_group.add_argument('n_hidden', type=int, default=64)
    potential_group.add_argument('n_vec', type=int, default=16)
    potential_group.add_argument('n_message_gvps', type=int, default=1)
    potential_group.add_argument('n_update_gvps', type=int, default=1)
    potential_group.add_argument('use_dst_feats', type=bool, default=False)
    potential_group.add_argument('vector_gating', type=bool, default=True)
    

    vector_field_group=p.add_argument_group('vector_field_model')
    vector_field_group.add_argument('n_layers', type=int, default=5)
    vector_field_group.add_argument('n_hidden', type=int, default=64)
    vector_field_group.add_argument('n_vec', type=int, default=16)
    vector_field_group.add_argument('n_message_gvps', type=int, default=1)
    vector_field_group.add_argument('n_update_gvps', type=int, default=1)
    vector_field_group.add_argument('n_coord_gvps', type=int, default=1)
    vector_field_group.add_argument('use_dst_feats', type=bool, default=False)
    vector_field_group.add_argument('vector_gating', type=bool, default=True)
    vector_field_group.add_argument('model_checkpoint', type=str, default=None)

    interpolant_group=p.add_argument_group('interpolant')
    interpolant_group.add_argument('interpolant_type', type=str, default='linear')

    args=p.parse_args()
    return args

def merge_global_args(args):
    #merge args and global_vars
    for key, value in global_vars.items():
        args[key]=value
    return args


def merge_model_args(args):
    #add arguments to models that are common
    args['potential_model']['num_particles']=args['num_particles']
    args['vector_field_model']['num_particles']=args['num_particles']
    args['interpolant']['num_particles']=args['num_particles']
    args['interpolant']['n_dimensions']=args['n_dimensions']
    args['interpolant']['dim']=args['dim']
    return args

def get_args():
    args=parse_arguments()
    if args.config is not None:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)
    return args

def train_vector_field(dataloader: alanine_dataset,interpolant_obj: Interpolant, vector_model , optim_vector, scheduler_vector,n_epochs: int,window_size):
    for epoch in tqdm.tqdm(range(n_epochs)):
        losses=[]
        for it,g in enumerate(dataloader):
            optim_vector.zero_grad()
            g=g.to('cuda')
            batch_size=g.batch_size
            t=torch.rand(batch_size,1).cuda()
            sigma_t=interpolant_obj.sigma_t(t)
            sigma_t=sigma_t.view(-1,1)
            g=interpolant_obj.get_xt_and_vt(t,g)
            vector_target=g.ndata['v']
            vector_target=vector_target.view(-1,interpolant_obj.dim)
            vector=vector_model(t,g)
            vector=vector.view(-1,interpolant_obj.dim)
            loss_vector=torch.mean((vector - vector_target)**2)
            wandb.log({"vector_loss": loss_vector.item()})
            loss_vector.backward()
            optim_vector.step()
            losses.append(loss_vector.item())
        scheduler_vector.step(sum(losses)/len(losses))
        print(f"Epoch {epoch}: Vector Field Loss: {sum(losses)/len(losses)}")
    return vector_model

def train_potential( dataloader: alanine_dataset,interpolant_obj: Interpolant, potential_model , optim_potential, scheduler_potential,n_epochs: int,window_size):
    for epoch in tqdm.tqdm(range(n_epochs)):
        losses=[]
        losses_score=[]
        losses_nce=[]
        for it,g in enumerate(dataloader):
            optim_potential.zero_grad()
            g=g.to('cuda')
            batch_size=g.batch_size
            t=torch.rand(batch_size,1).cuda()
            sigma_t=interpolant_obj.sigma_t(t)
            sigma_t=sigma_t.view(-1,1)
            g=interpolant_obj.get_xt_and_vt(t,g)
            score_target=g.ndata['x1']
            score_target=score_target.view(-1,interpolant_obj.dim)
            score,ll_positives=potential_model(t,g)
            score=score.view(-1,interpolant_obj.dim)
            loss_score=torch.mean((sigma_t*score + score_target)**2)
            ll_positives=potential_model(t,g,return_logprob=True)
            t_negative=torch.randn(batch_size,1).cuda() * window_size + t
            t_negative=torch.clamp(t_negative,0,1)
            ll_negatives=potential_model(t_negative,g,return_logprob=True)
            loss_nce=-torch.mean(ll_positives-torch.logsumexp(torch.cat([ll_negatives.unsqueeze(1),ll_positives.unsqueeze(1)],dim=1),dim=1))
            loss=loss_score + loss_nce
            wandb.log({"potential_loss": loss.item()})
            wandb.log({"potential_score_loss": loss_score.item()})
            wandb.log({"potential_nce_loss": loss_nce.item()})
            loss.backward()
            optim_potential.step()
            losses.append(loss.item())
            losses_score.append(loss_score.item())
            losses_nce.append(loss_nce.item())
        scheduler_potential.step(sum(losses)/len(losses))
        print(f"Epoch {epoch}: Potential Loss: {sum(losses)/len(losses)}")
        print(f"Epoch {epoch}: Potential Score Loss: {sum(losses_score)/len(losses_score)}")
        print(f"Epoch {epoch}: Potential NCE Loss: {sum(losses_nce)/len(losses_nce)}")
    return potential_model

if __name__ == "__main__":
    args=get_args()
    args=merge_global_args(args)
    
    if args.wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_name)
        wandb.config.update(args)
    
    args=merge_model_args(args)
    # Save the config to a file
    with open(args.config_save_name, 'w') as f:
        yaml.dump(args, f)

    # Load the dataset and create the dataloader
    atom_types, h_initial, dataset, dataloader = get_alanine_types_dataset_dataloaders(**args['dataloader'])

    potential_model=GVP_EBM(**args['potential_model']).cuda()
    if args['potential_model_checkpoint'] is not None:
        potential_model.load_state_dict(torch.load(potential_model.model_checkpoint))
    vector_field=GVP_vector_field(**args['vector_field_model']).cuda()
    if args['vector_field_model_checkpoint'] is not None:
        vector_field.load_state_dict(torch.load(vector_field.model_checkpoint))
    # Initialize the interpolant
    interpolant_obj=Interpolant(h_initial=h_initial,potential_function=potential_model,vector_field=vector_field,**args['interpolant']).cuda()

    optim_potential=torch.optim.Adam(potential_model.parameters(), **args['optimizer'])
    scheduler_potential=torch.optim.lr_scheduler.ReduceLROnPlateau(optim_potential,**args['scheduler'])
    optim_vector=torch.optim.Adam(vector_field.parameters(), **args['optimizer'])
    scheduler_vector=torch.optim.lr_scheduler.ReduceLROnPlateau(optim_vector,**args['scheduler'])

    if args['model_type']=='vector_field':
        vector_field=train_vector_field( dataloader,interpolant_obj, vector_field , optim_vector, scheduler_vector,**args['training'])
        torch.save(vector_field.state_dict(), args['save_vector_field_checkpoint'])
        interpolant_obj.vector_field=vector_field

    elif args['model_type']=='potential':
        samples_np=gen_samples(n_samples=500,n_sample_batches=200,interpolant_obj=interpolant_obj,integral_type='ode',n_timesteps=1000)
        atom_types, h_initial, dataset, dataloader = get_alanine_types_dataset_dataloaders(samples_np,**args['dataloader'])
        potential_model=train_potential( dataloader,interpolant_obj, potential_model , optim_potential, scheduler_potential,**args['training'])
        torch.save(potential_model.state_dict(), args['save_potential_checkpoint']) 
        interpolant_obj.potential_function=potential_model
    else:
        raise ValueError("Model type must be either 'vector_field' or 'potential'") 
    






