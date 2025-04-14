
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
    p.add_argument('--wandb', type=bool,required=False, default=False)
    p.add_argument('--wandb_project', type=str,required=False, default='BoltzNCE_alanine')
    p.add_argument('--wandb_name', type=str,required=False, default=None)
    p.add_argument('--num_particles',type=int,required=False, default=None)
    p.add_argument('--n_dimensions',type=int,required=False, default=None)
    p.add_argument('--dim',type=int,required=False, default=None)
    p.add_argument('--load_potential_checkpoint', type=str,required=False,default=None)
    p.add_argument('--load_vector_field_checkpoint', type=str,required=False, default=None)
    p.add_argument('--save_potential_checkpoint', type=str,required=False, default=None)
    p.add_argument('--save_vector_field_checkpoint', type=str,required=False, default=None)
    p.add_argument('--config_save_name', type=str,required=False, default='./saved_models/default_config.yaml')
    p.add_argument('--model_type', type=str,required=False, default='vector_field')

    dataloader_group=p.add_argument_group('dataloader')
    dataloader_group.add_argument('--num_workers', type=int,required=False, default=8)
    dataloader_group.add_argument('--batch_size', type=int,required=False, default=512)
    dataloader_group.add_argument('--shuffle', type=bool,required=False, default=True)
    dataloader_group.add_argument('--scaling', type=float,required=False, default=1.0)


    training_group=p.add_argument_group('training')
    training_group.add_argument('--num_epochs', type=int,required=False, default=1000)
    training_group.add_argument('--window_size', type=float,required=False, default=0.025)

    optimizer_group=p.add_argument_group('optimizer')
    optimizer_group.add_argument('--lr', type=float,required=False, default=1e-3)
    optimizer_group.add_argument('--weight_decay', type=float,required=False, default=0.0)

    
    scheduler_group=p.add_argument_group('scheduler')
    scheduler_group.add_argument('--factor', type=float,required=False, default=0.5)
    scheduler_group.add_argument('--patience', type=int,required=False, default=20)
    scheduler_group.add_argument('--verbose', type=bool,required=False, default=True)
    scheduler_group.add_argument('--min_lr', type=float,required=False, default=1e-5)
    scheduler_group.add_argument('--mode', type=str,required=False, default='min')

    potential_group=p.add_argument_group('potential_model')
    potential_group.add_argument('--num_layers', type=int,required=False, default=8)
    
    

    vector_field_group=p.add_argument_group('vector_field_model')
    vector_field_group.add_argument('--n_layers', type=int,required=False, default=5)
    vector_field_group.add_argument('--n_coord_gvps', type=int,required=False, default=1)

    gvp_group=p.add_argument_group('gvp')
    gvp_group.add_argument('--n_hidden', type=int,required=False, default=64)
    gvp_group.add_argument('--n_vec', type=int,required=False, default=16)
    gvp_group.add_argument('--n_message_gvps', type=int,required=False, default=1)
    gvp_group.add_argument('--n_update_gvps', type=int,required=False, default=1)
    gvp_group.add_argument('--use_dst_feats', type=bool,required=False, default=False)
    gvp_group.add_argument('--vector_gating', type=bool,required=False, default=True)

    interpolant_group=p.add_argument_group('interpolant')
    interpolant_group.add_argument('--interpolant_type', type=str,required=False, default='linear')

    args=p.parse_args()
    return args,p

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
    #TODO: seperate out interpolant and dataloader scaling
    args['interpolant']['scaling']=args['dataloader']['scaling']
    return args

def args_to_dict(args,p):
    args_dict = {}
    for group in p._action_groups:
        
        if group.title == 'positional arguments':
            continue
        if group.title == 'options':
            for action in group._group_actions:
                if action.dest != 'help':
                    args_dict[action.dest] = args.__dict__[action.dest]
        else:
            args_dict[group.title] = {}
            for action in group._group_actions:
                if action.dest != 'help':
                    args_dict[group.title][action.dest] = args.__dict__[action.dest]
    return args_dict

def get_args():
    args,p=parse_arguments()
    args=args_to_dict(args,p)
    if args['config'] is not None:
        with open(args['config'], 'r') as f:
            config = yaml.safe_load(f)
        for key, value in config.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    args[key][sub_key] = sub_value
            else:
                args[key] = value
    return args

def train_vector_field(args,dataloader: alanine_dataset,interpolant_obj: Interpolant, vector_model , optim_vector, scheduler_vector,num_epochs,window_size):
    for epoch in tqdm.tqdm(range(num_epochs)):
        losses=[]
        for it,g in enumerate(dataloader):
            optim_vector.zero_grad()
            g=g.to('cuda')
            batch_size=g.batch_size
            t=torch.rand(batch_size,1).cuda()
            sigma_t=interpolant_obj.sigma(t)
            sigma_t=sigma_t.view(-1,1)
            g=interpolant_obj.get_xt_and_vt(t,g)
            vector_target=g.ndata['v']
            vector_target=vector_target.view(-1,interpolant_obj.dim)
            vector=vector_model(t,g)
            vector=vector.view(-1,interpolant_obj.dim)
            loss_vector=torch.mean((vector - vector_target)**2)
            if args['wandb']:
                wandb.log({"vector_loss": loss_vector.item()})
            loss_vector.backward()
            optim_vector.step()
            losses.append(loss_vector.item())
        scheduler_vector.step(sum(losses)/len(losses))
        print(f"Epoch {epoch}: Vector Field Loss: {sum(losses)/len(losses)}")
    return vector_model

def train_potential(args, dataloader: alanine_dataset,interpolant_obj: Interpolant, potential_model , optim_potential, scheduler_potential,num_epochs: int,window_size):
    for epoch in tqdm.tqdm(range(num_epochs)):
        losses=[]
        losses_score=[]
        losses_nce=[]
        for it,g in enumerate(dataloader):
            optim_potential.zero_grad()
            g=g.to('cuda')
            batch_size=g.batch_size
            t=torch.rand(batch_size,1).cuda()
            sigma_t=interpolant_obj.sigma(t)
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
            if args['wandb']:
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
    args=merge_model_args(args)
    
    if args['wandb']:
        wandb.init(project=args['wandb_project'], name=args['wandb_name'])
        wandb.config.update(args)
    
    # Save the config to a file
    with open(args['config_save_name'], 'w') as f:
        yaml.dump(args, f)

    # Load the dataset and create the dataloader
    atom_types, h_initial, dataset, dataloader = get_alanine_types_dataset_dataloaders(**args['dataloader'])

    potential_model=GVP_EBM(**args['potential_model'],**args['gvp']).cuda()
    pytorch_total_params = sum(p.numel() for p in potential_model.parameters())
    print(f"Total number of parameters in potential model: {pytorch_total_params}")
    if args['load_potential_checkpoint'] is not None:
        potential_model.load_state_dict(torch.load(args['load_potential_checkpoint']))
    vector_field=GVP_vector_field(**args['vector_field_model'],**args['gvp']).cuda()
    pytorch_total_params = sum(p.numel() for p in vector_field.parameters())
    print(f"Total number of parameters in vector field model: {pytorch_total_params}")
    if args['load_vector_field_checkpoint'] is not None:
        vector_field.load_state_dict(torch.load(args['load_vector_field_checkpoint']))
    # Initialize the interpolant
    interpolant_obj=Interpolant(h_initial=h_initial,potential_function=potential_model,vector_field=vector_field,**args['interpolant']).cuda()

    optim_potential=torch.optim.Adam(potential_model.parameters(), **args['optimizer'])
    scheduler_potential=torch.optim.lr_scheduler.ReduceLROnPlateau(optim_potential,**args['scheduler'])
    optim_vector=torch.optim.Adam(vector_field.parameters(), **args['optimizer'])
    scheduler_vector=torch.optim.lr_scheduler.ReduceLROnPlateau(optim_vector,**args['scheduler'])

    if args['model_type']=='vector_field':
        vector_field=train_vector_field(args, dataloader,interpolant_obj, vector_field , optim_vector, scheduler_vector,**args['training'])
        torch.save(vector_field.state_dict(), args['save_vector_field_checkpoint'])
        interpolant_obj.vector_field=vector_field

    elif args['model_type']=='potential':
        #hardcoded arguments for now
        samples_np=gen_samples(n_samples=500,n_sample_batches=200,interpolant_obj=interpolant_obj,integral_type='ode',n_timesteps=1000)
        atom_types, h_initial, dataset, dataloader = get_alanine_types_dataset_dataloaders(torch.from_numpy(samples_np).float(),**args['dataloader'])
        potential_model=train_potential(args, dataloader,interpolant_obj, potential_model , optim_potential, scheduler_potential,**args['training'])
        torch.save(potential_model.state_dict(), args['save_potential_checkpoint']) 
        interpolant_obj.potential_function=potential_model
    else:
        raise ValueError("Model type must be either 'vector_field' or 'potential'") 
    






