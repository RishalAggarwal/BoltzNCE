
import torch
from dataset.ad2_dataset import get_alanine_types_dataset_dataloaders,alanine_dataset
from models.ebm import GVP_EBM
from models.vector_field import GVP_vector_field
from models.interpolant import Interpolant
import tqdm
import dgl
import copy
import yaml
import argparse
import wandb
from utils.utils import load_models
from utils.arguments import get_args
from infer_ad2 import gen_samples
from ema_pytorch import EMA

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
    p.add_argument('--potential_type', type=str,required=False, default='gvp')
    p.add_argument('--optimizer_type', type=str,required=False, default='adam')
    p.add_argument('--ema', type=bool,required=False, default=False)

    dataloader_group=p.add_argument_group('dataloader')
    dataloader_group.add_argument('--num_workers', type=int,required=False, default=8)
    dataloader_group.add_argument('--batch_size', type=int,required=False, default=512)
    dataloader_group.add_argument('--shuffle', type=bool,required=False, default=True)
    dataloader_group.add_argument('--scaling', type=float,required=False, default=1.0)


    training_group=p.add_argument_group('training')
    training_group.add_argument('--num_epochs', type=int,required=False, default=1000)
    training_group.add_argument('--grad_norm', type=float,required=False, default=None)

    train_potential_group=p.add_argument_group('train_potential')
    train_potential_group.add_argument('--window_size', type=float,required=False, default=0.025)
    train_potential_group.add_argument('--num_negatives', type=int,required=False, default=1)
    train_potential_group.add_argument('--nce_weight', type=float,required=False, default=1.0)

    train_vector_group=p.add_argument_group('train_vector')
    train_vector_group.add_argument('--endpoint', type=bool,required=False, default=False)
    train_vector_group.add_argument('--self_conditioning', type=bool,required=False, default=False)
    train_vector_group.add_argument('--tweight_max', type=float,required=False, default=1.5)

    optimizer_group=p.add_argument_group('optimizer')
    optimizer_group.add_argument('--lr', type=float,required=False, default=1e-3)
    optimizer_group.add_argument('--weight_decay', type=float,required=False, default=0.0)

    ema_group=p.add_argument_group('ema_model')
    ema_group.add_argument('--beta', type=float,required=False, default=0.999)
    ema_group.add_argument('--update_every', type=int,required=False, default=10)
    ema_group.add_argument('--update_model_with_ema_every', type=int,required=False, default=None)
    
    scheduler_group=p.add_argument_group('scheduler')
    scheduler_group.add_argument('--factor', type=float,required=False, default=0.5)
    scheduler_group.add_argument('--patience', type=int,required=False, default=20)
    scheduler_group.add_argument('--verbose', type=bool,required=False, default=True)
    scheduler_group.add_argument('--min_lr', type=float,required=False, default=1e-5)
    scheduler_group.add_argument('--mode', type=str,required=False, default='min')

    potential_group=p.add_argument_group('potential_model')
    potential_group.add_argument('--num_layers', type=int,required=False, default=8)
    
    graphormer_group=p.add_argument_group('graphormer')
    graphormer_group.add_argument('--embed_dim', type=int,required=False, default=128)
    graphormer_group.add_argument('--ffn_embed_dim', type=int,required=False, default=128)
    graphormer_group.add_argument('--attention_heads', type=int,required=False, default=32)
    graphormer_group.add_argument('--dropout', type=float,required=False, default=0.1)
    graphormer_group.add_argument('--attention_dropout', type=float,required=False, default=0.1)
    graphormer_group.add_argument('--input_dropout', type=float,required=False, default=0.1)
    graphormer_group.add_argument('--num_kernel', type=int,required=False, default=50)
    graphormer_group.add_argument('--blocks', type=int,required=False, default=3)


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
    interpolant_group.add_argument('--ot', type=bool,required=False, default=False)

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
    args['interpolant']['endpoint']=args['train_vector']['endpoint']
    args['vector_field_model']['self_conditioning']=args['train_vector']['self_conditioning']
    args['interpolant']['self_conditioning']=args['train_vector']['self_conditioning']
    return args



def train_vector_field(args,dataloader: alanine_dataset,interpolant_obj: Interpolant, vector_model , optim_vector, scheduler_vector,num_epochs,grad_norm,endpoint,self_conditioning,tweight_max):
    if args['ema']:
        ema = EMA(vector_model, allow_different_devices = True,**args['ema_model'])
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
            if endpoint:
                vector_target=g.ndata['x0']
                alpha_t=interpolant_obj.alpha(t).view(-1,1)
                alpha_dot_t=interpolant_obj.alpha_dot(t).view(-1,1)
                time_weight=torch.min(torch.max(torch.tensor([0.005]).to(alpha_t.device),torch.abs((alpha_dot_t*sigma_t - alpha_t)/sigma_t)),torch.tensor([tweight_max]).to(alpha_t.device)).view(-1,1)
                if self_conditioning and torch.rand(1).item() < 0.5:
                    with torch.no_grad():
                        vector_condition=vector_model(t,g)
                    vector=vector_model(t,g,condition=vector_condition)
                else:
                    vector=vector_model(t,g)
            else:
                vector_target=g.ndata['v']
                vector=vector_model(t,g)
                time_weight=torch.ones_like(t)
            vector_target=vector_target.view(-1,interpolant_obj.dim)
            vector=vector.view(-1,interpolant_obj.dim)
            time_weight=time_weight.to(device=vector.device)
            loss_vector=torch.mean(time_weight*(vector - vector_target)**2)
            if args['wandb']:
                wandb.log({"vector_loss": loss_vector.item()})
            loss_vector.backward()
            if grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(vector_model.parameters(), grad_norm)
            optim_vector.step()
            if args['ema']:
                ema.update()
            losses.append(loss_vector.item())
        scheduler_vector.step(sum(losses)/len(losses))
        print(f"Epoch {epoch}: Vector Field Loss: {sum(losses)/len(losses)}")
    if args['ema']:
        vector_model=ema.ema_model
    return vector_model

def train_potential(args, dataloader: alanine_dataset,interpolant_obj: Interpolant, potential_model: GVP_EBM, optim_potential:torch.optim.Optimizer, scheduler_potential: torch.optim.lr_scheduler.ReduceLROnPlateau,num_epochs: int,window_size,num_negatives: int,nce_weight: float,grad_norm: float):
    #hardcoding arguments for now
    if args['ema']:
        ema = EMA(potential_model, beta=0.999, allow_different_devices = True)
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
            if window_size<1:
                t_negative=torch.randn(batch_size,num_negatives).cuda() * window_size + t
            else:
                t_negative=torch.rand(batch_size,num_negatives).cuda()
            t_negative=torch.clamp(t_negative,0,1)
            t_negative=t_negative.transpose(0,1).reshape(-1,1)
            large_g=dgl.batch([g]* num_negatives)
            ll_negatives=potential_model(t_negative,large_g,return_logprob=True)
            ll_negatives=ll_negatives.view(num_negatives,-1).transpose(0,1)
            loss_nce=-torch.mean(ll_positives-torch.logsumexp(torch.cat([ll_negatives,ll_positives],dim=1),dim=1).view(-1,1))
            loss=loss_score + nce_weight*loss_nce
            if args['wandb']:
                wandb.log({"potential_loss": loss.item()})
                wandb.log({"potential_score_loss": loss_score.item()})
                wandb.log({"potential_nce_loss": loss_nce.item()})
            loss.backward()
            if grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(potential_model.parameters(), grad_norm)
            optim_potential.step()
            if args['ema']:
                ema.update()
            losses.append(loss.item())
            losses_score.append(loss_score.item())
            losses_nce.append(loss_nce.item())
        scheduler_potential.step(sum(losses)/len(losses))
        print(f"Epoch {epoch}: Potential Loss: {sum(losses)/len(losses)}")
        print(f"Epoch {epoch}: Potential Score Loss: {sum(losses_score)/len(losses_score)}")
        print(f"Epoch {epoch}: Potential NCE Loss: {sum(losses_nce)/len(losses_nce)}")
    if args['ema']:
        potential_model=ema.ema_model
    return potential_model

if __name__ == "__main__":
    args,p=parse_arguments()
    args=get_args(args,p)
    args=merge_global_args(args)
    args=merge_model_args(args)
    
    if args['wandb']:
        wandb.init(project=args['wandb_project'], name=args['wandb_name'])
        wandb.config.update(args)
    
    

    # Load the dataset and create the dataloader
    h_initial, dataset, dataloader = get_alanine_types_dataset_dataloaders(**args['dataloader'])

    potential_model, vector_field, interpolant_obj=load_models(args,h_initial=h_initial)

    optimizer=torch.optim.Adam
    if args['optimizer_type']=='adam':
        optimizer=torch.optim.Adam
    elif args['optimizer_type']=='adamw':
        optimizer=torch.optim.AdamW
    else:
        raise ValueError("Optimizer type must be either 'adam' or 'adamw'")
    optim_potential=optimizer(potential_model.parameters(), **args['optimizer'])
    scheduler_potential=torch.optim.lr_scheduler.ReduceLROnPlateau(optim_potential,**args['scheduler'])
    optim_vector=optimizer(vector_field.parameters(), **args['optimizer'])
    scheduler_vector=torch.optim.lr_scheduler.ReduceLROnPlateau(optim_vector,**args['scheduler'])

    if args['model_type']=='vector_field':
        vector_field=train_vector_field(args, dataloader,interpolant_obj, vector_field , optim_vector, scheduler_vector,**args['training'],**args['train_vector'])
        torch.save(vector_field.state_dict(), args['save_vector_field_checkpoint'])
        interpolant_obj.vector_field=vector_field

    elif args['model_type']=='potential':
        #TODO hardcoded arguments for now
        samples_np,_=gen_samples(n_samples=500,n_sample_batches=200,interpolant_obj=interpolant_obj,integral_type='ode',n_timesteps=1000)
        h_initial, dataset, dataloader = get_alanine_types_dataset_dataloaders(dataset=torch.from_numpy(samples_np).float(),**args['dataloader'])
        potential_model=train_potential(args, dataloader,interpolant_obj, potential_model , optim_potential, scheduler_potential,**args['training'],**args['train_potential'])
        torch.save(potential_model.state_dict(), args['save_potential_checkpoint']) 
        interpolant_obj.potential_function=potential_model
    else:
        raise ValueError("Model type must be either 'vector_field' or 'potential'") 
    
    # Save the config to a file
    with open(args['config_save_name'], 'w') as f:
        yaml.dump(args, f)





