import torch
from BoltzNCE.dataset.aa2_dataloader import get_ad2_dataloader
from BoltzNCE.models.ebm import GVP_EBM
from BoltzNCE.models.vector_field import GVP_vector_field
from BoltzNCE.models.interpolant import Interpolant
import tqdm
import dgl
import copy
import yaml
import argparse
import wandb
from BoltzNCE.utils.utils import load_models
from BoltzNCE.utils.arguments import get_args
from ema_pytorch import EMA


def parse_arguments():
    p =argparse.ArgumentParser()
    p.add_argument('--config', type=str, default=None)
    
    p.add_argument('--wandb', type=bool,required=False, default=False)
    p.add_argument('--wandb_project', type=str,required=False, default='BoltzNCE_alanine')
    p.add_argument('--wandb_name', type=str,required=False, default=None)
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
    dataloader_group.add_argument('--data_path',type=str,default="data/2AA-1-large/",required=False)

    training_group=p.add_argument_group('training')
    training_group.add_argument('--num_epochs', type=int,required=False, default=1000)
    training_group.add_argument('--grad_norm', type=float,required=False, default=None)

    '''train_potential_group=p.add_argument_group('train_potential')
    train_potential_group.add_argument('--window_size', type=float,required=False, default=0.025)
    train_potential_group.add_argument('--num_negatives', type=int,required=False, default=1)
    train_potential_group.add_argument('--nce_weight', type=float,required=False, default=1.0)'''

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
    
    '''graphormer_group=p.add_argument_group('graphormer')
    graphormer_group.add_argument('--embed_dim', type=int,required=False, default=128)
    graphormer_group.add_argument('--ffn_embed_dim', type=int,required=False, default=128)
    graphormer_group.add_argument('--attention_heads', type=int,required=False, default=32)
    graphormer_group.add_argument('--dropout', type=float,required=False, default=0.1)
    graphormer_group.add_argument('--attention_dropout', type=float,required=False, default=0.1)
    graphormer_group.add_argument('--input_dropout', type=float,required=False, default=0.1)
    graphormer_group.add_argument('--num_kernel', type=int,required=False, default=50)
    graphormer_group.add_argument('--blocks', type=int,required=False, default=3)'''


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
    interpolant_group.add_argument('--integration_interpolant', type=str,required=False, default='linear')

    args=p.parse_args()
    return args,p

if __name__ == "__main__":
    args,p=parse_arguments()
    args=get_args(args,p)

    if args['wandb']:
        wandb.init(project=args['wandb_project'], name=args['wandb_name'])
        wandb.config.update(args)

    dataloader = get_ad2_dataloader(**args['dataloader'])
    vector_field_model = GVP_vector_field(**args['vector_field_model'], **args['gvp']).cuda()
    