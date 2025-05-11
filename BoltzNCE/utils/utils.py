import torch
from models.ebm import GVP_EBM, graphormer_EBM
from models.vector_field import GVP_vector_field
from models.interpolant import Interpolant


def load_models(args,h_initial):
    if args['potential_type'] == 'gvp':
        potential_model=GVP_EBM(**args['potential_model'],**args['gvp']).cuda()
    elif args['potential_type'] == 'graphormer':
        potential_model=graphormer_EBM(**args['potential_model'],**args['graphormer']).cuda()
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
    interpolant_obj=Interpolant(h_initial=h_initial,potential_function=potential_model,vector_field=vector_field, **args['interpolant']).cuda()
    return potential_model, vector_field, interpolant_obj