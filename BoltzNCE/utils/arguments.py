import argparse
import yaml


def args_to_dict(args,p: argparse.ArgumentParser):
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

def get_args(args,p):
    args=args_to_dict(args,p)
    if args['config'] is not None:
        with open(args['config'], 'r') as f:
            config = yaml.safe_load(f)
        for key, value in config.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if key not in args:
                        args[key] = {}
                    args[key][sub_key] = sub_value
            else:
                args[key] = value
    return args