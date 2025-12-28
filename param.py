import os
import random
import numpy as np
import torch
import pprint
import yaml
import math
from omegaconf import OmegaConf
import argparse
from argparse import Namespace
def get_optimizer(optim, verbose=False):
    # Bind the optimizer
    if optim == 'rms':
        if verbose:
            print("Optimizer: Using RMSProp")
        optimizer = torch.optim.RMSprop
    elif optim == 'adam':
        if verbose:
            print("Optimizer: Using Adam")
        optimizer = torch.optim.Adam
    elif optim == 'adamw':
        if verbose:
            print("Optimizer: Using AdamW")
        optimizer = 'adamw'
    elif optim == 'adamax':
        if verbose:
            print("Optimizer: Using Adamax")
        optimizer = torch.optim.Adamax
    elif optim == 'sgd':
        if verbose:
            print("Optimizer: SGD")
        optimizer = torch.optim.SGD
    else:
        assert False, "Please add your optimizer %s in the list." % optim

    return optimizer

def parse_args():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Training configuration")

    # Added: yaml configuration file path parameter
    parser.add_argument('--config', type=str, default='./configs/test_v4_2_debug.yaml',
                        help="Path to the YAML config file")

    # Add argparse arguments for frequently changed parameters
    # parser.add_argument('--prompt_selection', type=int, default=0)
    # parser.add_argument('--save_by_step', type=int, default=50)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--output', type=str, default="")
    parser.add_argument('--wandb_run_name', type=str, default="")
    parser.add_argument('--dataset', type=str, default="m_IO")
    parser.add_argument('--dro_temperature', type=float, default=1.0)
    # parser.add_argument('--backbone', type=str, default="Qwen2.5-1.5B")
    # parser.add_argument('--version', type=str, default="base")

    # Parse the command-line arguments
    args_from_cli = parser.parse_args()

    # Use config path passed from command line
    config = OmegaConf.load(args_from_cli.config)

    # Merge YAML config with CLI args (CLI takes precedence)
    cli_dict = vars(args_from_cli)
    config = OmegaConf.merge(config, OmegaConf.create(cli_dict))

    # Initialize Config class with merged config
    args = Config(config)

    # Bind optimizer class
    verbose = False
    args.optimizer = get_optimizer(args.optim, verbose=verbose)

    # Set seeds
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

    args.fp16 = True
    args.valid_first = True
    # args.valid_ratio = 0.1

    # Gradient accumulation steps based on batch size and number of GPUs
    args.gradient_accumulation_steps = math.ceil(4 / args.batch_size / args.num_gpus)

    return args


class Config(object):
    def __init__(self, config_dict):
        """Configuration Class: set config_dict as class attributes with setattr"""
        for k, v in config_dict.items():
            setattr(self, k, v)

    @property
    def config_str(self):
        return pprint.pformat(self.__dict__)

    def __repr__(self):
        """Pretty-print configurations in alphabetical order"""
        config_str = 'Configurations\n'
        config_str += self.config_str
        return config_str

    def save(self, path):
        with open(path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            kwargs = yaml.load(f, Loader=yaml.FullLoader)
        return cls(kwargs)

if __name__ == '__main__':
    # Specify the path to your config.yaml file
    config_path = './configs/minor-HUM-small.yaml'
    config_path = './configs/minor-HUM-small-infer.yaml'
    args = parse_args(config_path)
    # print(args)
