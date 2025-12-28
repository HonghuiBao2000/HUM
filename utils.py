import re
import numpy as np
import torch
import torch.distributed as dist
import collections

amazon_dataset2fullname = {
    'Beauty': 'All_Beauty',
    'Fashion': 'AMAZON_FASHION',
    'Appliances': 'Appliances',
    'Arts': 'Arts_Crafts_and_Sewing',
    'Automotive': 'Automotive',
    'Books': 'Books',
    'CDs': 'CDs_and_Vinyl',
    'Cell': 'Cell_Phones_and_Accessories',
    'Clothing': 'Clothing_Shoes_and_Jewelry',
    'Music': 'Digital_Music',
    'Electronics': 'Electronics',
    'Gift': 'Gift_Cards',
    'Food': 'Grocery_and_Gourmet_Food',
    'Home': 'Home_and_Kitchen',
    'Scientific': 'Industrial_and_Scientific',
    'Kindle': 'Kindle_Store',
    'Luxury': 'Luxury_Beauty',
    'Magazine': 'Magazine_Subscriptions',
    'Movies': 'Movies_and_TV',
    'Instruments': 'Musical_Instruments',
    'Office': 'Office_Products',
    'Garden': 'Patio_Lawn_and_Garden',
    'Pantry': 'Prime_Pantry',
    'Pet': 'Pet_Supplies',
    'Software': 'Software',
    'Sports': 'Sports_and_Outdoors',
    'Tools': 'Tools_and_Home_Improvement',
    'Toys': 'Toys_and_Games',
    'Games': 'Video_Games',
}

class LossMeter(object):
    def __init__(self, maxlen=100):
        """Computes and stores the running average"""
        self.vals = collections.deque([], maxlen=maxlen)

    def __len__(self):
        return len(self.vals)

    def update(self, new_val):
        self.vals.append(new_val)

    @property
    def val(self):
        return sum(self.vals) / len(self.vals)

    def __repr__(self):
        return str(self.val)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_state_dict(state_dict_path, loc='cpu'):
    state_dict = torch.load(state_dict_path, map_location=loc)
    # Change Multi GPU to single GPU
    original_keys = list(state_dict.keys())
    for key in original_keys:
        if key.startswith("module."):
            new_key = key[len("module."):]
            state_dict[new_key] = state_dict.pop(key)
    return state_dict


def print_rank0(msg, rank=0):
    """Compatibility with original print_rank0 function, prioritizing HUM Logger"""
    if rank == 0:
        # 尝试使用全局logger
        try:
            from hum_logger import get_global_logger
            global_logger = get_global_logger()
            if global_logger:
                global_logger.print_rank0(msg)
                return
        except ImportError:
            pass
        
        # 回退到原始print
        print(msg)

