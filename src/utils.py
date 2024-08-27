import os
import glob
import importlib
import json

import librosa
import soundfile as sf
import torch

def import_attr(import_path):
    module, attr = import_path.rsplit('.', 1)
    return getattr(importlib.import_module(module), attr)

class Params():
    """Class that loads hyperparameters from a json file.
    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__
    
def load_net(expriment_config, return_params=False):
    params = Params(expriment_config)
    #print(expriment_config)
    params.pl_module_args['init_ckpt'] = None
    
    pl_module = import_attr(params.pl_module)(**params.pl_module_args)

    with open(expriment_config) as f:
        params = json.load(f)
    
    if return_params:
        return pl_module, params
    else:
        return pl_module
    

def load_net_torch(expriment_config, return_params=False):
    params = Params(expriment_config)
    # print(params.pl_module_args)
    params.pl_module_args['init_ckpt'] = None
    params.pl_module_args["use_dp"] = False
    pl_module = import_attr(params.pl_module)(**params.pl_module_args)

    with open(expriment_config) as f:
        params = json.load(f)
    
    if return_params:
        return pl_module, params
    else:
        return pl_module


def load_pretrained(run_dir, return_params=False, map_location='cpu'):
    #print(run_dir)
    config_path = os.path.join(run_dir, 'config.json')
    pl_module, params = load_net(config_path, return_params=True)

    # Get all "best" checkpoints
    ckpts = os.listdir(os.path.join(run_dir, 'best'))

    if len(ckpts) == 0:
        raise FileNotFoundError(f"Given run ({run_dir}) doesn't have any pretrained checkpoints!")

    # Go over each checkpoint and obtain its epoch
    ckpt_epochs = []
    for ckpt in ckpts:
        epoch_idx = ckpt.find('epoch=') + len('epoch=')
        epoch_end_idx = ckpt.find('-')
        epoch = int(ckpt[epoch_idx:epoch_end_idx])
        ckpt_epochs.append((epoch, ckpt))
     
    # Sort by epoch
    ckpt_epochs = sorted(ckpt_epochs, key=lambda x: x[0])
    
    # Get checkpoint wiht latest epoch
    ckpt_path = ckpt_epochs[-1][1]
    ckpt_path = os.path.join(run_dir, 'best', ckpt_path)
    print("Loading checkpoint from", ckpt_path)
    
    # Load checkpoint
    state_dict = torch.load(ckpt_path, map_location=map_location)['state_dict']
    pl_module.load_state_dict(state_dict)
    
    if return_params:
        return pl_module, params
    else:
        return pl_module
    
def load_torch_pretrained(run_dir, return_params=False, map_location='cpu'):
    #print(run_dir)
    config_path = os.path.join(run_dir, 'config.json')

    print(config_path)
    pl_module, params = load_net_torch(config_path, return_params=True)

    # Get all "best" checkpoints
    ckpt_path = os.path.join(run_dir, 'checkpoints/best.pt')

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Given run ({run_dir}) doesn't have any pretrained checkpoints!")
    
    print("Loading checkpoint from", ckpt_path)
    
    # Load checkpoint
    # state_dict = torch.load(ckpt_path, map_location=map_location)['state_dict']
    pl_module.load_state(ckpt_path, map_location)
    print('Loaded module at epoch', pl_module.epoch)
    
    if return_params:
        return pl_module, params
    else:
        return pl_module

def read_audio_file(file_path, sr):
    """
    Reads audio file to system memory.
    """
    return librosa.core.load(file_path, mono=False, sr=sr)[0]


def write_audio_file(file_path, data, sr, subtype='PCM_16'):
    """
    Writes audio file to system memory.
    @param file_path: Path of the file to write to
    @param data: Audio signal to write (n_channels x n_samples)
    @param sr: Sampling rate
    """
    sf.write(file_path, data.T, sr, subtype)

def read_json(path):
    with open(path, 'rb') as f:
        return json.load(f)


import random
import numpy as np

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
