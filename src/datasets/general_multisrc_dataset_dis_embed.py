"""
Torch dataset object for synthetically rendered
spatial data
"""
import json
import random

from typing import Tuple
from pathlib import Path

import torch
import numpy as np
import os

import helpers.utils as utils
from src.datasets.perturbations.audio_perturbations import AudioPerturbations


class Dataset(torch.utils.data.Dataset):
    """
    Dataset of mixed waveforms and their corresponding ground truth waveforms
    recorded at different microphone.

    Data format is a pair of Tensors containing mixed waveforms and
    ground truth waveforms respectively. The tensor's dimension is formatted
    as (n_microphone, duration).

    Each scenario is represented by a folder. Multiple datapoints are generated per
    scenario. This can be customized using the points_per_scenario parameter.
    """
    def __init__(self, dataset_dirs, n_mics=6, sr=48000,
                directional=True, fair_compare = False,
                prob_neg = 0, # Unused
                perturbations = [],
                downsample = 1, mic_config = [], sig_len = 4.5,
                reference_channels = None,
                split = 'val'):
        super().__init__()
        self.dirs = [] #sorted(list(Path(rw_dir).glob('[0-9]*')))
        self.dis_embeds = []
        for _dir in dataset_dirs:
            dirpath = _dir['path'] # Get path to dataset
            limit = _dir['max_samples'] # Maximum samples to use from this dataset
            samples = sorted(list(Path(dirpath).glob('[0-9]*')))
            for j in range(limit):
                if dirpath.split('/')[-2] == "syn_1m":
                    self.dis_embeds.append(1.0)
                elif dirpath.split('/')[-2] == "syn_1_5m":
                    self.dis_embeds.append(1.5)
                elif dirpath.split('/')[-2] == "syn_2m":
                    self.dis_embeds.append(2.0)
                elif dirpath.split('/')[-2] == "glasses_1m":
                    self.dis_embeds.append(1.0)
                elif dirpath.split('/')[-2] == "glass_1_5m":
                    self.dis_embeds.append(1.5)
                elif dirpath.split('/')[-2] == "glass_2m":
                    self.dis_embeds.append(2.0)
                elif dirpath.split('/')[-2] == "hearing_1_5m":
                    self.dis_embeds.append(1.5)
                elif dirpath.split('/')[-2] == "hearing2_1_5m":
                    self.dis_embeds.append(1.5)
                elif dirpath.split('/')[-3] == "binural_1_5m":
                    self.dis_embeds.append(1.5)

                else:
                    raise ValueError("Invalid distance daatset.")
            
            samples = samples[:limit]
            self.dirs.extend(samples)

            
        self.downsample = downsample
        self.mic_lists = mic_config
        
        if reference_channels is None:
            reference_channels = [0]
        self.reference_mics = reference_channels

        # self.dirs = sorted(list(Path(input_dir).glob('00058')))
        self.valid_dirs = []
        # Physical params
        self.directional = directional
        self.n_mics = n_mics
        self.sr = sr
        self.fair_compare = fair_compare
        self.sig_len = int(sig_len*sr/downsample)
        
        # Data augmentation
        self.perturbations = AudioPerturbations(perturbations)
        
        
        ### calculate the stat

        self.split = split
        self.valid_dirs = self.dirs
        
        
    def __len__(self) -> int:
        return len(self.valid_dirs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            mixed_data - M x T
            target_voice_data - M x T
            window_idx_one_hot - 1-D
        """
        #print("Fetch idx: ", idx)
        curr_dir = self.valid_dirs[idx%len(self.valid_dirs)]
        dis_thred = self.dis_embeds[idx%len(self.valid_dirs)]

        return self.get_mixture_and_gt(curr_dir, dis_thred)

    def get_mixture_and_gt(self, curr_dir, dis_thred):
        """
        Given a target position and window size, this function figures out
        the voices inside the region and returns them as GT waveforms
        """
        # Get metadata
        metadata = utils.read_json(os.path.join(curr_dir, 'metadata.json'))

        # Iterate over different sources
        voices = [key for key in metadata.keys() if 'voice' in key]
        mics = self.mic_lists #
        mics_all = [key for key in metadata.keys() if 'mic' in key]
        
        #voice_positions = np.array([metadata[key]['position'] for key in voices])
        #mic_positions = np.array([metadata[key]['position'] for key in mics])
        
        assert (self.n_mics==len(mics))
  
        #mic_center = np.mean(mic_positions[:, :], axis = 0)

        dir_idx = int(os.path.basename(curr_dir))
        
        mixture = utils.read_audio_file_torch(os.path.join(curr_dir, 'mixture.wav'), self.downsample)
        if len(mics) < mixture.shape[0]:
            #print(mics)
            mics_num = [int(mi[-2:]) for mi in mics]
            # print(mics_num)
            mixture = mixture[mics_num, :]

        

        #print(mixture.shape)
        target_voice_inside = torch.zeros((len(self.reference_mics), mixture.shape[-1]))
        target_voice_outside = torch.zeros((1, mixture.shape[-1]))
        outside_voice = []
        inside_voice = []
        distances = []

        num_tgt_speakers = 0

        real = metadata['real']
        
        for voice in voices:
            if real:
                d = int(metadata[voice]['dis']) / 100 # Distance in meters
            else:
                d = metadata[voice]['dis'] # Distance in meters
            distances.append(d)
            
            # If inside the bubble, load and add it to gt
            if d <= dis_thred: #self.dis_threshold:
                # Add audio to the gt at the right reference channel
                
                for ch_idx, mic in enumerate(self.reference_mics):
                    audio = utils.read_audio_file_torch(os.path.join(curr_dir, f'{mics_all[mic]}_{voice}.wav'), self.downsample)
                    target_voice_inside[ch_idx] += audio[0]
                
                num_tgt_speakers += 1
        
        if num_tgt_speakers == 0:
            assert torch.abs(target_voice_inside).max() == 0, "When there are no inside speakers, the target should be zero"
        else:
            assert torch.abs(target_voice_inside).max() > 0, "When there is at least one speaker, the target should be more than zero"

        if self.sig_len < mixture.shape[-1]:
            delta_len = mixture.shape[-1] - self.sig_len
            begin_idx = np.random.randint(low = 1000, high = delta_len - 1)
            mixture = mixture[..., begin_idx:begin_idx+self.sig_len]
            target_voice_inside = target_voice_inside[..., begin_idx:begin_idx+self.sig_len]
        
        # scale = 1 / torch.abs(mixture).max() * 0.8
        # mixture = mixture * scale
        # target_voice_inside = target_voice_inside * scale
        # target_voice_outside = target_voice_outside * scale
        #if num_tgt_speakers == 0:
        #    print(curr_dir)
        #    print(metadata['room'])
        if self.split == 'train':
            mixture, target_voice_inside = self.perturbations.apply_random_perturbations(mixture, target_voice_inside)
        
        if dis_thred == 1:
            dis_embed = torch.tensor([0, 0, 1.])
        elif dis_thred == 1.5:
            dis_embed = torch.tensor([0, 1., 0])
        elif  dis_thred == 2: 
            dis_embed = torch.tensor([1., 0, 0])
        else:
            raise ValueError("Invalid distance")


        inputs = {
            'mixture':mixture.float(),
            'reference_channels':self.reference_mics,
            'dis_embed': dis_embed.float()
        }
        
        targets = {
            'target':target_voice_inside.float(),
            'targets_outside':target_voice_outside.float(),
            'num_target_speakers':num_tgt_speakers,
            'num_interfering_speakers':len(voices) - num_tgt_speakers,
            'num_noises': metadata["n_BG"]
        }
        
        return inputs, targets
