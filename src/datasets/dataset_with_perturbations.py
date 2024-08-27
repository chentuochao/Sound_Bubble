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

import sep.helpers.utils as utils
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
    def __init__(self, input_dir, n_mics=6, sr=48000,
                dis_threshold = 1.5, directional=True, fair_compare = False,
                 prob_neg = 0, # Unused
                 perturbations = [],
                 downsample = 1, mic_config = [], sig_len = 4.5,
                split = 'val'):
        super().__init__()
        self.dirs = sorted(list(Path(input_dir).glob('[0-9]*')))
        self.downsample = downsample
        self.mic_lists = mic_config
        # self.dirs = sorted(list(Path(input_dir).glob('00058')))
        self.valid_dirs = []
        # Physical params
        self.directional = directional
        self.n_mics = n_mics
        self.sr = sr
        self.dis_threshold = dis_threshold
        self.fair_compare = fair_compare
        self.sig_len = int(sig_len*sr/downsample)
        
        # Data augmentation
        self.perturbations = AudioPerturbations(perturbations)
        
        
        ### calculate the stat
        near_num = 0
        far_num = 0
        idx = 0

        self.split = split

        dis_ths = [1, 1.5, 2, 2.5, 3, 3.5, 4, 100]
        dis_nums = [0 for i in range(len(dis_ths))]

        for curr_dir in self.dirs:
            if os.path.exists(Path(curr_dir) / 'metadata.json'):
                self.valid_dirs.append(curr_dir)
            else:
                continue
            with open(Path(curr_dir) / 'metadata.json') as json_file:
                metadata = json.load(json_file)        
                voice_keys = [key for key in metadata.keys() if 'voice' in key]
                find_close = False
                dises = []
                for v in voice_keys:  
                    dises.append(metadata[v]["dis"])
                    for i, dis_th in enumerate(dis_ths):
                        if metadata[v]["dis"] < dis_th:
                            dis_nums[i] += 1
                            break
                
                
            #if find_close:
            #    print(idx, dises)
            idx += 1
        print("Dataset distribution: near - ", dis_nums)

        ### calculate the stat
        near_num = 0
        far_num = 0
        for curr_dir in self.valid_dirs:
            with open(Path(curr_dir) / 'metadata.json') as json_file:
                metadata = json.load(json_file)        
                voice_keys = [key for key in metadata.keys() if 'voice' in key]
                for v in voice_keys:  
                    if metadata[v]["dis"] < self.dis_threshold:
                        near_num += 1
                    else:
                        far_num += 1
        print("Dataset distribution: near - ", near_num, "far - ", far_num )
        print("dataset number: ", len(self.valid_dirs))
        
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

        return self.get_mixture_and_gt(curr_dir)

    def get_mixture_and_gt(self, curr_dir):
        """
        Given a target position and window size, this function figures out
        the voices inside the region and returns them as GT waveforms
        """
        # Get metadata
        metadata = utils.read_json(os.path.join(curr_dir, 'metadata.json'))

        # Iterate over different sources
        voices = [key for key in metadata.keys() if 'voice' in key]
        mics = self.mic_lists #[key for key in metadata.keys() if 'mic' in key]
        
        voice_positions = np.array([metadata[key]['position'] for key in voices])
        mic_positions = np.array([metadata[key]['position'] for key in mics])
        
        assert (self.n_mics==len(mics))
  
        mic_center = np.mean(mic_positions[:, :], axis = 0)

        dir_idx = int(os.path.basename(curr_dir))
        
        mixture = utils.read_audio_file_torch(os.path.join(curr_dir, 'mixture.wav'), self.downsample)
        
        #print(mixture.shape)
        target_voice_inside = torch.zeros((1, mixture.shape[-1]))
        target_voice_outside = torch.zeros((1, mixture.shape[-1]))
        outside_voice = []
        inside_voice = []
        distances = []

        num_tgt_speakers = 0
        
        for voice in voices:
            d = int(metadata[voice]['dis']) / 100 # Distance in meters
            distances.append(d)
            
            # If inside the bubble, load and add it to gt
            if d <= self.dis_threshold:
                audio = utils.read_audio_file_torch(os.path.join(curr_dir, f'{mics[0]}_{voice}.wav'), self.downsample)
                target_voice_inside += audio
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
        if num_tgt_speakers == 0:
            print(curr_dir)
            print(metadata['room'])
        if self.split == 'train':
            mixture, target_voice_inside = self.perturbations.apply_random_perturbations(mixture, target_voice_inside)
        
        inputs = {
            'mixture':mixture.float()
        }
        
        targets = {
            'target':target_voice_inside.float(),
            'targets_outside':target_voice_outside.float(),
            'num_target_speakers':num_tgt_speakers,
            'num_interfering_speakers':len(voices) - num_tgt_speakers
        }
        
        return inputs, targets
