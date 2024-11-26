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


class SyntheticDataset(torch.utils.data.Dataset):
    """
    Synthetic Dataset of mixed waveforms and their corresponding ground truth waveforms
    recorded at different microphone.

    Data format is a pair of Tensors containing mixed waveforms and
    ground truth waveforms respectively. The tensor's dimension is formatted
    as (n_microphone, duration).

    Each scenario is represented by a folder. Multiple datapoints are generated per
    scenario. This can be customized using the points_per_scenario parameter.
    """
    def __init__(self, input_dir, n_mics=2, sr=48000, dis_threshold = 1, directional=True, fair_compare = False, prob_neg = 0, downsample = 1, mic_config = [], sig_len = 4.5):
        super().__init__()
        self.dirs = sorted(list(Path(input_dir).glob('[0-9]*')))
        self.prob_neg = prob_neg
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
        ### calculate the stat
        near_num = 0
        far_num = 0
        idx = 0

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

        metadata, mixture, target_voice_inside, target_voice_outside = self.get_mixture_and_gt(curr_dir)

        return (mixture.float(),
                target_voice_inside.float(),
                target_voice_outside.float())

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


        # TODO: Include error but not in gt computation
        
        if np.random.rand() < self.prob_neg:
            negative_sample = True
        else:
            negative_sample= False

        mixture = []

        for mic in mics:
            # channel = utils.read_audio_file(os.path.join(curr_dir, mic) + '_mixed.wav', self.sr)
            channel = utils.read_audio_file_torch(os.path.join(curr_dir, mic) + '_mixed.wav', self.downsample)
            mixture.append(channel)

        mixture = torch.vstack(mixture)
        #print(mixture.shape)
        target_voice_inside = torch.zeros((1, mixture.shape[-1]))
        target_voice_outside = torch.zeros((1, mixture.shape[-1]))
        outside_voice = []
        inside_voice = []
        
        # dises = []
        # for voice_pos, voice in zip(voice_positions, voices):
        #     dis = np.linalg.norm(voice_pos - mic_center)
        #     dises.append(dis)
        near_id = 0 #np.argmin(dises)
        
        idx = 0
        gt = utils.read_audio_file_torch(os.path.join(curr_dir, mics[0] + '_voice00.wav'), self.downsample)
        target_voice_inside += gt 

        if self.sig_len < mixture.shape[-1]:
            delta_len = mixture.shape[-1] - self.sig_len
            begin_idx = np.random.randint(low = 1000, high = delta_len - 1)
            mixture = mixture[..., begin_idx:begin_idx+self.sig_len]
            target_voice_inside = target_voice_inside[..., begin_idx:begin_idx+self.sig_len]
        
        # print(mixture.shape, target_voice_inside.shape)
        return metadata, mixture, target_voice_inside, target_voice_outside
