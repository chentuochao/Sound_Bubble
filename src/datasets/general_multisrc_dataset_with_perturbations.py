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
import glob


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
                dis_threshold = 1.5, directional=True, fair_compare = False,
                prob_neg = 0, # Unused
                perturbations = [],
                downsample = 1, mic_config = [], sig_len = 4.5,
                reference_channels = None,
                split = 'val'):
        super().__init__()
        self.dirs = [] #sorted(list(Path(rw_dir).glob('[0-9]*')))

        for _dir in dataset_dirs:
            dirpath = _dir['path'] # Get path to dataset
            print(dirpath)
            limit = _dir['max_samples'] # Maximum samples to use from this dataset
            samples = sorted(list(Path(dirpath).glob('[0-9]*')))

            assert limit <= len(samples), "Limit is less than the number of samples"

            if limit <= len(samples):
                
                # For training split, randomly shuffle samples
                #if split == 'train':
                #    random.shuffle(samples)
                
                samples = samples[:limit]
                self.dirs.extend(samples)

            #print('Dataset:', dirpath)
            #print('Num samples:', len(samples))
            
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
        self.dis_threshold = dis_threshold
        self.fair_compare = fair_compare
        
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
        
        # Flag to check if sample is real or synthetic
        real = metadata['real']

        # Iterate over different sources
        voices = [key for key in metadata.keys() if 'voice' in key]
        mics = self.mic_lists #[key for key in metadata.keys() if 'mic' in key]
        assert (self.n_mics==len(mics)), 'Number of mics in metadata not equal to mics list in config'
  
        # Read mixture audio
        mixture = utils.read_audio_file_torch(os.path.join(curr_dir, 'mixture.wav'))
        
        # Read ground truth audio files
        distances = []
        num_tgt_speakers = 0
        target_voice_inside = torch.zeros((len(self.reference_mics), mixture.shape[-1]))
        for voice in voices:
            # Read distance in meters
            if real:
                d = int(metadata[voice]['dis']) / 100 
            else:
                d = metadata[voice]['dis']
            
            # If inside the bubble, load and add it to gt
            if d <= self.dis_threshold:
                
                # Add audio to the gt at the right reference channel
                for ch_idx, mic in enumerate(self.reference_mics):
                    audio = utils.read_audio_file_torch(os.path.join(curr_dir, f'{mics[mic]}_{voice}.wav'))
                    target_voice_inside[ch_idx] += audio[0]
                
                num_tgt_speakers += 1
            
            distances.append(d)
        
        # THIS ASSERT CAN BE REMOVED
        # Check to make sure number of target speakers is in line with wavfiles
        assert num_tgt_speakers == len(glob.glob(os.path.join(curr_dir,'*.wav'))) - 1

        # Sanity check audio content
        if num_tgt_speakers == 0:
            assert torch.abs(target_voice_inside).max() == 0, "When there are no inside speakers, the target should be zero"
        else:
            assert torch.abs(target_voice_inside).max() > 0, "When there is at least one speaker, the target should be more than zero"
        
        # Apply perturbations to entire audio
        if self.split == 'train':
            mixture, target_voice_inside = self.perturbations.apply_random_perturbations(mixture, target_voice_inside)
        
        # Scale mixture so that peak is <= 1
        peak = np.abs(mixture).max()
        if peak > 1:
            mixture /= peak
            for i in range(len(target_voice_inside)):
                target_voice_inside[i] = target_voice_inside[i] / peak
        
        # Define inputs
        inputs = {
            'mixture':mixture.float(),
            'reference_channels':self.reference_mics,
            'sample_dir':str(curr_dir)
        }
        
        # Define targets
        num_noises = len([voice for voice in voices if metadata[voice]['speaker_id'] == 'noise'])
        targets = {
            'target':target_voice_inside.float(),
            'num_target_speakers':num_tgt_speakers,
            'num_interfering_speakers':len(voices) - num_tgt_speakers - num_noises,
            'num_noises':num_noises
        }
        
        return inputs, targets
