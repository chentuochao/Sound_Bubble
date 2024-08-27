"""
Torch dataset object for synthetically rendered
spatial data
"""
import json
import random

from typing import Tuple
from pathlib import Path
import librosa
import torch
import numpy as np
import os
import glob
import pandas as pd
import time
import sep.helpers.utils as utils
from src.datasets.perturbations.audio_perturbations import AudioPerturbations

from generate_realdata_from_denoised import rescale_mixture_to_target_snr, snr_at_reference


WHAM_NOISE_START = 12
REFERENCE_CHANNEL = 0

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
    def __init__(self,
                dataset_dirs,
                iterations_per_epoch,
                n_mics=6, 
                sr=24000,
                dis_threshold = 1.5, 
                prob_neg = 0, # Unused
                perturbations = [],
                downsample = 1,
                mic_config = [],
                near_speakers_min = 0,
                near_speakers_max = 2,
                far_speakers_min = 1,
                far_speakers_max = 3,
                noise_min = 0,
                noise_max = 1,
                signal_duration_s = 5,
                train_target_snr_min = -10,
                train_target_snr_max = 5,
                reference_channels = None,
                split = 'val'):
        super().__init__()
        self.room_descriptors = []

        for _dir in dataset_dirs:
            room_dirs = glob.glob(os.path.join(_dir, '*'))
            
            print('ROOM DIRS', room_dirs)
            room_descriptors = []
            for room_dir in room_dirs:
                distance_config = os.listdir(room_dir)

                # Filter out weird files that start with '.'. Could be a Mac thing?
                distance_config = [cfg for cfg in distance_config if not cfg.startswith('.')] 
                
                print("DISTANCE CONFIG", distance_config)

                room_desc = dict(path=room_dir, near_distances = [], far_distances = [], noise_distances = [])

                # Read the metadata for every distance and store it in a dict
                for config in distance_config:
                    config_path = os.path.join(room_dir, config)
                    metadata_path = os.path.join(config_path, 'metadata.csv')
                    
                    df = pd.read_csv(metadata_path, header=0)
                    distance = df['distance']

                    assert distance.max() == distance.min(), "Expected the distances for all samples to be the same." +\
                                                            f"Found max = {distance.max()}, min = {distance.min()}"
                    
                    distance = int(distance.iloc[0])
                    cfg_descriptor = dict(config_name=config,
                                          distance=distance,
                                          metadata=df)
                    if distance/100 <= dis_threshold:
                        room_desc['near_distances'].append(cfg_descriptor)
                    else:
                        room_desc['far_distances'].append(cfg_descriptor)

                        # Add exception for single room & distance due to human error during data collection
                        room = os.path.basename(room_dir.rstrip('/'))
                        if not (room == 'Tuochao_cse415' and distance == 300):
                            room_desc['noise_distances'].append(cfg_descriptor)
                
                room_descriptors.append(room_desc)
            self.room_descriptors.extend(room_descriptors)

        self.dataset_length = iterations_per_epoch
        self.downsample = downsample
        self.mic_lists = mic_config
        
        if reference_channels is None:
            reference_channels = [0]
        self.reference_mics = reference_channels

        # self.dirs = sorted(list(Path(input_dir).glob('00058')))
        self.valid_dirs = []
        
        # Audio params 
        self.duration = signal_duration_s
        self.train_target_snr_min = train_target_snr_min
        self.train_target_snr_max = train_target_snr_max

        # Physical params
        self.n_mics = n_mics
        self.sr = sr
        self.dis_threshold = dis_threshold
        
        # Data augmentation
        self.perturbations = AudioPerturbations(perturbations)
        
        self.near_speakers_min = near_speakers_min
        self.near_speakers_max = near_speakers_max

        self.far_speakers_min = far_speakers_min
        self.far_speakers_max = far_speakers_max

        self.noises_min = noise_min
        self.noises_max = noise_max
        
        self.split = split
        
    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            mixed_data - M x T
            target_voice_data - M x T
            window_idx_one_hot - 1-D
        """
        np.random.seed(idx + os.getpid() + int(time.time()))
        room = np.random.choice(self.room_descriptors)

        return self.get_mixture_and_gt(room)

    def get_mixture_and_gt(self, curr_dir):
        """
        Given a target position and window size, this function figures out
        the voices inside the region and returns them as GT waveforms
        """

        # Choose number of near speakers
        num_near_speakers = np.random.randint(self.near_speakers_min, self.near_speakers_max + 1)
        # Choose near distances
        near_distances = np.random.choice(curr_dir['near_distances'], size=num_near_speakers, replace=False).tolist()

        # WARNING: setting replace=True may cause the same speaker to occur multiple times in the same mixture
        # This might be ok for far speakers
    
        # Choose number of far speakers
        num_far_speakers = np.random.randint(self.far_speakers_min, self.far_speakers_max + 1)
        # Choose far distances
        far_distances = np.random.choice(curr_dir['far_distances'], size=num_far_speakers, replace=True).tolist()

        # Choose number of noise sources
        num_noise = np.random.randint(self.noises_min, self.noises_max + 1)
        # Choose noise distances
        noise_distances = np.random.choice(curr_dir['noise_distances'], size=num_noise, replace=True).tolist()

        mixture, target_voice_inside = self.create_mixture(curr_dir, near_distances, far_distances, noise_distances)

        mixture = torch.from_numpy(mixture)
        target_voice_inside = torch.from_numpy(target_voice_inside)[REFERENCE_CHANNEL].unsqueeze(0)

        if self.split == 'train':
            mixture, target_voice_inside = self.perturbations.apply_random_perturbations(mixture, target_voice_inside)
        
        # Scale mixture so that peak is <= 1
        peak = np.abs(mixture).max()
        if peak > 1:
            mixture /= peak
            for i in range(len(target_voice_inside)):
                target_voice_inside[i] = target_voice_inside[i] / peak

        inputs = {
            'mixture':mixture.float(),
            'reference_channels':self.reference_mics
        }
        
        targets = {
            'target':target_voice_inside.float(),
            'num_target_speakers':num_near_speakers,
            'num_interfering_speakers':num_far_speakers
        }
        
        return inputs, targets

    def create_mixture(self, room, near, far, noises):
        speaker_ids = []
        heights = []
        angles = []

        far_list_omni_noisy = []
        near_list_omni_noisy = []
        near_list_omni_denoised = []
        far_list_omni_denoised = []

        #print('NEAR TYPE', type(near))    
        #print('FAR TYPE', type(far))    
        #print('NOISE TYPE', type(noises))    
        distance_combination = near + far + noises
        
        # Go over all speakers
        for distance_idx in range(len(distance_combination)):
            d = distance_combination[distance_idx]
            
            # Bool that tells whether this is a near or a far speaker
            is_near = (distance_idx < len(near))

            # Bool that tells us whether this is a noise source
            is_noise = distance_idx >= (len(near) + len(far))

            if is_near:
                assert (d not in far) and (d not in noises), "Speaker cannot be considered near and far!"
            else:
                assert (d in far) or (d in noises), "Speaker is neither near or far!"

            config_name, metadata = d['config_name'], d['metadata']
            num_recordings = metadata.shape[0]
            
            chosen_recording = np.random.randint(0, num_recordings)
        
            # Get recording info from metadata dataframe
            recording_info = metadata.iloc[chosen_recording]
            height = recording_info['height']
            angle = recording_info['angle']
            
            if is_noise:
                speaker_id = 'noise'
            else:
                speaker_id = str(recording_info['speaker_id'])
                if '/' in speaker_id:
                    speaker_id = speaker_id.split('/')[-1]

            # Store info
            heights.append(height)
            angles.append(angle)
            speaker_ids.append(speaker_id)

            # flag = False
            # if (is_noise and os.path.basename(room['path'].rstrip('/')) == 'Tuochao_cse415' and d['distance'] == 300):
            #     flag = True
            # # print(os.path.basename(room['path'].rstrip('/')), d['distance'])
            
            # if flag:
            #     print("BAD ROOM FOR NOISE")
            #     input()

            # omni_noisy_recording_path = os.path.join(room['path'], config_name, f'speaker{chosen_recording:02d}_omni_noisy.wav')
            # omni_denoised_recording_path = os.path.join(room['path'], config_name, f'speaker{chosen_recording:02d}_omni.wav')

            if is_noise:
                omni_noisy_recording_path = \
                    os.path.join(room['path'], config_name, f'speaker{chosen_recording:02d}_omni_noise_noisy.wav')
                omni_denoised_recording_path = \
                    os.path.join(room['path'], config_name, f'speaker{chosen_recording:02d}_omni_noise.wav')
            else:
                omni_noisy_recording_path = \
                    os.path.join(room['path'], config_name, f'speaker{chosen_recording:02d}_omni_noisy.wav')
                omni_denoised_recording_path = \
                    os.path.join(room['path'], config_name, f'speaker{chosen_recording:02d}_omni.wav')
            
            # Read and reorder recordings
            omni = utils.read_audio_file(omni_noisy_recording_path, self.sr)
            omni_denoised = utils.read_audio_file(omni_denoised_recording_path, self.sr)

            # # If this is supposed to be a noise source, skip the first several seconds
            # speech_duration = (WHAM_NOISE_START * self.sr)
            # if is_noise:
            #     omni = omni[:, speech_duration:]
            #     omni_denoised = omni_denoised[:, speech_duration:]
            # else:
            #     omni = omni[:, :speech_duration]
            #     omni_denoised = omni_denoised[:, :speech_duration]

            # if flag:
            #     print("OMNI SHAPE", omni.shape)
            #     print("DENOISED SHAPE", omni_denoised.shape)
            #     input()
            
            # Trim audio to get required number of seconds of audio
            num_samples = int(round(self.duration * self.sr))
            omni_mix, omni_denoised = self.random_trim_voices_omni(omni, omni_denoised, num_samples, is_near=is_near)
            
            # Sanity check for any bugs while preprocessing/trimming
            assert np.abs(omni_denoised).max() > 0, "Denoised audio should not be zero."
            
            # Store gt
            if is_near:    
                near_list_omni_denoised.append(omni_denoised)
            else:
                far_list_omni_denoised.append(omni_denoised)
                
            # Store audio without denoising
            if is_near:
                near_list_omni_noisy.append(omni_mix)
            else:
                far_list_omni_noisy.append(omni_mix)

        assert len(near_list_omni_denoised) >= self.near_speakers_min and \
            len(near_list_omni_denoised) <= self.near_speakers_max, f"Number of GT speakers is \
            {len(near_list_omni_denoised)}. Expected a number between {self.near_speakers_min} \
            and {self.near_speakers_max}."
       
        #
        #print("FINISHED TRIMMING")

        # Create lists of audio that will be added to mixture
        near_list_omni = []
        far_list_omni = []

        num_audio = len(near_list_omni_denoised) + len(far_list_omni_denoised)
        """
        for i in range(len(near_list_omni_denoised)):
            audio = near_list_omni_noisy[i]
            near_list_omni.append(audio.copy())
    
        for i in range(len(far_list_omni_denoised)):
            audio = far_list_omni_noisy[i]
            far_list_omni.append(audio.copy())
        """
        noisy_audio_idx = np.random.randint(0, num_audio)
        
        # """

        # Choose audio from different lists (i.e. sometimes from denoised, sometimes from noisy)
        for i in range(len(near_list_omni_denoised)):
            if i == noisy_audio_idx:
                audio = near_list_omni_noisy[i]
            else:
                audio = near_list_omni_denoised[i]
            near_list_omni.append(audio.copy())

        for i in range(len(far_list_omni_denoised)):
            if i + len(near_list_omni_denoised) == noisy_audio_idx:
                audio = far_list_omni_noisy[i]
            else:
                audio = far_list_omni_denoised[i]
            far_list_omni.append(audio.copy())
        
        # """

        # If there is at least one target speaker, scale noise to get target SNR
        if len(near_list_omni_denoised) > 0:
            # Sample target SNR 

            # target_snr = np_rng.normal(loc=args.target_snr_mean, scale=args.target_snr_std)
            low = self.train_target_snr_min
            high = self.train_target_snr_max
            
            target_snr = np.random.uniform(low=low, high=high)
            
            # Scale to target SNR
            adjusted_target_snr_omni, far_list_omni = \
                rescale_mixture_to_target_snr(near_list_omni, far_list_omni, near_list_omni_denoised, target_snr)
        else:
            target_snr = None
            adjusted_target_snr_omni = None

        # Get mixtures by summing near and far audio
        mixture_omni = None
        for audio in near_list_omni + far_list_omni:
            if mixture_omni is None:
                mixture_omni = audio.copy()
            else:
                mixture_omni += audio
        
        # Renormalize if amplitude > 1
        if np.abs(mixture_omni).max() > 1:
            div = np.abs(mixture_omni).max()
            mixture_omni /= div
            for i in range(len(near_list_omni_denoised)):
                near_list_omni_denoised[i] /= div
                #near_list_omni[i] /= div

        # If there is at least one target speaker, sanity check SNR after summing
        if len(near_list_omni_denoised) > 0:
            omni_snr = snr_at_reference(mixture_omni, near_list_omni_denoised, reference_channel=REFERENCE_CHANNEL)
            assert np.abs(adjusted_target_snr_omni - omni_snr) < 1e-3, "Omni SNR is not equal to target SNR"

        target = np.zeros_like(mixture_omni)
        for audio in near_list_omni_denoised:
            target += audio
        
        return mixture_omni, target

    def random_trim_voices_omni(self, audio_content_omni, audio_content_omni_gt, 
                                num_samples, is_near = False, random_state: np.random.RandomState=None):
        if random_state is None:
            random_state = np.random.RandomState()
        
        # Remove leading and trailing silence
        _, begin_end_omni = librosa.effects.trim(audio_content_omni_gt, top_db=18) #22
        
        begin_idx = max([begin_end_omni[0] - 8000, 0])
        end_idx = min([begin_end_omni[1] + 8000, audio_content_omni_gt.shape[-1]])
        
        # Trim omni raw and gt audio
        voice_omni_mix = audio_content_omni[:, begin_idx:end_idx].copy()
        voice_omni_gt = audio_content_omni_gt[:, begin_idx:end_idx].copy()
            
        # Cut or pad audio to get target length
        remain = num_samples - voice_omni_gt.shape[-1]
        
        # If we need to add extra samples, pad zeros to the left and right
        if remain > 0:
            pad_front = random_state.randint(0, remain)
            
            # Omni
            voice_omni_mix = np.pad(voice_omni_mix, ((0,0), (pad_front, remain - pad_front)))
            voice_omni_gt = np.pad(voice_omni_gt, ((0,0), (pad_front, remain - pad_front)))
        
        # Otherwise, randomly trim some number of samples equal to the target length
        elif remain < 0:
            min_bound = voice_omni_gt.shape[-1] - num_samples
            begin_i = random_state.choice(min_bound)

            # Omni    
            voice_omni_mix = voice_omni_mix[:, begin_i:begin_i+num_samples]
            voice_omni_gt = voice_omni_gt[:, begin_i:begin_i+num_samples]

        scale = 1
        # Randomly scale the amplitude to within some range
        if not is_near:
            scale = random_state.uniform(1, 2)

        # Apply scaling
        # Omni
        voice_omni_mix *= scale
        voice_omni_gt *= scale

        return voice_omni_mix, voice_omni_gt