"""A collection of useful helper functions"""

from random import sample
import numpy as np
import torch
import torchaudio
import os
import json
import librosa
from scipy.io.wavfile import write as wavwrite

from pathlib import Path

from torch.nn.functional import pad

from helpers.constants import EPSILON, SPEED_OF_SOUND, ALL_RADIUS_SIZES

from scipy.fftpack import fftshift

from typing import Dict, Tuple, List
import noisereduce as nr
import random


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def read_json(path):
    assert os.path.exists(path), f'File not found: {path}'
    
    with open(path, 'rb') as f:
        return json.load(f)
    
def write_json(dict, path):    
    with open(path, 'w') as f:
        json.dump(dict, f, indent=4)

def denoise(signal, noise_sample, sr, stationary=False, n_jobs=1):
    return nr.reduce_noise(y=signal, sr=sr, y_noise=noise_sample, stationary=stationary, n_jobs=n_jobs)

def gcc_phat(sig, refsig, target_radius = ALL_RADIUS_SIZES[0]):
    '''
    This function computes the offset between the signal sig and the reference signal refsig
    using the Generalized Cross Correlation - Phase Transform (GCC-PHAT)method.
    '''
    
    # make sure the length for the FFT is larger or equal than len(sig) + len(refsig)
    n = sig.shape[0] + refsig.shape[0]

    # Generalized Cross Correlation Phase Transform
    SIG = np.fft.rfft(sig)
    REFSIG = np.fft.rfft(refsig)
    R = SIG * np.conj(REFSIG)

    cc = np.fft.irfft(R / (np.abs(R) + 1e-6))

    cc = fftshift(cc)
    zero = cc.shape[-1]//2

    max_samples = int(round(1.3 * target_radius * 16000 / 343))
    # print(max_samples)
    cc[0:zero-max_samples] = 0
    cc[zero+max_samples:] = 0

    shift = np.argmax(np.abs(cc)) - zero
    # print(shift)

    return shift, cc

def closest_mic(mic_positions, target_pos):
    return np.argmin(np.linalg.norm(mic_positions - target_pos, axis=1))

def gcc_adjust_shifts(shifted, mic_positions, target_pos, target_radius):
    shifts = []
    k = closest_mic(mic_positions, target_pos)
    for i in range(shifted.shape[0]):
        shift, _ = gcc_phat(shifted[i], shifted[k], target_radius)
        shifted[i] = shift_fn(shifted[i], -shift)
        shifts.append(shift)
    return shifted, shifts

def list_top_level_directories(path) -> List[str]:
    return [a for a in os.listdir(path) if os.path.isdir(os.path.join(path, a))]

def read_metadata(dir_path) -> dict:
    metadata_path = os.path.join(dir_path, 'metadata.json')
    with open(metadata_path, 'r') as json_file:
        metadata = json.load(json_file)
    assert metadata, 'Something went wrong when reading scene metadata. Are you sure this file exists in the specified directory?'
    return metadata

def read_audio_file(file_path, sr):
    """
    Reads audio file to system memory.
    """
    return librosa.core.load(file_path, mono=False, sr=sr)[0]

def read_audio_file_torch(file_path, downsample=1):
    waveform, sample_rate = torchaudio.load(file_path)
    if downsample >1:
        waveform = torchaudio.functional.resample(waveform, sample_rate, sample_rate//downsample)
    return waveform




def save_audio_file_torch(file_path, wavform, sample_rate = 48000):
    wavform = wavform/torch.max(wavform)*0.9
    torchaudio.save(file_path, wavform, sample_rate)



import soundfile
def write_audio_file(file_path, data, sr):
    """
    Writes audio file to system memory.
    @param file_path: Path of the file to write to
    @param data: Audio signal to write (n_channels x n_samples)
    @param sr: Sampling rate
    """
    #wavwrite(file_path, sr, data.T)
    soundfile.write(file_path, data, sr, subtype = "FLOAT")

def read_input_dir(input_dir, sr) -> Tuple[Dict, np.ndarray]:
    metadata = read_metadata(input_dir)
    n_mics = len(metadata['mic'])
    
    mic_mixture_file_path = [os.path.join(input_dir,f'mixture_mic{i:02d}.wav') for i in range(n_mics)]

    mixture = np.stack([read_audio_file(path, sr) for path in mic_mixture_file_path])

    return metadata, torch.from_numpy(mixture)

def optimal_channel_reordering(mixed_data, target_position, microphone_positions):
    """
    Reorders channels based on their distances to target postion
    """
    distances_idx = np.argsort(np.linalg.norm(microphone_positions - target_position, axis = 1))
    processed_data = np.zeros_like(mixed_data)
    new_microphone_positions = np.zeros_like(microphone_positions)
    for i in range(mixed_data.shape[0]):
        processed_data[i] = mixed_data[distances_idx[i]]
        new_microphone_positions[i] = microphone_positions[distances_idx[i]]
    
    return new_microphone_positions, processed_data

def phase_offset(a, b, sr):
    if len(a.shape) == 1:
        return (np.linalg.norm(b-a, axis=0) * sr/SPEED_OF_SOUND)
        # return (np.sum(((b-a) ** 2 + 0.42 ** 2)) ** 0.5) * sr/SPEED_OF_SOUND
    else:
        return (np.linalg.norm(b-a, axis=1) * sr/SPEED_OF_SOUND)

def criterion(m, s, sr):
    """
    Shift audio from channel at m to appear as though it started at s.
    """
    samples = -phase_offset(m, s, sr)
    return int(round(samples))

def pad_audio(x, padding):
    # Check if numpy or torch
    if isinstance(x, np.ndarray):
        # shift_fn = np.roll
        return np.pad(x, padding)
    elif isinstance(x, torch.Tensor):
        # shift_fn = torch.roll
        return torch.nn.functional.pad(x, padding)
    else:
        raise TypeError("Unknown input data type: {}".format(type(x)))

# def shift_fn(x, shift_samples):
#     if shift_samples > 0:
#         return pad_audio(x, (shift_samples, 0))[:-shift_samples]
#     elif shift_samples < 0:
#         return pad_audio(x, (0, -shift_samples))[-shift_samples:]
#     else:
#         return x

def shift_fn(x, shift_samples):
    if isinstance(x, np.ndarray):
        return np.roll(x, shift_samples)
    elif isinstance(x, torch.Tensor):
        return torch.roll(x, shift_samples)
    else:
        return x

def shift_mixture_given_samples(input_data, shifts, inverse=False):
    """
    Shifts the input given  a vector fo sample shifts
    """
    output_data = input_data * 0
    num_channels = input_data.shape[0]

    # Shift each channel of the mixture to align with mic0
    for channel_idx in range(num_channels):
        shift_samples = shifts[channel_idx]

        if np.abs(shift_samples) > input_data.shape[1]:
            shift_samples = input_data.shape[1]
            shifts[channel_idx] = shift_samples
            output_data[channel_idx] *= 0
            continue
        
        if inverse:
            shift_samples *= -1
        output_data[channel_idx] = shift_fn(input_data[channel_idx], shift_samples)
        shifts[channel_idx] = shift_samples

    return output_data, shifts


def shift_mixture(input_data, target_position, mic_positions, sr, reference_channel=0, inverse=False):
    """
    Shifts the input according to the voice position
    """
    output_data = input_data * 0
    num_channels = input_data.shape[0]
    shifts = np.zeros(num_channels)

    # Shift each channel of the mixture to align with mic0
    for channel_idx in range(num_channels):
        shift_samples = criterion(mic_positions[channel_idx], target_position, sr)

        if np.abs(shift_samples) > input_data.shape[1]:
            shift_samples = input_data.shape[1]
            shifts[channel_idx] = shift_samples
            output_data[channel_idx] *= 0
            continue
        
        if inverse:
            shift_samples *= -1

        output_data[channel_idx] = shift_fn(input_data[channel_idx], shift_samples)
        shifts[channel_idx] = shift_samples

    return output_data, shifts

def get_shift_vector(target_position, mic_positions, sr, reference_channel=0, inverse=False):
    vec = []
    
    for channel_idx in range(mic_positions.shape[0]):
        shift_samples = criterion(mic_positions[channel_idx], target_position, sr) - criterion(mic_positions[reference_channel], target_position, sr)
        vec.append(shift_samples)
    
    return np.array(vec)

def shift_mixture2(input_data, target_position, mic_positions, sr, reference_channel=0, inverse=False):
    """
    Shifts the input according to the voice position relative to a reference channel
    """
    output_data = input_data * 0
    num_channels = input_data.shape[0]
    shifts = np.zeros(num_channels)

    # Shift each channel of the mixture to align with mic0
    for channel_idx in range(num_channels):
        shift_samples = criterion(mic_positions[channel_idx], target_position, sr) - criterion(mic_positions[reference_channel], target_position, sr)

        if np.abs(shift_samples) > input_data.shape[1]:
            shift_samples = input_data.shape[1]
            shifts[channel_idx] = shift_samples
            output_data[channel_idx] *= 0
            continue
        
        if inverse:
            shift_samples *= -1

        output_data[channel_idx] = shift_fn(input_data[channel_idx], shift_samples)
        shifts[channel_idx] = shift_samples

    return output_data, shifts

def to_categorical(index: int, num_classes: int):
    """Creates a 1-hot encoded np array"""
    data = np.zeros((num_classes))
    data[index] = 1
    return data

# def trim_silence(audio, window_size=22050, cutoff=0.001):
#     """Trims all silence within an audio file"""
#     idx = 0
#     new_audio = []
#     while idx * window_size < audio.shape[1]:
#         segment = audio[:, idx*window_size:(idx+1)*window_size]
#         if segment.std() > cutoff:
#             new_audio.append(segment)
#         idx += 1

    # return np.concatenate(new_audio, axis=1)

def check_valid_dir(dir, requires_n_voices=2):
    """Checks that there is at least n voices"""
    if len(list(Path(dir).glob('*_voice00.wav'))) < 1:
        return False

    if requires_n_voices == 2:
        if len(list(Path(dir).glob('*_voice01.wav'))) < 1:
            return False

    if requires_n_voices == 3:
        if len(list(Path(dir).glob('*_voice02.wav'))) < 1:
            return False

    if requires_n_voices == 4:
        if len(list(Path(dir).glob('*_voice03.wav'))) < 1:
            return False

    if len(list(Path(dir).glob('metadata.json'))) < 1:
        return False

    return True
def get_items(curr_dir: str, args):
    metadata = read_metadata(curr_dir)
    
    voices = [key for key in metadata.keys() if 'voice' in key]
    mics = [key for key in metadata.keys() if 'mic' in key]

    mic_positions = np.array([metadata[key]['position'] for key in mics])

    mixture = []
    gt = []

    for mic in mics:
        # channel = utils.read_audio_file(os.path.join(curr_dir, mic) + '_mixed.wav', self.sr)
        channel = read_audio_file(os.path.join(curr_dir, mic) + '_mixed.wav', args.sr)
        mixture.append(channel)

    for voice in voices:
        channel = read_audio_file(os.path.join(curr_dir, f'mic00_{voice}.wav'), args.sr)
        gt.append(channel)
    
    mixture = np.array(mixture)
    gt = np.array(gt)

    return mixture, gt, mic_positions 
