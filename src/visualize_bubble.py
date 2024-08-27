
from copyreg import pickle
import os
import sys

import argparse
import json
from typing import List
from numpy.random import randint, uniform
from pathlib import Path
import tqdm
import random
import glob

import multiprocessing.dummy as mp

import scipy
import scipy.spatial 
import numpy as np
import librosa
import pyroomacoustics as pra
import soundfile as sf
from scipy.io.wavfile import write as wav_write
import matplotlib.pyplot as plt
import time
import src.utils as utils
import torch
import torch.nn as nn
from asteroid.losses.sdr import SingleSrcNegSDR
import torchaudio

from src.metrics.metrics import compute_decay

# create directivity object
from pyroomacoustics.directivities import (
    DirectivityPattern,
    DirectionVector,
    CardioidFamily,
)

from generate_adaptive_dataset import list_tts_folders, random_select_speaker, FG_VOL_MIN, FG_VOL_MAX


def seed_all(seed):
    np.random.seed(seed)
    random.seed(seed)

def point_in_box(pos, left, right, top, bottom):
    return pos[0] >= left and pos[0] <= right and pos[1] <= top and pos[1] >= bottom


def visualize_config(mic_pos, speaker_pos):
    speaker_pos = np.array(speaker_pos)
    fig, ax = plt.subplots(figsize=(10, 10))

    R_max = 8
    ax.set(xlim=(0, R_max), ylim = (0, R_max))

    for i in range(0, speaker_pos.shape[0]):
        plt.scatter(speaker_pos[i, 0], speaker_pos[i, 1], marker='x', c='k')
        plt.text(speaker_pos[i, 0] + 0.1, speaker_pos[i, 1] + 0.1, "spk " + str(i))


    left = mic_pos[:2, :2]
    right = mic_pos[2:4, :2]

    mic_center = (left[0, :] + right[0, :]) / 2
    circle1 = plt.Circle((mic_center[0], mic_center[1]), 0.55/2, color = 'k', fill=False)
    ax.add_patch(circle1)
    theta = np.arctan2( right[0, 1] - left[0, 1] , right[0, 0] - left[0, 0] )
    R_polar = 0.1
    circle1_1 = plt.Circle((left[0, 0] + R_polar*np.cos(theta), left[0, 1]+ R_polar*np.sin(theta)), R_polar, color = 'r', alpha = 0.5)
    circle1_2 = plt.Circle((left[0, 0] - R_polar*np.cos(theta), left[0, 1]- R_polar*np.sin(theta)), R_polar, color = 'r', alpha = 0.5)
    ax.add_patch(circle1_1)
    ax.add_patch(circle1_2)
    plt.text(left[0, 0] + R_polar*np.cos(theta), left[0, 1]+ R_polar*np.sin(theta), "0")
    circle2_1 = plt.Circle((left[0, 0] + R_polar*np.sin(theta), left[0, 1]- R_polar*np.cos(theta)), R_polar, color = 'b', alpha = 0.5)
    circle2_2 = plt.Circle((left[0, 0] - R_polar*np.sin(theta), left[0, 1] + R_polar*np.cos(theta)), R_polar, color = 'b', alpha = 0.5)
    ax.add_patch(circle2_1)
    ax.add_patch(circle2_2)
    plt.text(left[0, 0] + R_polar*np.sin(theta), left[0, 1]- R_polar*np.cos(theta), "1")


    circle3_1 = plt.Circle((right[0, 0] + R_polar*np.cos(theta), right[0, 1]+ R_polar*np.sin(theta)), R_polar, color = 'r', alpha = 0.5)
    circle3_2 = plt.Circle((right[0, 0] - R_polar*np.cos(theta), right[0, 1]- R_polar*np.sin(theta)), R_polar, color = 'r', alpha = 0.5)
    ax.add_patch(circle3_1)
    ax.add_patch(circle3_2)
    plt.text(right[0, 0] + R_polar*np.cos(theta), right[0, 1] +  R_polar*np.sin(theta), "2")

    circle4_1 = plt.Circle((right[0, 0] + R_polar*np.sin(theta), right[0, 1]- R_polar*np.cos(theta)), R_polar, color = 'b', alpha = 0.5)
    circle4_2 = plt.Circle((right[0, 0] - R_polar*np.sin(theta), right[0, 1] + R_polar*np.cos(theta)), R_polar, color = 'b', alpha = 0.5)
    ax.add_patch(circle4_1)
    ax.add_patch(circle4_2)   
    plt.text(right[0, 0] + R_polar*np.sin(theta), right[0, 1]- R_polar*np.cos(theta), "3")



def generate_data_scenario(mic_positions_omni,
                           voice_positions,
                           voices_data,
                           noise_data,
                           total_samples,
                           corners,
                           args,
                           N_in,
                           absorption,
                           max_order,
                           ceiling=None,
                           shift_perb = 0,
                           scale_perb = 0):
    # [5]
    # FG
    n_omni = mic_positions_omni.shape[0]

    length = np.max(corners[0])
    width = np.max(corners[1])

    room_dims = [length, width]

    if args.dimensions == 3:
        assert ceiling is not None, "If using 3D simulation must pass ceiling height"
        room_dims.append(ceiling)

    room = pra.ShoeBox(p=room_dims,
                        fs=args.sr,
                        max_order=max_order,
                        absorption = absorption)
    mic_fused = mic_positions_omni #np.concatenate((mic_positions_omni, mic_positions), axis = 0)

    mic_dir = []
    for i in range(mic_positions_omni.shape[0]):
        mic_dir.append(
            CardioidFamily(
            orientation=DirectionVector(azimuth=0, colatitude=0, degrees=True),
            pattern_enum=DirectivityPattern.OMNI,)
        )

    # print(mic_fused.shape, len(mic_dir), mic_angles.shape, mic_positions_omni.shape, mic_positions.shape)
    room.add_microphone_array(mic_fused.T, directivity = mic_dir)

   
    
    for voice_idx in range(len(voice_positions)):
        voice_loc = voice_positions[voice_idx]
        room.add_source(voice_loc, 
                        signal=voices_data[voice_idx][0])


    premix_reverb = room.simulate(return_premix=True)
    gt_signals = np.zeros((len(voice_positions), n_omni, total_samples))

    rt60 = np.mean(room.measure_rt60())
    for i in range(len(voice_positions)):
        for j in range(n_omni):
            gt_signals[i][j] = np.pad(premix_reverb[i][j], (0,total_samples))[:total_samples]
    
    shifts = []
    if shift_perb != 0:
        for j in range(n_omni):
            
            random_S = np.random.randint(low = - shift_perb, high = shift_perb + 1)
            shifts.append(random_S)
            gt_signals[:, j] = np.roll(gt_signals[:, j], random_S, axis=-1)

    scales = []
    if scale_perb != 0:
        for j in range(n_omni):
            random_scale = np.random.uniform(low=-scale_perb, high=scale_perb)
            scales.append(10**(random_scale/20))
            gt_signals[:, j] *= (10**(random_scale/20))
        # print(scales)
    ### gt_signals N_voice (= N_in + N_out) x N_mic x T
    #### targeted_voice

    ### inference speech 
    inference_voices = np.sum(gt_signals[N_in:, :, :], axis=0)
    all_mixture = np.sum(gt_signals[:, :, :], axis=0)

   
    MAX_VOL = np.amax(np.abs(all_mixture))
    SCALE = np.random.uniform(0.5, 0.9)
    all_mixture = all_mixture/MAX_VOL*SCALE
    inference_voices = inference_voices/MAX_VOL*SCALE
    gt_signals = gt_signals/MAX_VOL*SCALE

    return all_mixture, gt_signals, rt60
def save_audio_file_torch(file_path, wavform, sample_rate = 48000, rescale = True):
    # if rescale:
    #     wavform = wavform/torch.max(wavform)*0.9
    torchaudio.save(file_path, wavform, sample_rate)


def main(args: argparse.Namespace):
    seed_all(args.seed)
    with open(args.split_path, 'rb') as f:
        split_data = json.load(f)
    use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print("Using device {}".format('cuda' if use_cuda else 'cpu'))  


    ### load model
    # model = utils.load_pretrained(args.run_dir).model
    model = utils.load_torch_pretrained(args.run_dir).model

    model = model.to(device)
    model.eval()


    ## only run on the test dataset
    subdir = "test"
    voices = split_data[subdir]
    voices_list = [os.path.join(args.input_voice_dir, x) for x in voices]
    if args.tts_dir is None:
        tts_list = {}
    else:
        tts_folder = os.path.join(args.tts_dir, "test-clean")
        tts_list = list_tts_folders(tts_folder)
    print("VCTK speaker num = ", len(voices_list), "TTS speaker num = ", len(tts_list.keys()))

    # eVAL Metrics
    sisdsrloss = SingleSrcNegSDR("sisdr")
    snrloss = SingleSrcNegSDR("snr")

    args.total_samples = int(args.duration * args.sr)
    total_samples = args.total_samples

    ## room setup


    ### mic positions 
    center_pos_x = 1
    center_pos_y = 2.5
    MIC_HEIGHT_SIM = 1.5

    HEAD_SIZE =  0.24
    HEAD_HEIGHT = 0.18

    ### mic positions for omni-directional
    mic_positions = np.array([
        np.array([-12.8, -1.5, 0]),
        np.array([-10.2, 0, 11.3]),
        np.array([-3.8, 0, 16.9]),
        np.array([3.8, 0, 16.9]),
        np.array([10.6, 0, 11.7]),
        np.array([13.1, -1.5, 0.7])
    ])

    # mic_positions = mic_positions[:, [1, 0, 2]]
    # mic_positions = np.array([
    #     np.array([-12.8, -1.5, 0]),
    #     np.array([-10.2, 0, 11.3]),
    #     np.array([-3.8, 0, 16.9]),
    #     np.array([3.8, 0, 16.9]),
    #     np.array([10.6, 0, 11.7]),
    #     np.array([13.1, -1.5, 0.7])
    # ])

    mic_positions = mic_positions/100
    mic_center = np.array([center_pos_x, center_pos_y, MIC_HEIGHT_SIM])    
    mic_positions = mic_positions + mic_center

    N_voice_times = 25
    n_voices = 1  

    print(mic_positions)
    near_angles = np.deg2rad([0])
    # dises = [0.5 + 0.1*i for i in range(0, 29)]
    dises = [0.6 + 0.05*i for i in range(0, 28)]
    for i in range(0, 9):
        dises.append(2 + i * 0.1)

    absorption = 0.7 #np.random.uniform(low=0.15, high=0.9)
    max_order = 10 #np.random.randint(low=12, high=72) #args.max_order
    _id = 1
    rt_selections = [[1.0, 0], [0.8, 4], [0.6, 10], [0.4, 20], [0.2, 30], [0.1, 60]]
    
    room_size = [[4, 4, 2], [6, 6, 3],  [8, 8, 4], [10, 10, 5]]

    rts = [0.4, 20] #rt_selections[_id] 
    right, top, ceiling = room_size[_id] # [7, 7, 3.5]
    corners = np.array([[0, 0], [0, top],
                        [right, top], [right, 0]]).T
    rt_selections = [rts]
    # rts = [absorption, max_order]
    # rt_selections = [rts]

    voices_dict = []
    n_in = 1
    for n in range(N_voice_times):
        voices_data = random_select_speaker(voices_list, tts_list, n_voices, args)
        voices_data_new = []
        for i in range(n_voices):
            if i < n_in:
                random_volume = np.random.uniform(FG_VOL_MIN - 0.2, FG_VOL_MAX- 0.2)
            else:
                random_volume = np.random.uniform(FG_VOL_MIN, FG_VOL_MAX)
            sig  = voices_data[i][0] *random_volume
            voices_data_new.append((sig, voices_data[i][1]))
        voices_data = voices_data_new 
        voices_dict.append(voices_data)

    save_folder = "debug/bubble_room/distance" + str(args.dis_threshold)
    os.makedirs(save_folder, exist_ok=True)
    amplitude_perbs = [0, 1, 2, 3]
    shift_perbs = [0, 2, 4, 6]
    shift_perb = 0
    amplitude_perb = 0
    print(corners)
    angle_num = 0   
    for a0 in near_angles:
        pos_list = []
        result_list = []
        rt60_lists = []
        ## put speaker in different distance
        for near_dis in dises:
            voice_pos_near = np.array([mic_center[0] + near_dis*np.cos(a0), mic_center[1] + near_dis*np.sin(a0), MIC_HEIGHT_SIM])
            assert point_in_box(voice_pos_near, 0, right, top, 0)
            voice_pos_near = voice_pos_near[np.newaxis, :]
            
            for rts in rt_selections:
                # print("reverb params: ", rts, amplitude_perb, shift_perb)
                inputs = []
                gts = []
                embeds = []
                rt60s = []
                for n in range(N_voice_times):
                    voices_data = voices_dict[n]
 
                    input_signals, gt_signals, rt60 = generate_data_scenario(
                           mic_positions_omni = mic_positions,
                           voice_positions = voice_pos_near,
                           voices_data = voices_data,
                           noise_data = None,
                           total_samples = total_samples,
                           corners = corners,
                           args = args,
                           N_in = 1,
                           absorption = rts[0],
                           max_order = rts[1],
                           ceiling=ceiling,
                           shift_perb=shift_perb,
                           scale_perb=amplitude_perb)
                    rt60s.append(rt60)
                    data_input = input_signals
                    gt = gt_signals[0, 0:1, :]
                    
                    scale = 1 / np.abs(input_signals).max() * 0.8
                    data_input = data_input * scale
                    gt = gt * scale

                    inputs.append(data_input)
                    gts.append(gt)
                    if args.dis_threshold == 1:
                        embeds.append(np.array([0, 0, 1.0]))
                    elif args.dis_threshold == 1.5:
                        embeds.append(np.array([0, 1.0, 0]))
                    elif args.dis_threshold == 2:
                        embeds.append(np.array([1.0, 0, 0]))

                gts = np.array(gts)
                inputs = np.array(inputs)
                embeds = np.array(embeds)
                #print(embeds)

                with torch.no_grad():
                    gts = torch.from_numpy(gts).float().to(device)
                    inputs = torch.from_numpy(inputs ).float().to(device) #, dtype =torch.float
                    embeds = torch.from_numpy(embeds).float().to(device)
                    # print(mix.shape)
                    inputs_dict = {
                        'mixture': inputs,
                        'dis_embed':embeds
                    }
                    # inputs_dict = {'mixture': inputs}
                    output_signal = model(inputs_dict)
                    output_signal = output_signal['output']
                    # print(output_signal.shape, inputs.shape, gts.shape)
                    decay =  compute_decay(output_signal, inputs)
                    
                print("-"*10)
                print("rt60 = ", np.mean(rt60s))
                print(a0, near_dis, decay.mean().item())
                pos_list.append([a0, near_dis])
                result_list.append(decay.mean().item())
                rt60_lists.append(np.mean(rt60s))
                torch.cuda.empty_cache() 
            # raise KeyboardInterrupt
        # np.savez(save_folder + "/spatial_a" + str(angle_num) + ".npz",
        np.savez(save_folder + "/spatial_room" + str(_id) + ".npz", 
            room_size = np.array([right, top, ceiling]),
            reverb = np.array(rt_selections[0]),
            pos_x = np.array(pos_list),
            result_y = np.array(result_list),
            rt60_lists = np.array(rt60_lists)
        )
        angle_num += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()


    
    parser.add_argument('input_voice_dir',
                        type=str,
                        help="Directory with voice wav files")

    
    parser.add_argument('run_dir',
                        type=str,
                        help='Path to model run')
        
    parser.add_argument('--tts_dir',
                        type=str,
                        default = None,
                        help="Directory with LibriTTS files")

    parser.add_argument('--bg_voice_dir',
                        type=str,
                        default = None,
                        help="Directory with noise wav files")


    parser.add_argument('--split_path2',
                        type=str,
                        default='datasets/WHAM_split.json')

    parser.add_argument('--split_path',
                        type=str,
                        default='datasets/vctk_split.json')
    parser.add_argument('--n_mics', type=int, default=6)

    parser.add_argument('--n_inside', type=int, default=2)
    parser.add_argument('--n_out_min', type=int, default=1)
    parser.add_argument('--n_out_max', type=int, default=3)

    parser.add_argument('--n_outputs_train', type=int, default=12000)
    parser.add_argument('--n_outputs_test', type=int, default=4000)
    parser.add_argument('--n_outputs_val', type=int, default=0)

    parser.add_argument('--n_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--sr', type=int, default=24000)
    parser.add_argument('--start_index', type=int, default=0)

    parser.add_argument('--dimensions', type=int, default=3, choices=(2, 3))

    parser.add_argument('--dereverber', action='store_true', help="whether the ground-truth is reverber or noe reverber")

    parser.add_argument('--use_cuda', dest='use_cuda', action='store_true',
                    help="Whether to use cuda")

    parser.add_argument('--dis_threshold', type=float, default=1.0)
    parser.add_argument('--duration', type=float, default=5)
    main(parser.parse_args())
