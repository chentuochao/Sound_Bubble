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
from helpers.constants import MAX_SHIFT

import multiprocessing.dummy as mp

import scipy
import scipy.spatial 
import numpy as np
import librosa
import pyroomacoustics as pra
import soundfile as sfsf
from scipy.io.wavfile import write as wav_write
import src.utils as utils
import matplotlib.pyplot as plt
import time

# create directivity object
from pyroomacoustics.directivities import (
    DirectivityPattern,
    DirectionVector,
    CardioidFamily,
)


from generate_realdata_from_denoised import rescale_mixture_to_target_snr, snr_at_reference

# Mean and STD of the signal peak

MIC_WALL_SPACING = 0.5
MAX_SPEAKER_HEIGHT = 0.8
MIC_HEIGHT = 1.5

Out_in_space = 0.3
### head should not overlap with each other
MIN_HEAD_DIS = 0.3

def angle_between_2d_vectors(vector1, vector2):
    vector1 = vector1[np.newaxis, :]
    vector2 = vector2[np.newaxis, :]
    if vector1.shape != (1, 2) or vector2.shape != (1, 2):
        raise ValueError("Input vectors must be 2D arrays with shape (1, 2)")

    dot_product = np.dot(vector1, vector2.T)  # Transpose the second vector for dot product
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)

    # Ensure that the magnitude is not zero to avoid division by zero
    if magnitude1 == 0.0 or magnitude2 == 0.0:
        return 0.0  # If either vector has zero magnitude, the angle is 0 degrees

    cosine_similarity = dot_product / (magnitude1 * magnitude2)
    angle_in_radians = np.arccos(cosine_similarity)

    # Convert the angle from radians to degrees
    angle_in_degrees = np.degrees(angle_in_radians)

    return angle_in_degrees

def handle_error(e):
    print("Error happen " + "!"*30)
    print(e)

def list_tts_folders(directory):
    folder_names = {}

    for folder in os.listdir(directory):
        spk_director = os.path.join(directory, folder)
        tmp_list = []
        if not folder.isnumeric():
            continue
        if os.path.isdir(spk_director):
            for folder2 in os.listdir(spk_director):
                section_director = os.path.join(spk_director, folder2)
                if folder2.isnumeric() and os.path.isdir(spk_director):
                    tmp_list.append(section_director)
        if len(tmp_list) > 0:
            folder_names[folder] = tmp_list 
    return folder_names



def visualize_config(mic_pos, speaker_pos, left, right, bottom, top):
    speaker_pos = np.array(speaker_pos)
    fig, ax = plt.subplots()
    ax.set(xlim=(left, right), ylim = (bottom, top))
    plt.scatter(mic_pos[:, 0], mic_pos[:, 1], marker='x', c='g')
    plt.scatter(speaker_pos[:, 0], speaker_pos[:, 1], marker='o', c='b')
    # plt.show()


def get_noise(noise_list, args ):
    audio = []
    total_samples_48 = args.total_samples * 2

    total_length = 0
    noises = []
    while total_length < total_samples_48:
        noise_file = np.random.choice(noise_list)
        noise, _ = librosa.core.load(noise_file, sr=48000, mono=True)
        noise /= np.abs(noise).max()
        total_length += noise.shape[-1]
        noises.append(noise)

    audio = np.concatenate(noises, axis = -1)[:total_samples_48]
    audio = librosa.resample(audio, orig_sr = 48000, target_sr = args.sr)
    
    return audio
    

def get_voices_vctk(voices_list, n_voices, args):
    voice_dirs = np.random.choice(voices_list, n_voices + 5, replace=False) 

    voices_data = []
    # print("-------------------")
    zero_front = np.random.randint(int(4000), int(48000)) 
    total_samples_48 = args.total_samples * 2

    for voice_dir in voice_dirs:
        voice_identity = voice_dir[-4:]

        files_lists = glob.glob(os.path.join(voice_dir, '*.wav'))
        random.shuffle(files_lists)
        zero_front = np.random.randint(4000, 64000) 
        total_length = zero_front
        speech_audio = [np.zeros((zero_front, ))]

        for i in range(0, min([len(files_lists), 3])):
            voice, _ = librosa.core.load(files_lists[i], sr=48000, mono=True)
            voice_trimmed, begin_end = librosa.effects.trim(voice, top_db=18) #22

            if voice_trimmed.std() <= 2e-4 or begin_end[1] - begin_end[0] < 2000:
                print("Invalid get voice and retry")
                continue
            
            begin_idx = max([begin_end[0] - 2000, 0])
            end_idx = min([begin_end[1] + 2000, voice.shape[-1]])
            voice = voice[begin_idx:end_idx]

            pad_s = int(np.random.uniform((0.2*48000), (0.8*48000)))

            speech_audio.append(voice)
            speech_audio.append(np.zeros((pad_s, )))
            total_length += voice.shape[-1] + pad_s
            
            if total_length >= total_samples_48:
                break
        if total_length < total_samples_48:
            audio = np.concatenate(speech_audio, axis = -1)
            audio = np.pad(audio, (0, total_samples_48 - total_length))
        else:
            audio = np.concatenate(speech_audio, axis = -1)[:total_samples_48]

        audio = librosa.resample(audio, orig_sr = 48000, target_sr = args.sr)
        audio /= np.abs(audio).max()

        voices_data.append((audio, voice_identity))

        if len(voices_data) == n_voices:
            return voices_data


def get_voices_tts(voices_list, n_voices, args):
    tts_lists = list(voices_list.keys())
    speakers = np.random.choice(tts_lists, n_voices + 5, replace=False) #random.sample(args.all_voices, n_voices + 5)
    voices_data = []
    # print("-------------------")
    zero_front = np.random.randint(int(4000), int(64000)) 
    total_samples_48 = args.total_samples * 2

    for speaker_id in speakers:
        files_lists = []
        voice_dir = voices_list[speaker_id]
        voice_identity = speaker_id

        for story_id in voice_dir:
            files = glob.glob(os.path.join(story_id, '*.wav'))
            
            files_lists.extend(files)
        random.shuffle(files_lists)

        zero_front = np.random.randint(4000, 48000) 
        total_length = zero_front
        speech_audio = [np.zeros((zero_front, ))]

        for i in range(0, min([len(files_lists), 3])):
            # print(files_lists[i])
            voice, _ = librosa.core.load(files_lists[i], sr=48000, mono=True)
            voice_trimmed, begin_end = librosa.effects.trim(voice, top_db=18) #22

            if voice_trimmed.std() <= 2e-4 or begin_end[1] - begin_end[0] < 2000:
                print("Invalid get voice and retry")
                continue
            
            begin_idx = max([begin_end[0] - 2000, 0])
            end_idx = min([begin_end[1] + 2000, voice.shape[-1]])
            voice = voice[begin_idx:end_idx]

            pad_s = int(np.random.uniform((0.2*48000), (0.8*48000)))

            speech_audio.append(voice)
            speech_audio.append(np.zeros((pad_s, )))
            total_length += voice.shape[-1] + pad_s
            
            if total_length >= total_samples_48:
                break
        if total_length < total_samples_48:
            audio = np.concatenate(speech_audio, axis = -1)
            audio = np.pad(audio, (0, total_samples_48 - total_length))
        else:
            audio = np.concatenate(speech_audio, axis = -1)[:total_samples_48]

        audio = librosa.resample(audio, orig_sr = 48000, target_sr = args.sr)
        audio /= np.abs(audio).max()

        voices_data.append((audio, voice_identity))

        if len(voices_data) == n_voices:
            return voices_data

def random_select_speaker(VCTK_data, TTS_data, n_voices, args):
    vctk_spk_num = len(VCTK_data)
    tts_lists = list(TTS_data.keys())
    tts_spk_num = len(tts_lists)
    if TTS_data is None:
        voices_data = get_voices_vctk(VCTK_data, n_voices, args)
    else:
        vctk_spk_num = len(VCTK_data)
        tts_spk_num = len(TTS_data)
        p_vctk = max([vctk_spk_num/(vctk_spk_num + tts_spk_num), 0.2])
        n_vctk = 0
        n_tts = 0
        
        for i in range(n_voices):
            if np.random.rand() < p_vctk:
                n_vctk += 1
            else:
                n_tts += 1
        voices_data = []
        #print("n_vctk", n_vctk, "n_tts", n_tts)
        if n_vctk > 0:
            voices_data0 = get_voices_vctk(VCTK_data, n_vctk, args)
            for p in voices_data0:
                voices_data.append(p)
        if n_tts > 0:
            voices_data1 = get_voices_tts(TTS_data, n_tts, args)
            for p in voices_data1:
                voices_data.append(p)

    return voices_data
        


def point_in_box(pos, left, right, top, bottom):
    return pos[0] >= left and pos[0] <= right and pos[1] <= top and pos[1] >= bottom



def get_random_mic_positions_headphone(n_mics,left, right, bottom, top):
    assert (n_mics == 6)
    min_x = left + MIC_WALL_SPACING
    max_x = right - MIC_WALL_SPACING
    
    min_y = bottom + MIC_WALL_SPACING
    max_y = top - MIC_WALL_SPACING


    mic_positions = np.zeros((6, 3))

    center_pos_x = ( max_x - min_x )*np.random.random() + min_x
    center_pos_y = ( max_y - min_y )*np.random.random() + min_y
    MIC_HEIGHT_SIM = np.random.uniform(low = MIC_HEIGHT - 0.3, high = MIC_HEIGHT + 0.3)
    mic_center = np.array([center_pos_x, center_pos_y, MIC_HEIGHT_SIM])    

    theta = np.random.uniform(low = - np.pi, high= np.pi)
    theta_deg = np.rad2deg(theta)
    
    mics = np.array([
        np.array([-12.8, -1.5, 0]),
        np.array([-10.2, 0, 11.3]),
        np.array([-3.8, 0, 16.9]),
        np.array([3.8, 0, 16.9]),
        np.array([10.6, 0, 11.7]),
        np.array([13.1, -1.5, 0.7])
    ])

    ### add imperfect mic array noise 
    x_err = np.random.uniform(low = -2, high = 2, size = (6, 1))
    x_err[2] = 0
    x_err[3] = 0
    
    z_err = np.random.uniform(low = -2, high = 2, size = (6, 1))
    z_err[2] = 0
    z_err[3] = 0
    # print(mics.shape, x_err, z_err)
    #mics[:, 0:1] += x_err
    #mics[:, 2:3] += z_err
    ## convert cm to m
    mics = mics/100

    ### rotate the array accordin to head angle
    rpy = [0, 0, theta]
    cr, cp, cy = np.cos(rpy)
    sr, sp, sy = np.sin(rpy)
    Rx = np.array([
        [1, 0, 0],
        [0, cr, -sr],
        [0, sr, cr]
    ]).T
    Ry = np.array([
        [cp, 0, sp],
        [0, 1, 0],
        [-sp, 0, cp]
    ]).T
    Rz = np.array([
        [cy, -sy, 0],
        [sy, cy, 0],
        [0, 0, 1]
    ]).T
    mics = mics @ Rx @ Ry @ Rz + mic_center

    return mic_center,  theta_deg, mics




def choose_point_with_circle_keepout(left, right, down, up, center, R_min, R_max):
    R =  np.random.uniform(low=R_min, high=R_max)
    angle_offset = np.random.uniform(low = 0, high = 1)
    angles = np.deg2rad(np.arange(0,360,1) + angle_offset)
    angel_index = np.arange(0,360,1)
    pos_x = R*np.cos(angles) + center[0]
    pos_y = center[1] + R*np.sin(angles)

    inside = (pos_x > left) & (pos_x < right) & (pos_y > down) & (pos_y < up)

    valid = np.sum(inside)

    if valid == 0:
        print("no radias intersection!")
        return choose_point_with_circle_keepout(left, right, up, down,  center, R_min, R_max)
    
    angel_choice = angel_index[inside]
    a = np.random.choice(angel_choice)
    voice_x = pos_x[a]
    voice_y = pos_y[a]


    return R, np.array([voice_x, voice_y])



def get_random_speaker_positions_dis_uniform(dis_threshold, n_in, n_out, mic_positions, mic_center, left, right, up, down):
    voices = []
    dis = []
    angles = []

    DESK_WALL_SAFE = 0.25
    SPEAK_MINX = left + DESK_WALL_SAFE
    SPEAK_MAXX = right - DESK_WALL_SAFE
    
    SPEAK_MINY = down + DESK_WALL_SAFE
    SPEAK_MAXY = up - DESK_WALL_SAFE

    r1 = np.linalg.norm([SPEAK_MINX - mic_center[0], SPEAK_MINY - mic_center[1]]) 
    r2 = np.linalg.norm([SPEAK_MAXX - mic_center[0], SPEAK_MINY - mic_center[1]]) 
    r3 = np.linalg.norm([SPEAK_MINX - mic_center[0], SPEAK_MAXY - mic_center[1]]) 
    r4 = np.linalg.norm([SPEAK_MAXX - mic_center[0], SPEAK_MAXY - mic_center[1]]) 
    R_max = max([r1,r2,r3,r4]) - 0.2


    for i in range(n_in):
        while True:
            R1_max = min([dis_threshold, R_max - 2])
            R, pos = choose_point_with_circle_keepout(SPEAK_MINX, SPEAK_MAXX, SPEAK_MINY, SPEAK_MAXY, mic_center, MIN_HEAD_DIS, R1_max)
            VALID = True
            for j, pos2 in enumerate(voices):
                R2 = dis[j]
                if np.linalg.norm(pos2 - pos) < 0.5:
                    print("Retrying inside speaker ...")
                    VALID = False
                    break
            if VALID:
                break
        voices.append(pos)
        dis.append(R)

    
    for i in range(n_out):
        while True:
            R2_min = min([dis_threshold + Out_in_space, R_max - 0.5])
            R, pos = choose_point_with_circle_keepout(SPEAK_MINX, SPEAK_MAXX, SPEAK_MINY, SPEAK_MAXY, mic_center, R2_min, R_max)

            VALID = True
            for j, pos2 in enumerate(voices):
                R2 = dis[j]
                if j < n_in: ### not too clise to the inside 
                    if np.linalg.norm(pos2 - pos) < 0.5  or np.abs(R2 - R) < Out_in_space:
                        print("Retrying outside speaker ... because close to inside")
                        VALID = False
                        break
                else:
                    if np.linalg.norm(pos2 - pos) < 0.5:
                        print("Retrying outside speaker ... because close to other outside")
                        VALID = False
                        break
            if VALID:
                break

        voices.append(pos)
        dis.append(R)

    return voices, dis


def generate_data_scenario(mic_positions_omni,
                           voice_positions,
                           voices_data,
                           noise_data,
                           total_samples,
                           corners,
                           args,
                           N_in,
                           N_out,
                           absorption,
                           max_order,
                           ceiling=None):
    # [5]
    # FG
    near_list_omni = []
    far_list_omni = []

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


    ### add microphone array to the simulator 
    mic_dir = []
    for i in range(mic_positions_omni.shape[0]):
        mic_dir.append(
            CardioidFamily(
            orientation=DirectionVector(azimuth=0, colatitude=0, degrees=True),
            pattern_enum=DirectivityPattern.OMNI,)
        )
    room.add_microphone_array(mic_positions_omni.T, directivity = mic_dir)

    for voice_idx in range(len(voice_positions)):
        voice_loc = voice_positions[voice_idx]
        room.add_source(voice_loc, signal=voices_data[voice_idx][0])

    premix_reverb = room.simulate(return_premix=True)
    gt_signals = np.zeros((len(voice_positions), n_omni, total_samples))
    noise_signals = np.zeros((n_omni, total_samples))
    rt60 = np.mean(room.measure_rt60())
    # print(np.mean(rt60), max_order, absorption)
    for i in range(len(voice_positions)):
        for j in range(n_omni):
            gt_signals[i][j] = np.pad(premix_reverb[i][j], (0,total_samples))[:total_samples]
    

    #### targeted_voice
    for i in range(N_in):
        SCALE = np.random.uniform(0.5, 0.9)
        gt = gt_signals[i].copy()
        gt = gt/np.amax(np.abs(gt))*SCALE
        
        near_list_omni.append(gt)
    ### inference speech 
    for i in range(N_in, N_in + N_out):
        SCALE = np.random.uniform(0.5, 0.9)
        gt = gt_signals[i].copy()
        gt = gt/np.amax(np.abs(gt))*SCALE
        far_list_omni.append(gt)

    if noise_data is not None:
        
        ### simulate the far-field background noise
        length2 = np.random.uniform(low=30, high=40)
        width2 = np.random.uniform(low=50, high=60)
        height2 = np.random.uniform(low=10, high=20)
        room_dims2 = [length2, width2, height2]
        room = pra.ShoeBox(p=room_dims2, fs=args.sr, max_order=15, absorption = absorption) 
        ### we set the max order to 15 because for simulation speed consideration 
        room.add_microphone_array(mic_positions_omni.T, directivity = mic_dir)

        bg_x = np.random.uniform(low = 15, high = length2 - 2)
        bg_y = np.random.uniform(low = 20, high = width2 - 2)
        bg_z = np.random.uniform(low = 3, high = height2 - 2)
        bg_loc = [bg_x, bg_y, bg_z]
        room.add_source(bg_loc, signal=noise_data)
        premix_bg = room.simulate(return_premix=True)
        SCALE = np.random.uniform(0.05, 0.4)
        premix_bg = premix_bg[:, :, 2000:]
        premix_bg = premix_bg * SCALE / np.abs(premix_bg).max()

        for i in range(n_omni):
            input_noise =  premix_bg[0, i, :]
            noise_signals[i] = np.pad(input_noise, (0, total_samples))[:total_samples]

        far_list_omni.append(noise_signals)


    return near_list_omni, far_list_omni, rt60

def write_data(room_name: str,
                room_info,
               output_dir: str,
               mic_positions_omni,
               mic_center,
               distances,
               voice_positions,
               voices_data,
               mixture,
               gt_noisy,
               gt,
               target_snr,
               snr_clipped,
               args: argparse.Namespace,
               n_in,
               n_out,
               n_BG):
    metadata = dict()

    #print(distances)
    #print()
    assert(len(distances) == len(voices_data))
    assert(len(distances) == voice_positions.shape[0])

    head_vector = mic_positions_omni[0, :2] - mic_positions_omni[-1, :2]

    for vidx in range(len(distances)):
        D = distances[vidx]
        pos = voice_positions[vidx]
        voice_vector = pos[:2] - mic_center[:2]
        angle = angle_between_2d_vectors(voice_vector, head_vector) - 90 # 0 degrees is when speaker is facing head        
        
        spk_info = dict(dis=distances[vidx],
                        angle=angle[0][0],
                        speaker_id=voices_data[vidx][1],
                        position=pos.tolist())
        metadata[f'voice{vidx:02d}'] = spk_info

    for midx in range(mixture.shape[0]):
        mic_info = dict(position=mic_positions_omni[midx].tolist())
        metadata[f'mic{midx:02d}'] = mic_info


    metadata["n_in"] = n_in
    metadata["n_out"] = n_out
    metadata["n_BG"] = n_BG
    metadata['real'] = False
    metadata['room'] = room_name
    metadata['room_info'] = room_info
    metadata['input_snr'] = target_snr
    metadata['snr_clipped'] = int(snr_clipped)

    os.makedirs(output_dir, exist_ok  = True)
    
    # Write audio
    for vidx in range(len(gt)):
        audio = gt[vidx]
        # Write each channel into a separate file
        for midx in range(gt[vidx].shape[0]):
            
            # Store reference channels or all channels
            if midx == 0:
                gt_path = os.path.join(output_dir, f'mic{midx:02d}_voice{vidx:02d}.wav')
                utils.write_audio_file(gt_path, audio[midx], args.sr)

    # Write mixture
    mixture_path = os.path.join(output_dir, f'mixture.wav')
    utils.write_audio_file(mixture_path, mixture, args.sr)
    #print(metadata)
    # Write metadata
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    


def generate_sample(voices_list: list, tts_list: list, noise_list: list, n_inside: int, args: argparse.Namespace, subdir: str, idx: int) -> int:
    """
    Generate a single sample. Return 0 on success.

    Steps:
    - [1] Load voice
    - [2] Create a scene
    - [3] Render sound
    - [4] Save metadata
    """
    # [1] load voice
    output_prefix_dir = os.path.join(args.output_path, subdir, '{:05d}'.format(idx))
    Path(output_prefix_dir).mkdir(parents=True, exist_ok=True)

    args.total_samples = int(args.duration * args.sr)
    total_samples = args.total_samples
    

    n_out = np.random.randint(args.n_out_min, args.n_out_max+1)

    n_in = n_inside


    # if numpy.random.rand() < 0,5:
    n_BG = np.random.randint(low = 0, high = 2)

    n_voices = n_out + n_in

    #t0 = time.time()
    voices_data = random_select_speaker(voices_list, tts_list, n_voices, args)
    if noise_list is None or n_BG == 0:
        noise_data = None
    else:
        noise_data = get_noise(noise_list, args )

    # Generate room parameters, each scene has a random room and absorption
    LWALL_MIN, LWALL_MAX = 0,0 
    RWALL_MIN, RWALL_MAX = 5, 8 
    BWALL_MIN, BWALL_MAX = 0,0 
    TWALL_MIN, TWALL_MAX = 4, 8 
    
    CEIL_MIN, CEIL_MAX = 2, 4

    left = np.random.uniform(low=LWALL_MIN, high=LWALL_MAX)
    right = np.random.uniform(low=RWALL_MIN, high=RWALL_MAX)
    top = np.random.uniform(low=TWALL_MIN, high=TWALL_MAX)
    bottom = np.random.uniform(low=BWALL_MIN, high=BWALL_MAX)    
    ceiling = np.random.uniform(low=CEIL_MIN, high=CEIL_MAX)


    corners = np.array([[left, bottom], [left, top],
                        [right, top], [right, bottom]]).T

    # [4]
    # Compute mic positions

    mic_center, head_angle, mic_positions= get_random_mic_positions_headphone(
                                             n_mics = args.n_mics,
                                             left=left,
                                             right=right,
                                             bottom=bottom,
                                             top=top)

    voice_positions, dis = get_random_speaker_positions_dis_uniform(args.dis_threshold, n_in, n_out, mic_positions[:, :2], mic_center, left=left, right=right, up=top, down=bottom)

    for i in range(len(voice_positions)):
        height = np.random.uniform(low = MIC_HEIGHT - 0.25, high = MIC_HEIGHT + 0.25)
        voice_positions[i] = voice_positions[i].tolist()
        voice_positions[i].append(height)
    voice_positions = np.array(voice_positions)
    
    # Verify speaker placement
    for i, pos in enumerate(voice_positions):
        assert point_in_box(pos, left, right, top, bottom)
        
        if i < n_in:
            if np.linalg.norm(pos[:2] - mic_center[:2] ) > args.dis_threshold:
                print(f'Source {i} too far!')
                print('Source position', pos)
                print('Mic position', mic_positions[0])
                exit(0)
        else:
            if np.linalg.norm( pos[:2] - mic_center[:2] ) <= args.dis_threshold:
                print(f'Source {i} too close!')
                print('Source position', pos)
                print('Mic position', mic_positions[0])
                exit(0)

    for pos in mic_positions:
        assert point_in_box(pos, left, right, top, bottom)

    absorption = np.random.uniform(low=0.1, high=0.9)
    max_order = np.random.randint(low=10, high=72) #72 #args.max_order

    near_list_omni, far_list_omni, rt60 = generate_data_scenario(N_in = n_in,
                                                        N_out = n_out,
                                                        mic_positions_omni = mic_positions,
                                                        voice_positions=voice_positions,
                                                        voices_data=voices_data,
                                                        noise_data = noise_data,
                                                        total_samples=total_samples,
                                                        corners=corners,
                                                        args=args,
                                                        absorption=absorption,
                                                        max_order = max_order,
                                                        ceiling=ceiling)


    if len(near_list_omni) > 0: ### some speaker inside threshold
        if subdir == 'train':
            low = args.train_target_snr_min
            high = args.train_target_snr_max
        elif subdir in ['val', 'test']:
            low = args.test_target_snr_min
            high = args.test_target_snr_max
        else:
            assert 0, "Code should not go here"
        target_snr = np.random.uniform(low=low, high=high)
        # Scale to target SNR
        adjusted_target_snr_omni, far_list_omni = \
            rescale_mixture_to_target_snr(near_list_omni, far_list_omni, near_list_omni, target_snr)
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
        for i in range(len(near_list_omni)):
            near_list_omni[i] /= div

    # If there is at least one target speaker, sanity check SNR after summing
    if len(near_list_omni) > 0:
        omni_snr = snr_at_reference(mixture_omni, near_list_omni, reference_channel=0)
        if np.abs(adjusted_target_snr_omni - omni_snr) > 1e-3:
            print(f"Omni SNR {omni_snr} is not equal to target SNR {adjusted_target_snr_omni}")
            print('Num near speakers', len(near_list_omni))
            print('Num far speakers', len(far_list_omni))
            for i, near_audio in enumerate(near_list_omni):
                utils.write_audio_file(f'debug/near{i}.wav', near_audio, 24000)
            for i, far_audio in enumerate(far_list_omni):
                utils.write_audio_file(f'debug/far{i}.wav', far_audio, 24000)
            utils.write_audio_file(f'debug/mixture_omni.wav', mixture_omni, 24000)
    
            assert 0

    room_info = {
        "walls": [left, right, top, bottom],
        "absorption": absorption,
        "max_order": max_order,
        "rt60": rt60
    }
    write_data(room_name="Synthetic",
                room_info = room_info,
                output_dir=output_prefix_dir,
                mic_positions_omni = mic_positions,
                mic_center = mic_center,
                distances=dis,
                voice_positions=voice_positions,
                voices_data = voices_data,
                mixture=mixture_omni,
                gt_noisy=near_list_omni,
                gt=near_list_omni,
                target_snr=adjusted_target_snr_omni,
                snr_clipped=(adjusted_target_snr_omni != target_snr),
                args=args,
                n_in = n_in,
                n_out = n_out,
                n_BG = n_BG)

    
def seed_all(seed):
    np.random.seed(seed)
    random.seed(seed)

def main(args: argparse.Namespace):
    seed_all(args.seed)
    
    with open(args.split_path, 'rb') as f:
        split_data = json.load(f)
    with open(args.split_path2, 'rb') as f:
        split_noise = json.load(f)    

    for subdir, voices in split_data.items():
        #if subdir == "train":
        #    continue
        n_outputs = getattr(args, "n_outputs_" + subdir)
        if n_outputs <= 0: 
            continue
        print(subdir)

        voices_list = [os.path.join(args.input_voice_dir, x) for x in voices]

        if args.bg_voice_dir is not None:
            noises = split_noise[subdir]
            noise_list = [os.path.join(args.bg_voice_dir, x) for x in noises]
        else:
            noise_list = None

        tts_list = None
        if args.tts_dir is not None:
            if subdir == "train":
                tts_folder = os.path.join(args.tts_dir, "train-clean-360")
                tts_list = list_tts_folders(tts_folder)
            elif subdir == "test":
                tts_folder = os.path.join(args.tts_dir, "test-clean")
                tts_list = list_tts_folders(tts_folder)
            elif subdir == "val":
                tts_folder = os.path.join(args.tts_dir, "dev-clean")
                tts_list = list_tts_folders(tts_folder)
        if tts_list is not None: 
            print("VCTK speaker num = ", len(voices_list), "TTS speaker num = ", len(tts_list.keys()))

        
        if len(voices_list) == 0:
            raise ValueError("No voice files found")

        
        pbar = tqdm.tqdm(total=n_outputs)
        pool = mp.Pool(args.n_workers)
        callback_fn = lambda _: pbar.update()
        total_num = args.start_index + n_outputs
        for i in range(n_outputs):
            if i + args.start_index < total_num/3:
                n_inside = 0
            elif i + args.start_index < total_num/3*2:
                n_inside = 1
            else:
                n_inside = 2
            #:print(i, total_num, args.n_inside)

            # generate_sample(voices_list, tts_list, noise_list, args, subdir, i+ args.start_index)
            # exit(0)
            pool.apply_async(generate_sample,
                            args=(voices_list, tts_list, noise_list, n_inside, args, subdir, i + args.start_index),
                             callback=callback_fn,
                             error_callback=handle_error)
        pool.close()
        pool.join()
        pbar.close()
    
    with open(os.path.join(args.output_path, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_voice_dir',
                        type=str,
                        help="Directory with voice wav files")

    parser.add_argument('output_path', type=str, help="Output directory to write the synthetic dataset")
    
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

    # parser.add_argument('--n_inside', type=int, default=2)
    parser.add_argument('--n_out_min', type=int, default=1)
    parser.add_argument('--n_out_max', type=int, default=2)

    parser.add_argument('--n_outputs_train', type=int, default=12000)
    parser.add_argument('--n_outputs_test', type=int, default=4000)
    parser.add_argument('--n_outputs_val', type=int, default=0)

    parser.add_argument('--n_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--sr', type=int, default=24000)
    parser.add_argument('--start_index', type=int, default=0)

    parser.add_argument('--dimensions', type=int, default=3, choices=(2, 3))


    # SNR parameters. Target SNR will be uniformly distributed. 
    parser.add_argument('--train_target_snr_min',
                        type=float,
                        help='Smallest snr of the input SNR distribution for training.',
                        default=-10)
    parser.add_argument('--train_target_snr_max',
                        type=float,
                        help='Largest snr of the input SNR distribution for training.',
                        default=5)

    parser.add_argument('--test_target_snr_min',
                        type=float,
                        help='Smallest snr of the input SNR distribution for test/val.',
                        default=-5)
    parser.add_argument('--test_target_snr_max',
                        type=float,
                        help='Largest snr of the input SNR distribution for test/val.',
                        default=5)

    parser.add_argument('--dis_threshold', type=float, default=1.0)
    parser.add_argument('--duration', type=float, default=5)

    main(parser.parse_args())

