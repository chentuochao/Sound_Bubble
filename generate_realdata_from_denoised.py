import numpy as np 
import json 
import os
import pandas as pd
import soundfile as sf
import argparse
import random
import multiprocessing as mp
import tqdm
from src.metrics.metrics import Metrics
import glob
import src.utils as utils

import time


EPS=1e-9
REFERENCE_CHANNEL = 0
EXCLUDED_DISTANCES = []
FAR_SCALE_MIN = 1
FAR_SCALE_MAX = 2

def handle_error(err):
    print("ERROR!", err)

def snr_at_reference(mix, gt_list, reference_channel):
    from src.metrics.metrics import Metrics
    snr_metric = Metrics('snr')
    mix = mix[reference_channel:reference_channel+1].copy()

    gt = None
    for audio in gt_list:
        if gt is None:
            gt = audio[reference_channel:reference_channel+1].copy()
        else:
            gt += audio[reference_channel:reference_channel+1].copy()
    
    snr = snr_metric(est=mix, gt=gt, mix=mix).item()
    return snr

def rescale_mixture_to_target_snr(near_audio: list, far_audio: list, near_audio_gt: list,
                                  target_snr: float, eps: float = 1e-9):
    """
    Scales background noise in the mixture to achieve a target SNR
    SNR is computed along reference channel
    target_snr is in dB

    Mixture = far + near
    Target = near_hat (isolated signal after denoising)
    Noise = Mixture - Target = far + near - near_hat
    SNR = 10 log10(near_hat^2 / (far + near - near_hat)^2)    
    """
    ref_id = REFERENCE_CHANNEL
    
    # Get signal
    near_hat = np.zeros((near_audio_gt[0].shape[-1], ))
    for audio in near_audio_gt:
        near_hat += audio[ref_id]

    # Get noisy near signal
    near = np.zeros_like(near_hat)
    for audio in near_audio:
        near += audio[ref_id]

    # Get noisy far signal
    far = np.zeros_like(near_hat)
    for audio in far_audio:
        far += audio[ref_id]

    # # Get noise
    # E = M - S

    # # Calculate noise power
    # N_pwr = np.sum(N ** 2)
    # N_pwr_dB = 10 * np.log10(N_pwr + eps)
        
    # Calculate the signal power (or equivalently, energy)
    near_hat_pwr = np.sum(near_hat ** 2)
    near_hat_pwr_dB = 10 * np.log10(near_hat_pwr + eps)

    # Calculate the required power
    target_noise_pwr_dB = (near_hat_pwr_dB - target_snr)
    target_noise_pwr = 10 ** (target_noise_pwr_dB / 10)

    # Calculate signal from isolated signal to denoised target signal
    near_error = (near - near_hat)

    # Given a scale factor k for far speakers
    # Denominator becomes k^2 power(far) + 2k F^T (near_error) + power(near_error) = target_noise_pwr
    # Solve quadratic to find k
    near_error_pwr = np.sum(near_error ** 2)
    far_pwr = np.sum(far ** 2)
    cross_pwr = far.dot(near_error)
    
    a = far_pwr
    b = 2 * cross_pwr

    # This should be non-positive to guarantee a real solution (i.e. denoising doesn't already reduce input SNR below target SNR)
    if target_noise_pwr < near_error_pwr:
        target_noise_pwr = near_error_pwr
        target_noise_pwr_dB = 10 * np.log10(target_noise_pwr)
        adjusted_target_snr = near_hat_pwr_dB - target_noise_pwr_dB
        print(f"[WARNING] target SNR {target_snr}dB cannot be physically realized due to denoising, clipping to {adjusted_target_snr}dB")
        target_snr = adjusted_target_snr
    
    c = near_error_pwr - target_noise_pwr

    possible_scale_factors = np.roots([a, b, c])

    assert not np.iscomplex(possible_scale_factors[0]), "Scale factors are complex, target SNR is too high (denoising is too much)"
    possible_scale_factors = sorted(possible_scale_factors)
    far_scale = possible_scale_factors[-1] # Always take positive root

    # Scale far audio
    far_audio = [far_scale * audio for audio in far_audio]

    # Get final mixture (i.e. check if it's correct)
    gt = np.zeros_like(near_hat)
    for audio in near_audio_gt:
        gt += audio[ref_id]
    
    mixture = np.zeros_like(near_hat)
    for audio in near_audio + far_audio:
        mixture += audio[ref_id]

    snr = 10 * np.log10( (np.sum(gt**2) + eps) / (np.sum((mixture - gt) ** 2) + eps) )

    # utils.write_audio_file('tests/mix.wav', mixture, 24000)
    # utils.write_audio_file('tests/gt.wav', gt, 24000)
    
    assert abs(snr - target_snr) < 1e-3, f"SNR is {snr}, should be {target_snr}"

    return target_snr, far_audio

def random_trim_voices_omni(audio_content_noisy_path, audio_content_denoised_path, 
                                start_frame, end_frame, num_samples, args, is_near = False, 
                                random_state: np.random.RandomState=None):
    if random_state is None:
            random_state = np.random.RandomState()
        
    # Initialize soundfiles
    noisy_file = sf.SoundFile(audio_content_noisy_path)
    denoised_file = sf.SoundFile(audio_content_denoised_path)
    if end_frame is None:
        end_frame = min([noisy_file.frames, denoised_file.frames]) - num_samples
    
    # Randomly sample start frame
    begin_frame = np.random.randint(start_frame, end_frame)
    
    # Seek to start frame
    noisy_file.seek(begin_frame)
    denoised_file.seek(begin_frame)
    
    # Trim omni raw and gt audio
    voice_omni_mix = noisy_file.read(frames=num_samples, dtype='float32').T
    voice_omni_gt = denoised_file.read(frames=num_samples, dtype='float32').T

        
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
        scale = random_state.uniform(FAR_SCALE_MIN, FAR_SCALE_MAX)

    # Apply scaling
    voice_omni_mix *= scale
    voice_omni_gt *= scale

    return voice_omni_mix, voice_omni_gt

def write_data(room_name: str,
               output_dir: str,
               distances: list,
               angles,
               heights,
               speaker_ids,
               mixture,
               gt,
               target_snr,
               snr_clipped,
               args: argparse.Namespace):
    metadata = dict()

    for vidx in range(len(distances)):
        D = distances[vidx]
        angle = 90 - angles[vidx] # 0 degrees is when speaker is facing head
        
        pos = np.array([np.cos(angle), np.sin(angle), 0]) * D
        pos[2] = heights[vidx]

        spk_info = dict(dis=distances[vidx],
                        angle=angle,
                        speaker_id=speaker_ids[vidx],
                        position=pos.tolist())
        metadata[f'voice{vidx:02d}'] = spk_info

    for midx in range(mixture.shape[0]):
        mic_info = dict(position=[0, 0, 0])
        metadata[f'mic{midx:02d}'] = mic_info

    metadata['real'] = True
    metadata['room'] = room_name
    metadata['input_snr'] = target_snr
    metadata['snr_clipped'] = int(snr_clipped)

    os.makedirs(output_dir)
    
    # Write audio
    for vidx in range(len(gt)):
        audio = gt[vidx]
        
        # Write each channel into a separate file
        for midx in range(gt[vidx].shape[0]):
            
            # Store reference channels or all channels
            if (args.reference_channels_only and midx == REFERENCE_CHANNEL) or \
                (not args.reference_channels_only):
                gt_path = os.path.join(output_dir, f'mic{midx:02d}_voice{vidx:02d}.wav')
                utils.write_audio_file(gt_path, audio[midx], args.sr)

    # Write mixture
    mixture_path = os.path.join(output_dir, f'mixture.wav')
    utils.write_audio_file(mixture_path, mixture, args.sr)

    # Write metadata
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)

def create_and_write_mixture(curr_dir, room, room_dir, split,
                             metadata_at_distance, near_distances,
                             far_distances, noise_distances, args):
    # initialize_denoiser()
    
    # Assign different random seed to this directory
    rng = random.Random(curr_dir + args.seed)
    np_rng = np.random.RandomState(curr_dir + args.seed)
    
    # Choose number of near speakers
    n_speakers_near = rng.randint(args.near_speakers_min, args.near_speakers_max)

    # Choose near speaker(s)
    near = rng.choices(near_distances, k=n_speakers_near)

    # Choose number of far speakers
    n_speakers_far = rng.randint(args.far_speakers_min, args.far_speakers_max)

    # Choose far speaker(s)
    far = rng.choices(far_distances, k=n_speakers_far)

    # Choose number of noise sources (only for human samples)
    n_noises = rng.randint(args.noise_sources_min, args.noise_sources_max)

    # Choose noise source(s)
    noises = rng.choices(noise_distances, k=n_noises)

    n_speakers = n_noises + n_speakers_near + n_speakers_far
    
    speaker_ids = []
    heights = []
    angles = []

    near_list_omni_noisy = []
    far_list_omni_noisy = []

    near_list_omni_denoised = []
    far_list_omni_denoised = []
    
    distance_combination = near + far + noises
    
    # Go over all speakers
    for idx, d in enumerate(distance_combination):
        # Bool that tells whether this is a near or a far speaker
        is_near = (d in near)

        # Bool that tells us whether this is a noise source
        is_noise = idx >= (len(near) + len(far))

        if is_near:
            assert (d not in far) and (d not in noises), "Speaker cannot be considered near and far!"
        else:
            assert (d in far) or (d in noises), "Speaker is neither near or far!"

        config_name, metadata = metadata_at_distance[d]
        num_recordings = metadata.shape[0]
        
        chosen_recording = rng.randint(0, num_recordings-1)
    
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

        # Read recordings
        num_samples = int(round(args.duration * args.sr))
        
        start_frame = 0
        end_frame = None
        
        if is_noise:
            omni_noisy_recording_path = \
                os.path.join(room_dir, config_name, f'speaker{chosen_recording:02d}_omni_noise_noisy.wav')
            omni_denoised_recording_path = \
                os.path.join(room_dir, config_name, f'speaker{chosen_recording:02d}_omni_noise.wav')
        else:
            omni_noisy_recording_path = \
                os.path.join(room_dir, config_name, f'speaker{chosen_recording:02d}_omni_noisy.wav')
            omni_denoised_recording_path = \
                os.path.join(room_dir, config_name, f'speaker{chosen_recording:02d}_omni.wav')
        
        # Trim audio to get required number of seconds of audio
        omni_mix, omni_denoised = \
                random_trim_voices_omni(omni_noisy_recording_path,
                                        omni_denoised_recording_path,
                                        start_frame, end_frame, num_samples, args,
                                        is_near=is_near, random_state=np_rng)
        
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

    assert len(near_list_omni_denoised) >= args.near_speakers_min and \
           len(near_list_omni_denoised) <= args.near_speakers_max, f"Number of GT speakers is \
           {len(near_list_omni_denoised)}. Expected a number between {args.near_speakers_min} \
           and {args.near_speakers_max}."
    
    # Create lists of audio that will be added to mixture
    near_list_omni = []
    far_list_omni = []

    num_audio = len(near_list_omni_denoised) + len(far_list_omni_denoised)
    noisy_audio_idx = rng.randint(0, num_audio - 1)
    
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

    # If there is at least one target speaker, scale noise to get target SNR
    if len(near_list_omni_denoised) > 0:
        # Sample target SNR 

        # target_snr = np_rng.normal(loc=args.target_snr_mean, scale=args.target_snr_std)
        if split == 'train':
            low = args.train_target_snr_min
            high = args.train_target_snr_max
        elif split in ['val', 'test']:
            low = args.test_target_snr_min
            high = args.test_target_snr_max
        else:
            assert 0, "Code should not go here"
        target_snr = np_rng.uniform(low=low, high=high)
        
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

    # If there is at least one target speaker, sanity check SNR after summing
    if len(near_list_omni_denoised) > 0:
        omni_snr = snr_at_reference(mixture_omni, near_list_omni_denoised, reference_channel=REFERENCE_CHANNEL)
        if np.abs(adjusted_target_snr_omni - omni_snr) > 1e-3:
            print(f"Omni SNR {omni_snr} is not equal to target SNR {adjusted_target_snr_omni}")
            print('Num near speakers', len(near_list_omni))
            print('Num far speakers', len(far_list_omni))
            print(heights)
            print(near, far)
            print(room_dir)
            print(angles)
            print(speaker_ids)
            for i, near_audio in enumerate(near_list_omni_denoised):
                utils.write_audio_file(f'debug/near{i}.wav', near_audio, 24000)
    
            for i, far_audio in enumerate(far_list_omni):
                utils.write_audio_file(f'debug/far{i}.wav', far_audio, 24000)
               
            utils.write_audio_file(f'debug/mixture_omni.wav', mixture_omni, 24000)
    
            assert 0

    # Write omni data
    output_sample_dir = os.path.join(args.output_dir, "omni", split, f"{curr_dir:06d}")
    write_data(room_name=room,
                output_dir=output_sample_dir,
                distances=distance_combination,
                angles=angles,
                heights=heights,
                speaker_ids=speaker_ids,
                mixture=mixture_omni,
                gt=near_list_omni_denoised,
                target_snr=adjusted_target_snr_omni,
                snr_clipped=(adjusted_target_snr_omni != target_snr),
                args=args)

def process_dataset(data_dir, split, n_outputs, args: argparse.Namespace):
    num_samples = n_outputs
    
    # Divide number of mixtures evenly across rooms
    rooms = sorted(os.listdir(data_dir))
    rooms = [r for r in rooms if not r.startswith('.')]

    num_rooms = len(rooms)
    mixtures_at_room = np.zeros(num_rooms, dtype=np.uint32)
    
    # Assign samples to human rooms
    if len(rooms) > 0:
        mixtures_at_room += num_samples // num_rooms
        remainder = num_samples % num_rooms
        if remainder > 0:
            mixtures_at_room[:remainder] += 1

    # # Assign samples to mannequin rooms
    # if len(mannequin_rooms) > 0:
    #     mixtures_at_room[-len(mannequin_rooms):] += num_mannequin_samples // len(mannequin_rooms)
    #     remainder = num_mannequin_samples % len(mannequin_rooms)
    #     if remainder > 0:
    #         mixtures_at_room[-len(mannequin_rooms):-len(mannequin_rooms) + remainder] += 1

    # Verify room distribution
    assert np.sum(mixtures_at_room) == n_outputs, \
        "Something went wrong trying to distribute mixture between configurations"
    # assert np.sum(mixtures_at_room[:len(human_rooms)]) == num_human_samples, \
    #     "Something went wrong trying to distribute mixture between configurations"
    # assert np.sum(mixtures_at_room[len(human_rooms):]) == num_mannequin_samples, \
    #     "Something went wrong trying to distribute mixture between configurations"

    # Shuffle order
    directory_order = np.arange(n_outputs)
    np.random.shuffle(directory_order)

    current_iteration = 0

    # Go over each room
    for room_idx, room in enumerate(rooms):
        room_dir = os.path.join(data_dir, room)
        
        distance_config = os.listdir(room_dir)

        # Filter out weird files that start with '.'. Could be a Mac thing?
        distance_config = [cfg for cfg in distance_config if not cfg.startswith('.')] 
        
        # Read the metadata for every distance and store it in a dict
        metadata_at_distance = {}
        for config in distance_config:
            config_path = os.path.join(room_dir, config)
            metadata_path = os.path.join(config_path, 'metadata.csv')
            
            df = pd.read_csv(metadata_path, header=0)
            distance = df['distance']

            assert distance.max() == distance.min(), "Expected the distances for all samples to be the same." +\
                                                     f"Found max = {distance.max()}, min = {distance.min()}"
            
            distance = int(distance.iloc[0])

            if distance not in EXCLUDED_DISTANCES:
                metadata_at_distance[distance] = (config, df)

        distances = sorted(list(metadata_at_distance.keys()))
        print(distances)
        
        # Split distances into near and far based on cutoff
        near_distances = [d for d in distances if d < args.distance_cutoff]
        assert len(near_distances) > 0, "No speakers inside the given cutoff."\
                                        "Are you sure distance cutoff has the right units?"\
                                        "It should be in centimeters."
        far_distances = [d for d in distances if d > args.distance_cutoff]
        assert len(far_distances) > 0, "No speakers outside the given cutoff."\
                                        "Are you sure distance cutoff has the right units?"\
                                        "It should be in centimeters."

        noise_distances = far_distances
        # Handle exception for one room, where there is no noise at 300cm due to collection error
        if room == 'Tuochao_cse415':
            noise_distances.remove(300)
        
        if args.num_workers > 1:
            pbar = tqdm.tqdm(total=mixtures_at_room[room_idx])
            pool = mp.Pool(args.num_workers)
            callback_fn = lambda _: pbar.update()
        
        # Go over the required number of outputs    
        for idx in range(mixtures_at_room[room_idx]):
            curr_dir = directory_order[current_iteration]
            if args.num_workers == 1:
                # if current_iteration <= 25:
                #      current_iteration += 1
                #      continue
                create_and_write_mixture(curr_dir, room, room_dir, split, metadata_at_distance,
                                         near_distances, far_distances, noise_distances, args)
                print(f"Finished {current_iteration+1}/{n_outputs}")
            else:
                pool.apply_async(create_and_write_mixture,
                                 args=(curr_dir, room, room_dir, split, metadata_at_distance,
                                       near_distances, far_distances, noise_distances, args),
                                 callback=callback_fn,
                                 error_callback=handle_error)
            current_iteration += 1

        if args.num_workers > 1:
            pool.close()
            pool.join()
            pbar.close()

def main(args: argparse.Namespace):
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    for split in ['train', 'val', 'test']:
        n_outputs = getattr(args, f"n_outputs_{split}")
        
        data_dir = os.path.join(args.data_dir, split)

        if n_outputs > 0:
            process_dataset(data_dir, split, n_outputs, args)

    # Print dataset statistics
    snr_metric = Metrics('snr')
    array_types = sorted(os.listdir(args.output_dir))
    
    for array_type in array_types:
        array_type_dir = os.path.join(args.output_dir, array_type)
        for split in sorted(os.listdir(array_type_dir)):
            split_dir = os.path.join(array_type_dir, split)
            
            input_snrs = []
            for sample in sorted(os.listdir(split_dir)):
                sample_dir = os.path.join(split_dir, sample)
                
                mixture_path = os.path.join(sample_dir, 'mixture.wav')
                mixture = utils.read_audio_file(mixture_path, args.sr)
        
                gt = np.zeros((1, mixture.shape[-1]))
                for gt_path in glob.glob(os.path.join(sample_dir, 'mic00_voice*.wav')):
                    if gt_path[:-4].endswith('noisy'):
                        continue
                    audio = utils.read_audio_file(gt_path, args.sr)
                    gt += audio

                # Sanity check input SNR (again)
                with open(os.path.join(sample_dir, 'metadata.json'), 'rb') as f:
                    metadata = json.load(f)
                expected_input_snr = metadata['input_snr']
                
                if np.abs(gt).max() == 0:
                    assert expected_input_snr is None,\
                         f"For examples where there are no target speakers,\
                         expected input SNR should be None, found {expected_input_snr}"
                else:
                    input_snr = snr_metric(est=mixture[0:1], gt=gt, mix=mixture[0:1]).item()
                    input_snrs.append(input_snr)
                    
                    snr_diff = abs(expected_input_snr - input_snr)
                    if snr_diff > 0.01:
                        print('Actual input SNR is not equal to that specified in the metadata')
                        print(f"Expected: {expected_input_snr}, Actual: {input_snr}")
    
            print(f'[{array_type}] [{split}] Mean Input SNR:', np.mean(input_snrs))

    # Store dataset args
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir',
                        type=str,
                        help="Directory to read raw recordings")
    parser.add_argument('output_dir',
                        type=str,
                        help="Directory to write mixtures")
    # Input dataset params
    parser.add_argument('--wham_noise_start',
                        type=float,
                        help="Time at which wham noise starts for far speakers.",
                        default=12)
    
    # Output dataset params     
    parser.add_argument('--distance_cutoff',
                        type=float,
                        help="Distance at which we cut off near from far speakers (in cm)",
                        default=150)

    parser.add_argument('--n_outputs_train', type=int, default=0)
    parser.add_argument('--n_outputs_test', type=int, default=0)
    parser.add_argument('--n_outputs_val', type=int, default=0)

    parser.add_argument('--duration',
                        type=float,
                        help="Duration of the mixtures in seconds",
                        default=5)
    parser.add_argument('--sr',
                        type=int,
                        help="Sampling rate",
                        default = 24000)
    
    parser.add_argument('--far_speakers_min',
                        type=int,
                        help="Minimum number of speakers",
                        default = 1)
    parser.add_argument('--far_speakers_max',
                        type=int,
                        help="Maximum number of speakers",
                        default=2)
    
    parser.add_argument('--near_speakers_min',
                        type=int,
                        help="Minimum number of speakers that should be near the listener",
                        default=1)
    parser.add_argument('--near_speakers_max',
                        type=int,
                        help="Maximum number of speakers that should be near the listener",
                        default=1)
    
    parser.add_argument('--noise_sources_min',
                        type=int,
                        help="Minimum number of noise sources",
                        default=0)
    parser.add_argument('--noise_sources_max',
                        type=int,
                        help="Maximum number of noise sources",
                        default=1)

    parser.add_argument('--reference_channels_only',
                        action='store_true',
                        help="Whether or not to store the refernce channels only")
    
    # Scale parameters (relative scaling between speakers, actual SNR is controlled with SNR params)
    parser.add_argument('--amplitude_scale_min',
                        type=float,
                        help="Normalize speaker audio peak to at least this value",
                        default = 1.0)
    parser.add_argument('--amplitude_scale_max',
                        type=float,
                        help="Normalize speaker audio peak to at most this value",
                        default=2.0)

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

    # Miscellaneous parameters
    parser.add_argument('--seed',
                        type=int,
                        help="Random seed",
                        default=0)
    parser.add_argument('--num_workers',
                        type=int,
                        help="Number of workers in Pool",
                        default=1)

    main(parser.parse_args())

