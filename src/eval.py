from src.metrics.metrics import compute_decay, Metrics
from src.utils import read_audio_file, write_audio_file
import src.utils as utils
import argparse
import os, json, glob
import numpy as np
import torch
import pandas as pd


def load_testcase(sample_dir: str, args):
    # [1] Load metadata
    metadata_path = os.path.join(sample_dir, 'metadata.json')
    with open(metadata_path, 'rb') as f:
        metadata = json.load(f)

    # [2] Load mixture
    mixture_path = os.path.join(sample_dir, 'mixture.wav')
    mixture = read_audio_file(mixture_path, args.sr)

    # [3] Load ground truth from metadata and count the number of speakers
    # in the ground truth
    gt = np.zeros((1, mixture.shape[-1])) # Single channel
    speakers = [key for key in metadata if key.startswith('voice')]

    tgt_speakers = []

    for speaker in speakers:
        if metadata["real"]:
            speaker_distance = metadata[speaker]['dis'] / 100
        else:
            speaker_distance = metadata[speaker]['dis'] 

        # If speaker is within the threshold, add to gt
        if speaker_distance <= args.distance_threshold:

            speaker_solo = read_audio_file(os.path.join(sample_dir, f'mic{0:02d}_{speaker}.wav'), args.sr)
            gt += speaker_solo

            tgt_speakers.append(metadata[speaker])

    # Check to make sure number of target speakers is in line with wavfiles
    assert len(tgt_speakers) == len(glob.glob(os.path.join(sample_dir,'*.wav'))) - 1

    return metadata, mixture, gt, tgt_speakers

def run_testcase(model, mixture: np.ndarray, device, dis_threshold = -1) -> np.ndarray:
    with torch.no_grad():
        # Create tensor and copy it to the device
        mixture = torch.from_numpy(mixture)
        mixture = mixture.to(device)
        
        if dis_threshold == 1:
            # Run inference
            inputs = dict(mixture=mixture.unsqueeze(0), dis_embed = torch.tensor([[0, 0, 1.0]]).to(device)    )
        elif  dis_threshold == 1.5:
            inputs = dict(mixture=mixture.unsqueeze(0), dis_embed = torch.tensor([[0, 1.0, 0]]).to(device)   )
        elif dis_threshold == 2: 
            inputs = dict(mixture=mixture.unsqueeze(0), dis_embed = torch.tensor([[1.0, 0, 0]]).to(device)  )
        else:
            inputs = dict(mixture=mixture.unsqueeze(0))

        outputs = model(inputs)
        output = outputs['output'].squeeze(0)
        
        # Copy to cpu and convert to numpy array
        output = output.cpu().numpy()

        return output

def main(args: argparse.Namespace):
    device = 'cuda' if args.use_cuda else 'cpu'
    
    os.makedirs(args.output_dir, exist_ok=False)

    sample_dirs = sorted(glob.glob(os.path.join(args.test_dir, '*')))
    
    # Load model
    model = utils.load_torch_pretrained(args.run_dir).model
    model = model.to(device)
    model.eval()

    # Initialize metrics
    snr = Metrics('snr')
    snr_i = Metrics('snr_i')
    
    si_snr = Metrics('si_snr')
    si_snr_i = Metrics('si_snr_i')
    
    si_sdr = Metrics('si_sdr')
    si_sdr_i = Metrics('si_sdr_i')

    pesq = Metrics('PESQ')
    stoi = Metrics('STOI')

    records = []

    for sample_dir in sample_dirs:
        sample_name = os.path.basename(sample_dir)
        print(f"Sample: {sample_name}", sample_dir)
        
        # Load data
        metadata, mixture, gt, tgt_speakers = load_testcase(sample_dir, args)
        n_tgt_speakers = len(tgt_speakers)

        # Run inference
        output = run_testcase(model, mixture, device, args.distance_threshold)
        
        row = {}
        
        # Fill basic information
        row['sample'] = sample_name
        row['room'] = metadata['room']
        row['dis']  = metadata['voice00']['dis']
        row['angle']  = metadata['voice00']['angle']
        row['tgt_speaker_ids'] = [spk['speaker_id'] for spk in tgt_speakers]
        row['tgt_speaker_distances'] = [spk['dis'] for spk in tgt_speakers]
        row['n_tgt_speakers'] = n_tgt_speakers
        row['snr_clipped'] = metadata['snr_clipped']
        
        #print('Num speakers:', n_tgt_speakers)
        if n_tgt_speakers == 0:
            # Compute decay for the case where there is no target speaker
            row['decay'] = compute_decay(est=output, mix=mixture[0:1]).item()
        else:
            # Compute SNR-based metrics for the case where there is at least one target speaker                       
            
            # Input SNR & SNR
            row['input_snr'] = snr(est=mixture[0:1], gt=gt, mix=mixture[0:1]).item()
            row['snri'] = snr_i(est=output, gt=gt, mix=mixture[0:1]).item()

            # Output SI-SNR & SI-SNRi
            row['input_sisnr'] = si_snr(est=mixture[0:1], gt=gt, mix=mixture[0:1]).item()
            row['sisnri'] = si_snr_i(est=output, gt=gt, mix=mixture[0:1]).item()
            
            # Input SI-SDR & SI-SDRi
            row['input_sisdr'] = si_sdr(est=mixture[0:1], gt=gt, mix=mixture[0:1]).item()
            row['sisdri'] = si_sdr_i(est=output, gt=gt, mix=mixture[0:1]).item()

            # subjective
            row['input_stoi'] = stoi(est=mixture[0:1], gt=gt, mix=mixture[0:1]).item()
            row['output_stoi'] = stoi(est=output, gt=gt, mix=mixture[0:1]).item()

            row['input_pesq'] = pesq(est=mixture[0:1], gt=gt, mix=mixture[0:1]).item()
            row['output_pesq'] = pesq(est=output, gt=gt, mix=mixture[0:1]).item()
        
        records.append(row)

    # Create DataFrame from records
    results_df = pd.DataFrame.from_records(records)
    
    # Save DataFrame
    results_csv_path = os.path.join(args.output_dir, 'results.csv')
    results_df.to_csv(results_csv_path)

    # Save arguments to this script
    args_path = os.path.join(args.output_dir, 'args.json')
    with open(args_path, 'w') as f:
        json.dump(args.__dict__, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('test_dir',
                        type=str,
                        help="Path to test dataset")
    parser.add_argument('run_dir',
                        type=str,
                        help='Path to model run')
    parser.add_argument('output_dir',
                        type=str,
                        help='Path to store output files')
    
    parser.add_argument('--distance_threshold',
                        type=float,
                        default=1.5,
                        help='Distance threshold to include/exclude speakers')
    parser.add_argument('--sr',
                        type=int,
                        default=24000,
                        help='Project sampling rate')

    parser.add_argument('--use_cuda',
                        action='store_true',
                        help='Whether to use cuda')

    main(parser.parse_args())




