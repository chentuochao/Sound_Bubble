import argparse

from pathlib import Path

import numpy as np
import tqdm

from scipy.signal import stft, istft

from sep.helpers.eval_utils import compute_sdr
from sep.helpers.utils import get_items


def compute_ibm(gt, mix, alpha, theta=0.5):
    """
    Computes the Ideal Binary Mask SI-SDR
    gt: (n_voices, n_channels, t)
    mix: (n_channels, t)
    """
    n_voices = gt.shape[0]
    nfft = 2048
    eps = np.finfo(np.float).eps

    N = mix.shape[-1] # number of samples
    X = stft(mix, nperseg=nfft)[2]
    (I, F, T) = X.shape # (6, nfft//2 +1, n_frame)

    # perform separation
    estimates = []
    for gt_idx in range(n_voices):
        # compute STFT of target source
        Yj = stft(gt[gt_idx], nperseg=nfft)[2]

        # Create binary Mask
        mask = np.divide(np.abs(Yj)**alpha, (eps + np.abs(X) ** alpha))
        mask[np.where(mask >= theta)] = 1
        mask[np.where(mask < theta)] = 0

        Yj = np.multiply(X, mask)
        target_estimate = istft(Yj)[1][:,:N]

        estimates.append(target_estimate)

    estimates = np.array(estimates) # (nvoice, 6, 6*sr)

    # eval
    eval_mix = np.repeat(mix[np.newaxis, :, :], n_voices, axis=0) # (nvoice, 6, 6*sr)
    eval_gt = gt # (nvoice, 6, 6*sr)

    eval_est = estimates

    input_sdr = compute_sdr(eval_gt[:, 0], eval_mix[:, 0])
    output_sdr = compute_sdr(eval_gt[:, 0], eval_est[:, 0])

    output = np.array([input_sdr, output_sdr]) # (2, nvoice)
    return output

def main(args):
    all_dirs = sorted(list(Path(args.input_dir).glob('[0-9]*')))
    # all_dirs = [x for x in all_dirs if check_valid_dir(x, args.n_voices)]

    all_input_sdr = [0] * len(all_dirs)
    all_output_sdr = [0] * len(all_dirs)

    def evaluate_dir(idx):
        curr_dir = all_dirs[idx]
        # Loads the data
        mixed_data, gt, _ = get_items(curr_dir, args)
        gt = np.expand_dims(gt, 1)
        mixed_data = mixed_data[0:1]
        output = compute_ibm(gt, mixed_data, alpha=args.alpha)
        all_input_sdr[idx] = output[0]
        all_output_sdr[idx] = output[1]

    with tqdm.tqdm(total=len(all_dirs)) as pbar:
        for i in range(len(all_dirs)):
            evaluate_dir(i)
            pbar.update()

    # pool = mp.Pool(args.n_workers)
    # with tqdm.tqdm(total=len(all_dirs)) as pbar:
    #     for i, _ in enumerate(pool.imap_unordered(evaluate_dir, range(len(all_dirs)))):
    #         pbar.update()
    
    # # tqdm.tqdm(pool.imap(evaluate_dir, range(len(all_dirs))), total=len(all_dirs))
    # pool.close()
    # pool.join()

    in_flattened = np.array([x for y in all_input_sdr for x in y])
    out_flattened = np.array([x for y in all_output_sdr for x in y])

    print("Median SI-SDRi: ", np.median(out_flattened - in_flattened))
    
    print("Mean SI-SDRi: ", np.mean(out_flattened - in_flattened))

    np.save("IBM_{}voices_{}kHz.npy".format(args.n_voices, args.sr), np.array([in_flattened, out_flattened]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=str, help="Path to the input dir")
    parser.add_argument('--sr', type=int, default=48000, help="Sampling rate")
    parser.add_argument('--n_channels',
                        type=int,
                        default=2,
                        help="Number of channels")
    parser.add_argument('--n_workers',
                        type=int,
                        default=8,
                        help="Number of parallel workers")
    parser.add_argument('--n_voices',
                        type=int,
                        default=2,
                        help="Number of voices in the dataset")
    parser.add_argument('--alpha',
                        type=int,
                        default=1,
                        help="See the original SigSep code for an explanation")
    args = parser.parse_args()

    main(args)


