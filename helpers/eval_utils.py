import math
import numpy as np

def __snr_base(est: np.ndarray, gt: np.ndarray, scale_invariant: bool = False):
    if scale_invariant:
        # Rescale
        a = np.dot(est, gt) / np.dot(gt, gt)
    else:
        a = 1

    e_signal = a * gt
    e_noise = e_signal - est

    Sss = (e_signal**2).sum()
    Snn = (e_noise**2).sum() + 1e-9

    return 10 * math.log10(Sss/Snn)

def snr(est: np.ndarray, gt: np.ndarray):
    return __snr_base(est, gt, False)

def si_sdr(est: np.ndarray, gt: np.ndarray):
    return __snr_base(est, gt, True)