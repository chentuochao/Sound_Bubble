import math
from pyexpat import model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics.functional import(
    scale_invariant_signal_noise_ratio as si_snr,
    signal_noise_ratio as snr,
    signal_distortion_ratio as sdr,
    scale_invariant_signal_distortion_ratio as si_sdr)
import speechbrain as sb
from speechbrain.lobes.models.dual_path import Encoder
from speechbrain.lobes.models.dual_path import Decoder

from .conv_tasnet import MaskNet, ChannelwiseLayerNorm


def mod_pad(x, chunk_size, pad):
    # Mod pad the input to perform integer number of
    # inferences
    mod = 0
    if (x.shape[-1] % chunk_size) != 0:
        mod = chunk_size - (x.shape[-1] % chunk_size)

    x = F.pad(x, (0, mod))
    x = F.pad(x, pad)

    return x, mod

class ConvTas_Net(nn.Module):
    def __init__(self, num_mic, L, N, B, H, P, X, R, causal=True, norm_type='cLN'):
        super(ConvTas_Net, self).__init__()
        self.L = L
        self.n_mic = num_mic
        self.encoder = Encoder(
            kernel_size=2 * L,
            in_channels=num_mic,
            out_channels=N)

        self.separator = MaskNet(
            N = N,
            B = B,
            H = H,
            P = P,
            X = X,
            R = R,
            C = 1,
            norm_type = norm_type,
            causal = causal,
            mask_nonlinear = 'relu')

        self.decoder = Decoder(
            in_channels=N,
            out_channels=1,
            kernel_size=2 * L,
            stride=L,
            bias=False)

    def forward(self, mixed):
        """
        Extracts the audio corresponding to the `label` in the given
        `mixture`.

        Args:
            mixed: [B, n_mics, T]
                input audio mixture
            label: [B, num_labels]
                one hot label
        Returns:
            out: [B, n_spk, T]
                extracted audio with sounds corresponding to the `label`
        """
        # print(mixed.shape)
        x, mod = mod_pad(mixed, self.L, pad=(0, self.L))
        # print(x.shape)
        # [B, n_mics, T] --> [B, n_channels, T']
        #print(x.shape)
        if self.n_mic == 1:
            x = x.squeeze(1)
        x = self.encoder(x)
        #print(x.shape)
        # Computes label embedding using a linear layer
        # [B, num_labels] --> [B, n_channels, 1]
        # Generate filtered latent space signal for each speaker
        mask = self.separator(x) # [1, B, n_channels, T']
        mask = mask.squeeze(0)

        # Decode filtered signals
        out = x * mask # {[B, n_channels, T'], ...}
        out = self.decoder(out).unsqueeze(1) # {[B, T], ...}
        out = out[..., : -self.L]

        #raise KeyboardInterrupt
        # Remove mod padding, if present.
        if mod != 0:
            out = out[:, :, :-mod]

        return out


class Net(nn.Module):
    def __init__(self,num_mic, L, N, B, H, P, X, R):
        super(Net, self).__init__()
        
        self.net = ConvTas_Net(
            num_mic = num_mic,
            L = L, 
            N = N, 
            B = B, 
            H = H, 
            P = P, 
            X = X, 
            R = R)


    def predict(self, x, pad=False):


        x = self.net(x)
        # x = x[..., : -self.stft_pad_size]
        

        return x, None


    def forward(self, inputs, input_state = None, pad=False):
        x = inputs['mixture']

        x, next_state= self.predict(x, pad)

        return {'output': x, 'next_state': next_state}



if __name__ == "__main__":
    model_params = {
        "num_mic": 2,
        "L": 8,
        "N": 256,
        "B": 256,
        "H": 256,
        "P": 3,
        "X": 8,
        "R": 4
    }


    net = Net(**model_params)

    a = torch.rand(2, 2, 24000)
    inputs = {
        'mixture': a
    }
    print(net(inputs)['output'].shape)
