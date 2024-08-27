import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import(
    scale_invariant_signal_noise_ratio as si_snr,
    signal_noise_ratio as snr,
    signal_distortion_ratio as sdr,
    scale_invariant_signal_distortion_ratio as si_sdr)

from .block import SBTransformerBlock_wnormandskip, ResourceEfficientSeparator
from speechbrain.lobes.models.dual_path import Encoder
from speechbrain.lobes.models.dual_path import Decoder

class Net(nn.Module):
    def __init__(self, num_mics, num_spk, model_dim=128,
                 buf_size=70):
        super(Net, self).__init__()
        self.n_channels = 256
        self.kernel_size = 32
        self.num_mics = num_mics
        self.num_spk = num_spk
        self.segment_size = 150

        self.encoder = Encoder(
            kernel_size=self.kernel_size,
            in_channels=self.num_mics,
            out_channels=self.n_channels)
        self.intra_mdl = SBTransformerBlock_wnormandskip(
            num_layers=8,
            d_model=self.n_channels,
            nhead=8,
            d_ffn=1024,
            dropout=0,
            use_positional_encoding=True,
            norm_before=True,
            use_norm=True,
            use_skip=True)
        self.mem_mdl = SBTransformerBlock_wnormandskip(
            num_layers=8,
            d_model=self.n_channels,
            nhead=8,
            d_ffn=1024,
            dropout=0,
            use_positional_encoding=True,
            norm_before=True,
            use_norm=True,
            use_skip=True)
        self.separator = ResourceEfficientSeparator(
            input_dim=self.n_channels,
            num_spk=self.num_spk,
            causal=False,
            unit=256,
            segment_size=self.segment_size,
            layer=2,
            mem_type='av',
            seg_model=self.intra_mdl,
            mem_model=self.mem_mdl)
        self.decoder = Decoder(
            in_channels=self.n_channels,
            out_channels=1,
            kernel_size=self.kernel_size,
            stride=self.kernel_size//2,
            bias=False)

    def forward(self, inputs):
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
        x = inputs['mixture']
        # [B, n_mics, T] --> [B, n_channels, T']
        m = self.encoder(x)

        # Computes label embedding using a linear layer
        # [B, num_labels] --> [B, n_channels, 1]

        # Generate filtered latent space signal for each speaker
        masks = self.separator(m) # [n_spk, B, n_channels, T']

        # Decode filtered signals
        out = [m * msk for msk in masks] # {[B, n_channels, T'], ...}
        out = [self.decoder(o) for o in out] # {[B, T], ...}
        out = torch.stack(out, dim=1) # [B, n_spk, T]
        # out = torch.tanh(out)

        return {'output': out, 'next_state': None}

