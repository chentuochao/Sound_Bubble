#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright 2022 Lucky Wong
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

import math
from pyclbr import Class

import torch

from .stft import ConvSTFT
from .mask import make_pad_mask
from typing import Optional
import torch.nn.functional as F

def _remove_mean(x: torch.Tensor, dim: Optional[int] = -1) -> torch.Tensor:
    return x - x.mean(dim=dim, keepdim=True)


class PLCPALoss(torch.nn.Module):
    """The power-law compressed phaseaware (PLCPA) loss

    Reference: 
    Human Listening and Live Captioning: Multi-Task Training for Speech Enhancement
    https://arxiv.org/abs/2106.02896

    Attributes:
        window_size:
            list of STFT window sizes.
        hop_size:
            list of hop_sizes, default is each window_size // 2.
        power:
            power for doing compression
        eps: (float)
            stability epsilon
        zero_mean:
            remove DC
    """

    def __init__(
        self,
        window_size: int = 320,
        hop_size: Optional[int] = 160,
        fft_len: Optional[int] = 512,
        power: float = 0.3,
        eps: float = 1.0e-12,
        zero_mean: bool = True,
        scale_asym: float = 0.0,
        scale_mag: float = 1.0,
        scale_phase: float = 1.0,
        return_all: bool = False
    ):
        super().__init__()

        if fft_len is None:
            fft_len = int(2 ** math.ceil(math.log2(window_size)))

        self.return_all = return_all

        self.stft = ConvSTFT(
            win_len=window_size,
            win_inc=window_size // 2 if hop_size is None else hop_size,
            fft_len=fft_len,
            win_type="hamming",
            feature_type="complex",
            fix=True,
        )
        self.feat_dim = fft_len // 2 + 1
        self.power = power
        self.eps = eps
        self.zero_mean = zero_mean

        self.scale_asym = scale_asym
        self.scale_mag = scale_mag
        self.scale_phase = scale_phase

    def _phasen_loss(self, ref_spectrograms, est_spectrograms):
        """
        The PHASEN loss comprises two parts: amplitude and phase-aware losses

        ref_spectrum: [B, F*2, T], the reference spectrograms
        est_spectrum: [B, F*2, T], the estimated spectrograms
        """
        def _amplitude(x):
            r = x[:, : self.feat_dim, :]
            i = x[:, self.feat_dim:, :]
            return torch.sqrt(r ** 2 + i ** 2 + self.eps)

        # step 1: amplitude loss
        est_amplitude = _amplitude(est_spectrograms)
        ref_amplitude = _amplitude(ref_spectrograms)

        # Hyper-parameter p is a spectral compression factor and is set to 0.3
        est_compression_amplitude = est_amplitude ** self.power
        ref_compression_amplitude = ref_amplitude ** self.power

        mag_loss = F.mse_loss(
            est_compression_amplitude, ref_compression_amplitude)

        # step 2: phase-aware losses
        '''
        s = a+i*b

        amplitude = (a^2 + b^2)
        r = amplitude**0.3

        -> s' = r*(cos(θ)+i*sin(θ))
        θ = arctan(b/a)
        cos(arctan(b/a)) = sqrt(a^2 / (a^2 + b^2))
        sin(arctan(b/a)) = sqrt(b^2 / (a^2 + b^2))
        
        -> s' = a'+i*b'
        a' = r*cos(θ) = r*sqrt(a^2 / (a^2 + b^2)) = a*r / amplitude
        b' = r*sin(θ) = r*sqrt(b^2 / (a^2 + b^2)) = b*r / amplitude
        '''
        est_compression_spectrum = est_spectrograms * \
            (est_compression_amplitude / est_amplitude).repeat(1, 2, 1)
        ref_compression_spectrum = ref_spectrograms * \
            (ref_compression_amplitude / ref_amplitude).repeat(1, 2, 1)

        phase_aware_loss = F.mse_loss(
            est_compression_spectrum, ref_compression_spectrum)

        # The PHASEN loss comprises two parts: amplitude and phase-aware losses
        loss = self.scale_mag * mag_loss + self.scale_phase * phase_aware_loss
        
        if self.scale_asym > 0.0:
            # To solve the speech over-suppression issue
            # Reference: TEA-PSE: Tencent-Ethereal-Audio-Lab Personalized Speech Enhancement System for ICASSP 2022 DNS Challenge
            delta = ref_compression_amplitude - est_compression_amplitude
            zero_tensor = torch.tensor(0.0, dtype=delta.dtype).to(delta.device)
            asym_loss = torch.mean(
                torch.square(torch.where(delta > 0, delta, zero_tensor))   
            )
            
            # print(mag_loss, phase_aware_loss, asym_loss)
            loss2 = loss + self.scale_asym * asym_loss
        else:
            loss2 = loss
            asym_loss = 0
        if self.return_all:
            return loss2, loss, asym_loss
        else:
            return loss2

    def forward(
        self,
        gt: torch.Tensor,
        est: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """phase-aware forward.

        Args:

            ref: Tensor, (..., n_samples)
                reference signal
            est: Tensor (..., n_samples)
                estimated signal

        Returns:
            loss: (...,)
                the PLCPA loss
        """
        ref = gt
        # print(ref.type(), est.type())
        assert ref.shape == est.shape

        if self.zero_mean:
            ref = _remove_mean(ref, dim=-1)
            est = _remove_mean(est, dim=-1)

        if lengths is not None:
            masks = make_pad_mask(lengths)  # (B, T)
            est = est.masked_fill(masks, 0.0)
            ref = ref.masked_fill(masks, 0.0)

        return self._phasen_loss(self.stft(ref), self.stft(est))




