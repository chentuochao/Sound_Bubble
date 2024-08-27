import torch


class FrequencyMaskingPerturbation:
    def __init__(self, min_freq_masks, max_freq_masks, unique=False, nfft=4096, reference_channels = [0]):
        self.min_freq_masks = min_freq_masks
        self.max_freq_masks = max_freq_masks

        self.unique = unique
        self.nfft = nfft
        self.reference_channels = reference_channels

    def __call__(self, audio_data, gt_audio):
        C = audio_data.shape[0]
        T = audio_data.shape[-1]
        N = self.nfft //2 + 1

        # Get shifts for each channel
        if self.unique:
            unique_nfreqs = torch.randint(self.min_freq_masks, self.max_freq_masks + 1, (1,)).item()
            freqs = torch.randperm(N)[:unique_nfreqs]
            freqs = [freqs] * C
        else:
            freqs = []
            for i in range(C):
                n_masks = torch.randint(self.min_freq_masks, self.max_freq_masks + 1, (1,)).item()
                freqs.append(torch.randperm(N)[:n_masks])
        
        perturbed_audio_data = audio_data
        perturbed_gt_audio = gt_audio
        
        gt_channel = 0
        for i in range(C):
            mask = freqs[i]
            
            S = torch.stft(perturbed_audio_data[i], n_fft=self.nfft, return_complex=True)
            S[mask] = 0
            perturbed_audio_data[i] = torch.istft(S, n_fft=self.nfft, length=T)

            if i in self.reference_channels:                
                S = torch.stft(perturbed_gt_audio[gt_channel], n_fft=self.nfft, return_complex=True)
                S[mask] = 0
                perturbed_gt_audio[gt_channel] = torch.istft(S, n_fft=self.nfft, length=T)
                
                gt_channel += 1
        
        return perturbed_audio_data, perturbed_gt_audio