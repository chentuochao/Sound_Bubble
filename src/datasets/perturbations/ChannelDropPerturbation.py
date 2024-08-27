import torch


class ChannelDropPerturbation:
    def __init__(self, max_channel_drops):
        self.max_channel_drops = max_channel_drops

    def __call__(self, audio_data, gt_audio):
        C = audio_data.shape[0]

        n_channels_to_drop = torch.randint(1, self.max_channel_drops + 1, (1,)).item()

        perm = 1 + torch.randperm(C-1)
        channels_to_drop = perm[:n_channels_to_drop]

        perturbed_audio_data = audio_data
        for ch in channels_to_drop:
            perturbed_audio_data[ch] *= 0
            
        return perturbed_audio_data, gt_audio