import torch


class ChannelGainPerturbation:
    def __init__(self, max_channel_gain_db, reference_channels = [0], unique=False):
        self.max_channel_gain = max_channel_gain_db
        self.unique = unique
        self.reference_channels = reference_channels

    def __call__(self, audio_data, gt_audio):
        C = audio_data.shape[0]

        def random_gain():
            gain_db = 2 * (torch.rand((1,)).item() - 0.5) * self.max_channel_gain
            scale = 10 ** (gain_db / 20)
            return scale

        # Get gain for each channel
        if self.unique:
            unique_gain = random_gain()
            gains = [unique_gain] * C
        else:
            gains = [random_gain() for _ in range(C)]

        # Apply shift on every channel
        gt_channel = 0
        for i in range(audio_data.shape[0]):
            gain = gains[i]
            audio_data[i] = audio_data[i] * gain
            
            if i in self.reference_channels:
                gt_audio[gt_channel] = gt_audio[gt_channel] * gain
                gt_channel += 1
            
        return audio_data, gt_audio