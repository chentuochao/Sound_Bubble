import torch


class SampleShiftPerturbation:
    def __init__(self, max_shift, reference_channels = [0], unique=False):
        """
        max_shift: Maximum shift (inclusive) in both directions
        unique: Whether the same shift across channels is unique
        """
        self.max_shift = max_shift
        self.unique = unique
        self.reference_channels = reference_channels

    def __call__(self, audio_data, gt_audio):
        C = audio_data.shape[0]
        
        # Get shifts for each channel
        if self.unique:
            unique_shift = torch.randint(-self.max_shift, self.max_shift + 1, (1,))
            shifts = [unique_shift] * C
        else:
            shifts = [torch.randint(-self.max_shift, self.max_shift + 1, (1,)) for _ in range(C)]

        # Apply shift on every channel
        gt_channel = 0
        for i in range(audio_data.shape[0]):
            shift = shifts[i].item()
            audio_data[i] = torch.roll(audio_data[i], shift, dims=-1)
            
            if i in self.reference_channels:
                gt_audio[gt_channel] = torch.roll(gt_audio[gt_channel], shift, dims=-1)
                gt_channel += 1
        
        return audio_data, gt_audio