import torch
import torchaudio
import torch.nn.functional as F

class SpeedPerturbation:
    def __init__(self, min_speed, max_speed, sample_rate = 24000):
        self.min_speed = min_speed
        self.max_speed = max_speed
        
        self.sample_rate = sample_rate

    def __call__(self, audio_data, gt_audio):
        T = audio_data.shape[-1]
        speed_factor = torch.rand((1,)).item() * (self.max_speed - self.min_speed) + self.min_speed

        # change speed and resample to original rate:
        sox_effects = [
            ["speed", str(speed_factor)],
            ["rate", str(self.sample_rate)],
        ]
        transformed_audio, _ = torchaudio.sox_effects.apply_effects_tensor(
            audio_data, self.sample_rate, sox_effects)
        
        gt_audio, _ = torchaudio.sox_effects.apply_effects_tensor(
            gt_audio, self.sample_rate, sox_effects)
        
        # Adjust size so it is the same as the original size
        if transformed_audio.shape[-1] > T:
            transformed_audio = transformed_audio[..., :T]
            gt_audio = gt_audio[..., :T]
        else:
            transformed_audio = F.pad(transformed_audio, (0, T - transformed_audio.shape[-1]))
            gt_audio = F.pad(gt_audio, (0, T - gt_audio.shape[-1]))

        assert transformed_audio.shape[-1] == T
        assert gt_audio.shape[-1] == T

        return transformed_audio, gt_audio
