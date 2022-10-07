from hw_asr.augmentations.base import AugmentationBase
import torch_audiomentations
import torch


class PitchShift(AugmentationBase):
    def __init__(self, sample_rate, p=0.15, mode="per_example", p_mode="per_example"):
        self.aug = torch_audiomentations.PitchShift(sample_rate=sample_rate, p=p, mode=mode, p_mode=p_mode)

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        x = data.unsqueeze(1)
        return self.aug(x).squeeze(1)
