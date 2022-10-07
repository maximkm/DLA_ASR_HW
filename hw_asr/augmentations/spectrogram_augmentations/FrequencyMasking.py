from hw_asr.augmentations.base import AugmentationBase
from hw_asr.augmentations.random_apply import RandomApply
import torchaudio
import torch


class FrequencyMasking(AugmentationBase):
    def __init__(self, freq_mask_param=15, p=0.15):
        self.aug = RandomApply(torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask_param), p=p)

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        return self.aug(data)
