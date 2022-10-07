from hw_asr.augmentations.base import AugmentationBase
from hw_asr.augmentations.random_apply import RandomApply
import torchaudio
import torch

import random


class Fade(AugmentationBase):
    def __init__(self, p=0.15, fade_shape="linear") -> torch.Tensor:
        self.aug = RandomApply(torchaudio.transforms.Fade(fade_shape=fade_shape), p=p)

    def __call__(self, waveform: torch.Tensor):
        self.aug.augmentation.fade_in_len = random.randint(0, waveform.size(-1) // 2)
        self.aug.augmentation.fade_out_len = random.randint(0, waveform.size(-1) // 2)
        return self.aug(waveform)
