from hw_asr.augmentations.base import AugmentationBase
from hw_asr.augmentations.random_apply import RandomApply
import torchaudio
import torch


class TimeMasking(AugmentationBase):
    def __init__(self, time_mask_param=40, count=4, p=0.15):
        self.augs = [RandomApply(torchaudio.transforms.TimeMasking(time_mask_param=time_mask_param), p=p)
                     for _ in range(count)]

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        for aug in self.augs:
            data = aug(data)
        return data