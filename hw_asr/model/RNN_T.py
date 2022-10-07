import torch
import torch.nn as nn

from hw_asr.base import BaseModel


class CTCTransformerDecoder(BaseModel):
    def __init__(self, **batch):
        super().__init__(**batch)

    def forward(self, spectrogram, **batch):
        raise NotImplementedError

    def transform_input_lengths(self, input_lengths):
        raise NotImplementedError
