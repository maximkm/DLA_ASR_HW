import logging
from typing import List

import torch
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    result_batch = {
        "text": [],
        "audio_path": [],
        "spectrogram": [],
        "spectrogram_length": [],
        "text_encoded": [],
        "text_encoded_length": []
    }

    for item in dataset_items:
        result_batch['text'].append(item['text'])  # TODO normalize?
        result_batch['audio_path'].append(item['audio_path'])

        result_batch['text_encoded_length'].append(item['text_encoded'].size(1))
        result_batch['spectrogram_length'].append(item['spectrogram'].size(2))

        result_batch['text_encoded'].append(item['text_encoded'].squeeze(0))
        result_batch['spectrogram'].append(item['spectrogram'].squeeze(0).T)

    for key in ['text_encoded_length', 'spectrogram_length']:
        result_batch[key] = torch.IntTensor(result_batch[key])
    for key in ['spectrogram', 'text_encoded']:
        result_batch[key] = pad_sequence(result_batch[key], batch_first=True)
    result_batch['spectrogram'] = result_batch['spectrogram'].transpose(1, 2)

    return result_batch
