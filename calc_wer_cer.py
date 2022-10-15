import argparse
import json
import os
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

import hw_asr.model as module_model
from hw_asr.trainer import Trainer
from hw_asr.utils import ROOT_PATH
from hw_asr.utils.object_loading import get_dataloaders
from hw_asr.utils.parse_config import ConfigParser
from hw_asr.metric.utils import calc_wer, calc_cer
from joblib import Parallel, delayed
from pyctcdecode import build_ctcdecoder
from timeit import default_timer as timer
import multiprocessing
import optuna

DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "default_test_model" / "checkpoint.pth"


def calc_wer_batch(texts, preds):
    wer = 0
    for text, pred in zip(texts, preds):
        wer += calc_wer(text, pred)
    return wer / len(texts)
    
def calc_cer_batch(texts, preds):
    cer = 0
    for text, pred in zip(texts, preds):
        cer += calc_cer(text, pred)
    return cer / len(texts)

def main(config):
    with open(config.test_data_file, 'r') as file:
        data = json.load(file)
    true_texts = []
    argmax_texts =[]
    pred_texts = []
    for elem in data:
        true_texts.append(elem['ground_trurh'])
        argmax_texts.append(elem['pred_text_argmax'])
        pred_texts.append(elem['pred_text_beam_search'])
    print(f'{config.test_data_file} CER (argmax): {calc_cer_batch(true_texts, argmax_texts)}')
    print(f'{config.test_data_file} WER (argmax): {calc_wer_batch(true_texts, argmax_texts)}')
    print(f'{config.test_data_file} CER (bs): {calc_cer_batch(true_texts, pred_texts)}')
    print(f'{config.test_data_file} WER (bs): {calc_wer_batch(true_texts, pred_texts)}')


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-t",
        "--test-data-file",
        default=None,
        type=str,
        help="Path to dataset",
    )
    
    args = args.parse_args()
    main(args)