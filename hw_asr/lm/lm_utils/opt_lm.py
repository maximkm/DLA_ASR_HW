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

def train_beam_search(texts, logits_list, text_encoder):
    def optuna_trial(trial):
        ctc_decoder = build_ctcdecoder(text_encoder.get_vocab(),
                                       kenlm_model_path='5_full_gram.arpa',
                                       alpha=trial.suggest_float('alpha', 0.55, 0.8),
                                       beta=trial.suggest_float('beta', 0.15, 0.4))
        with multiprocessing.get_context("fork").Pool() as pool:
            pred_list = ctc_decoder.decode_batch(
                pool,
                logits_list,
                beam_prune_logp=trial.suggest_float('beam_prune_logp', -10, -2),
                token_min_logp=trial.suggest_float('token_min_logp', -10, -2),
                beam_width=trial.suggest_int('beam_width', 500, 1500))
        return calc_wer_batch(texts, pred_list)
    return optuna_trial
    

def main(config, out_file):
    logger = config.get_logger("test")

    # define cpu or gpu if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # text_encoder
    text_encoder = config.get_text_encoder()

    # setup data_loader instances
    dataloaders = get_dataloaders(config, text_encoder)

    # build model architecture
    model = config.init_obj(config["arch"], module_model, n_class=len(text_encoder))
    logger.info(model)

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    try:
        checkpoint = torch.load(config.resume, map_location=device)
    except:
        import pathlib

        pathlib.PosixPath = pathlib.WindowsPath
        checkpoint = torch.load(config.resume, map_location=device)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device)
    model.eval()

    texts = []
    pred_text_argmax = []
    logits_list = []
    ctc_decoder = build_ctcdecoder(text_encoder.get_vocab(), kenlm_model_path='5_full_gram.arpa')
    start = timer()
    with torch.no_grad():
        for batch_num, batch in enumerate(tqdm(dataloaders["val"])):
            batch = Trainer.move_batch_to_device(batch, device)
            output = model(**batch)
            if type(output) is dict:
                batch.update(output)
            else:
                batch["logits"] = output
            batch["log_probs"] = torch.log_softmax(batch["logits"], dim=-1)
            batch["log_probs_length"] = model.transform_input_lengths(
                batch["spectrogram_length"]
            )
            batch["probs"] = batch["log_probs"].exp().cpu()
            batch["argmax"] = batch["probs"].argmax(-1)
            
            for i in range(len(batch["text"])):
                argmax = batch["argmax"][i]
                argmax = argmax[: int(batch["log_probs_length"][i])]
                texts.append(text_encoder.normalize_text(batch["text"][i]))
                pred_text_argmax.append(text_encoder.ctc_decode(argmax.cpu().numpy()))
                logits_list.append(batch["log_probs"][i][:batch["log_probs_length"][i]].cpu().numpy())
    print(f'calc batches {timer() - start}')
    
    start = timer()
    study_beam_search = optuna.create_study(direction='minimize')
    study_beam_search.optimize(train_beam_search(texts, logits_list, text_encoder), n_trials=30)
    print('study', timer() - start)
    print(f'Best params {study_beam_search.best_params}')
    print(f'Result {study_beam_search.best_value}')


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=str(DEFAULT_CHECKPOINT_PATH.absolute().resolve()),
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    args.add_argument(
        "-o",
        "--output",
        default="output.json",
        type=str,
        help="File to write results (.json)",
    )
    args.add_argument(
        "-t",
        "--test-data-folder",
        default=None,
        type=str,
        help="Path to dataset",
    )
    args.add_argument(
        "-b",
        "--batch-size",
        default=20,
        type=int,
        help="Test dataset batch size",
    )
    args.add_argument(
        "-j",
        "--jobs",
        default=1,
        type=int,
        help="Number of workers for test dataloader",
    )

    args = args.parse_args()

    # set GPUs
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # first, we need to obtain config with model parameters
    # we assume it is located with checkpoint in the same folder
    model_config = Path(args.resume).parent / "config.json"
    with model_config.open() as f:
        config = ConfigParser(json.load(f), resume=args.resume)

    # update with addition configs from `args.config` if provided
    if args.config is not None:
        with Path(args.config).open() as f:
            config.config.update(json.load(f))

    # if `--test-data-folder` was provided, set it as a default test set
    if args.test_data_folder is not None:
        test_data_folder = Path(args.test_data_folder).absolute().resolve()
        assert test_data_folder.exists()
        config.config["data"] = {
            "test": {
                "batch_size": args.batch_size,
                "num_workers": args.jobs,
                "datasets": [
                    {
                        "type": "CustomDirAudioDataset",
                        "args": {
                            "audio_dir": str(test_data_folder / "audio"),
                            "transcription_dir": str(
                                test_data_folder / "transcriptions"
                            ),
                        },
                    }
                ],
            }
        }

    assert config.config.get("data", {}).get("val", None) is not None
    config["data"]["val"]["batch_size"] = args.batch_size
    config["data"]["val"]["n_jobs"] = args.jobs

    main(config, args.output)
