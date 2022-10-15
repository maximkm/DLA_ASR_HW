# ASR project

## Installation guide

< Write your installation guide here >

```shell
pip install -r ./requirements.txt
```

## Description of the work done

- [x] Write all the basic functions and classes
- [x] Pass unit tests 
- [x] Write and test a CTC transformer encoder from RNN-T
- [x] Write augmentations
- [x] Conduct the first experiments and select the optimal hyperparameters and augmentations
- [x] Train BPE on training texts and implement it into model training
- [x] Train the final model with the best parameters
- [x] Train a LM on training texts
- [x] Implement LM in beam search
- [x] Choose optimal beam search hyperparameters using optuna
- [x] Write an implementation of the Common Voice dataset and write a config for the finetune model
- [ ] Finetune model on Common Voice

## The final score received

| Dataset | Type predict | CER  | WER |
| ------------- | ------------- | ------------- | ------------- |
| LibriSpeech: test-clean | beam search | **0.06742**  | **0.12988** |
| LibriSpeech: test-other | beam search | **0.17529**  |  **0.27248** |
| LibriSpeech: test-clean | argmax  | 0.07794 | 0.21284  |
| LibriSpeech: test-other | argmax  | 0.17529 | 0.38656  |

## Independent code testing

You need to download:
1) The final checkpoint of the model and put the save folder in the main directory
2) LM and place the file in the hw_asr/lm directory

Now you can run the code:
1) You need to run the model with the following command:
```bash
python test.py -c hw_asr/configs/test_ctc_big_clean.json -r saved/models/baseline/1013_154403/model_best.pth -o test-clean.json
```
This command loads the prepared `test_ctc_big_clean.json` config inside of which contains the description of the model and dataset.

After processing all the data will save the predictions in `test-clean.json`.

Similarly, the **`test_ctc_big_other.json`** config was created. Also at test.py there is a `-t` argument to specify a folder with a dataset.

2) The last step is to run a script to calculate the WAR and CER metrics
```bash
python calc_wer_cer.py -t test-clean.json
```

## Credits

This repository is based on a heavily modified fork
of [pytorch-template](https://github.com/victoresque/pytorch-template) repository.

The CTC transformer architecture is based on [Transformers with convolutional context for ASR](https://arxiv.org/pdf/1904.11660.pdf).
