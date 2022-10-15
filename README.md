# ASR project
The project is made for educational purposes, as the homework of the course [deep learning for audio processing](https://github.com/markovka17/dla).

## Installation guide
It is recommended to use python 3.8 or 3.9

You need to clone the repository and install the libraries:
```shell
git clone https://github.com/maximkm/DLA_ASR_HW.git
cd DLA_ASR_HW
pip install -r requirements.txt
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

[Wandb report](https://wandb.ai/maximkm/asr_project/reports/Homework-report--VmlldzoyNzk4OTIw?accessToken=68k3szx6w2gvylj2b0vnid1c9lx628q0v0b770rkglmyw5m61qgnanoc4auhemdt)

## The final score received

| Dataset | Type predict | CER  | WER |
| ------------- | ------------- | ------------- | ------------- |
| LibriSpeech: test-clean | beam search | **0.06742**  | **0.12988** |
| LibriSpeech: test-other | beam search | **0.17529**  |  **0.27248** |
| LibriSpeech: test-clean | argmax  | 0.07794 | 0.21284  |
| LibriSpeech: test-other | argmax  | 0.17529 | 0.38656  |

## Independent code testing

You need to download:
1) [The final checkpoint](https://drive.google.com/uc?id=10Ubmu6-w415A2jiUXobJL4ZzMy7A5fxW) of the model and put the save folder in the main directory
2) [LM](https://drive.google.com/uc?id=1WGFJgzrh850BSXkaCb-dzsWqK894Dmd0) and place the file in the hw_asr/lm directory

You can run this script:
```shell
gdown https://drive.google.com/uc?id=10Ubmu6-w415A2jiUXobJL4ZzMy7A5fxW
unzip saved.zip
gdown https://drive.google.com/uc?id=1WGFJgzrh850BSXkaCb-dzsWqK894Dmd0
mv 5_full_gram.arpa hw_asr/lm
```

Now you can run the code:
1) You need to run the model with the following command:
```bash
python test.py -c hw_asr/configs/test_ctc_big_clean.json -r saved/models/baseline/1013_154403/model_best.pth -o test-clean.json
```
This command loads the prepared `test_ctc_big_clean.json` config inside of which contains the description of the model and dataset.

After processing all the data will save the predictions in `test-clean.json`.

Similarly, the **`test_ctc_big_other.json`** config was created. Also at test.py there is a `-t` argument to specify a folder with a dataset.

2) The last step is to run a script to calculate the WER and CER metrics
```bash
python calc_wer_cer.py -t test-clean.json
```

## Credits

This repository is based on a heavily modified fork
of [pytorch-template](https://github.com/victoresque/pytorch-template) repository.

The CTC transformer architecture is based on [Transformers with convolutional context for ASR](https://arxiv.org/pdf/1904.11660.pdf).
