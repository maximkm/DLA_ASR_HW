from hw_asr.base.base_text_encoder import BaseTextEncoder
from json import load

with open('data/datasets/librispeech/train-clean-100_index.json', 'r') as dataset:
    data = load(dataset)

texts = []
for row in data:
    texts.append(BaseTextEncoder.normalize_text(row['text']))

with open("hw_asr/text_encoder/train_data_new_line.txt", "w") as file:
    for text in texts:
        file.write(text + '\n')
