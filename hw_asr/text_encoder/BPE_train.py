from hw_asr.base.base_text_encoder import BaseTextEncoder
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from string import ascii_lowercase
from json import load

VOCAB_SIZE = 500

tokenizer = Tokenizer(BPE())
trainer = BpeTrainer(vocab_size=VOCAB_SIZE, initial_alphabet=list(ascii_lowercase))

with open('data/datasets/librispeech/train-clean-100_index.json', 'r') as dataset:
    data = load(dataset)

texts = []
for row in data:
    texts.append(BaseTextEncoder.normalize_text(row['text']))

tokenizer.pre_tokenizer = Whitespace()
tokenizer.train_from_iterator(texts, trainer)
tokenizer.save(f'hw_asr/text_encoder/BPE_tokenizer_{VOCAB_SIZE}.json')
print('tokenizer save')
