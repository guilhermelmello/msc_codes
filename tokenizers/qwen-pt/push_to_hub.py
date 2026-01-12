import os
import huggingface_hub
from transformers import AutoTokenizer

model_names = [
    ('bpe5k-nfc',  'tokenizer-bpe-pt-5k'),
    ('bpe8k-nfc',  'tokenizer-bpe-pt-8k'),
    ('bpe10k-nfc', 'tokenizer-bpe-pt-10k'),
    ('bpe15k-nfc', 'tokenizer-bpe-pt-15k'),
    ('bpe30k-nfc', 'tokenizer-bpe-pt-30k'),
    ('bpe50k-nfc', 'tokenizer-bpe-pt-50k'),
    ('unigram5k-nfc', 'tokenizer-unigram-pt-5k'),
    ('unigram8k-nfc', 'tokenizer-unigram-pt-8k'),
    ('unigram10k-nfc', 'tokenizer-unigram-pt-10k'),
    ('unigram15k-nfc', 'tokenizer-unigram-pt-15k'),
    ('unigram30k-nfc', 'tokenizer-unigram-pt-30k'),
    ('unigram50k-nfc', 'tokenizer-unigram-pt-50k'),
]

huggingface_hub.login()

for local_name, repo_id in model_names:
    tokenizer_path = os.path.join('models', local_name)

    print('Saving tokenizer:')
    print('Local Path:', tokenizer_path)
    print('Hub ID:', repo_id)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, padding_side='left')
    tokenizer.push_to_hub(f'guilhermelmello/{repo_id}')
