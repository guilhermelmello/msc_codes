import os
import huggingface_hub
from transformers import AutoTokenizer

model_names = [
    ('bpe5k',  'tokenizer-bpe-pt-5k'),
    ('bpe10k', 'tokenizer-bpe-pt-10k'),
    ('bpe30k', 'tokenizer-bpe-pt-30k'),
    ('bpe50k', 'tokenizer-bpe-pt-50k'),
    ('unigram5k', 'tokenizer-unigram-pt-5k'),
    ('unigram10k', 'tokenizer-unigram-pt-10k'),
    ('unigram30k', 'tokenizer-unigram-pt-30k'),
    ('unigram50k', 'tokenizer-unigram-pt-50k'),
]

huggingface_hub.login(os.environ['HF_TOKEN'])

for local_name, repo_id in model_names:
    tokenizer_path = os.path.join('models', local_name)

    print('Saving tokenizer:')
    print('Local Path:', tokenizer_path)
    print('Hub ID:', repo_id)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.push_to_hub(repo_id, private=False)

