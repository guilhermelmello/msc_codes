import os
import huggingface_hub
from transformers import AutoTokenizer

model_names = [
    ('bpe10k', 'tokenizer-bpe-pt-10k'),
    ('bpe30k', 'tokenizer-bpe-pt-30k'),
    ('bpe50k', 'tokenizer-bpe-pt-50k'),
]

huggingface_hub.login(os.environ['HF_TOKEN'])

for local_name, repo_id in model_names:
    tokenizer_path = os.path.join('models', local_name)

    print('Saving tokenizer:')
    print('Local Path:', tokenizer_path)
    print('Hub ID:', repo_id)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.push_to_hub(repo_id, private=True)
