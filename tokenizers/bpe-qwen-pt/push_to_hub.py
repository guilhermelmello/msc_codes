import os
import huggingface_hub
from transformers import AutoTokenizer

repository_path = 'guilhermelmello/bpe-pt-30000'
tokenizer_path = './models/tokenizer-30000.json'

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# huggingface_hub.notebook_login()
huggingface_hub.login(os.environ['HF_TOKEN'])
tokenizer.push_to_hub(repository_path)
