from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
from huggingface_hub import login

import os

def main():
    login() # hf hub

    # required datasets
    load_dataset('carolina-c4ai/corpus-carolina', revision='v2.0.1', split='corpus', trust_remote_code=True)

    # required tokenizers
    AutoTokenizer.from_pretrained('guilhermelmello/tokenizer-unigram-pt-10k')
    AutoTokenizer.from_pretrained('guilhermelmello/tokenizer-bpe-pt-10k')

    # required models
    AutoModel.from_pretrained('Qwen/Qwen3-0.6B')


if __name__ == '__main__':
    main()
