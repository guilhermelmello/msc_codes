from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset


load_dataset('carolina-c4ai/corpus-carolina', revision='v2.0.1', split='corpus')

AutoTokenizer.from_pretrained('guilhermelmello/tokenizer-unigram-pt-10k')
AutoTokenizer.from_pretrained('guilhermelmello/tokenizer-bpe-pt-10k')

AutoModel.from_pretrained('Qwen/Qwen3-0.6B')
