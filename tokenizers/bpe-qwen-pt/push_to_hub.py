import os
import huggingface_hub
from transformers import PreTrainedTokenizerFast

repository_path = 'guilhermelmello/bpe-pt-30000'
tokenizer_path = './tokenizers/tokenizer_30000.json'

tokenizer = PreTrainedTokenizerFast(
    tokenizer_file=tokenizer_path,
    model_max_length=2048,
    add_prefix_space=True,
    pad_token='[PAD]',
    eos_token='[EOS]',
    sep_token='[SEP]',
)

# huggingface_hub.notebook_login()
huggingface_hub.login(os.environ['HF_TOKEN'])
tokenizer.push_to_hub(repository_path)
