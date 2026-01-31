from enum import Enum
from transformers import AutoConfig, AutoModelForCausalLM

import torch


class InitMode(Enum):
    BASE_CONFIG='base-config'
    DEFAULT_CONFIG='default-config'
    PRETRAINED='pretrained'


def initialize_clm(model_name, tokenizer, init_mode: InitMode):
    vocab_kw = dict(
        vocab_size=tokenizer.vocab_size,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        max_position_embeddings=tokenizer.model_max_length,
    )

    if init_mode == InitMode.BASE_CONFIG:
        print('Initializing random CLM with BASE size.')
        model_config = AutoConfig.from_pretrained(
            model_name,
            # hidden_size=768,
            num_hidden_layers=12,
            # num_attention_heads=12,
            **vocab_kw,
        )
        return AutoModelForCausalLM.from_config(model_config)

    if init_mode == InitMode.DEFAULT_CONFIG:
        print('Initializing random CLM from CONFIG.')
        model_config = AutoConfig.from_pretrained(
            model_name,
            **vocab_kw,
        )
        return AutoModelForCausalLM.from_config(model_config)

    raise NotImplementedError
    # if init_config == InitConfig.PRETRAINED:
    #     print('Initializing CLM from PRETRAINED model.')
    #     return AutoModelForCausalLM.from_pretrained(model_name)


def print_model_size(model):
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    size_all_mb = (param_size + buffer_size) / 1024**2

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total Trainable Params: {total_params:,}")
    print(f"Model size (MB): {size_all_mb:.2f}")
    print('Vocab Size:', model.config.vocab_size)
    print('L:', model.config.num_hidden_layers)
    print('H:', model.config.hidden_size)
    print('A:', model.config.num_attention_heads)


def print_gpu_usage():
    # Check if CUDA is available
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties()
        print(f'Total GPU Memory: {gpu.total_memory / 1024**3:.2f} GB')

        print(f"Allocated memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"Reserved memory: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        print(f"Reserved memory (Max): {torch.cuda.max_memory_reserved() / 1024**2:.2f} MB")

        print(torch.cuda.memory_summary())
    else:
        print("CUDA not available.")