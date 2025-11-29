from transformers import AutoConfig, AutoModelForCausalLM

import torch


def initialize_clm_from_config(model_name, tokenizer):
    model_config = AutoConfig.from_pretrained(
        model_name,
        vocab_size=tokenizer.vocab_size,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        max_position_embeddings=tokenizer.model_max_length,
    )

    # create new model from config
    model = AutoModelForCausalLM.from_config(model_config)
    return model


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