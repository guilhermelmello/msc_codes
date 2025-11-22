from transformers import AutoConfig, AutoModelForCausalLM


def load_clm_from_config(model_name, tokenizer):
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
