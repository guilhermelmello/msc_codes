from itertools import chain
from transformers import AutoConfig, AutoModelForCausalLM

def get_tokenize_fn(tokenizer):
    def tokenize_dataset(batch):
        encoded = tokenizer(
            batch['text'],
            padding=False,
            truncation=False,
            return_overflowing_tokens=True,
            return_token_type_ids=False,
            add_special_tokens=True,
        )

        # replace EOS by SEP tokens. Sequences will be concatenated and
        # EOS is not meant to appear on the left side of any token.
        for id_seq in encoded['input_ids']:
            id_seq[-1] = tokenizer.sep_token_id

        return encoded
    return tokenize_dataset


def get_group_texts_fn(tokenizer, max_length):
    def group_texts(batch):
        # convert batch into a single sentence
        concat_batch = {k: list(chain(*batch[k])) for k in ['input_ids', 'attention_mask']}
        new_batch = {
            'input_ids': [],
            'attention_mask': []
        }

        # create chuncks of max_lenght
        total_tokens = len(concat_batch['input_ids'])
        for i in range(0, total_tokens, max_length):
            for k, v in concat_batch.items():
                new_batch[k].append(v[i:i+max_length])

        # the last sentence may not have max_length, and the [SEP]
        # token must be replaced by [EOS] token and padded on the
        # left side as well. [EOS] tokens will be used as sequence
        # representation and must be the last token on the batch.
        last_seq_len = len(new_batch['input_ids'][-1])
        if last_seq_len < max_length:
            pad_token_id = tokenizer.pad_token_id
            missing_tokens = max_length - last_seq_len
            new_batch['input_ids'][-1] = [pad_token_id] * missing_tokens + new_batch['input_ids'][-1]
            new_batch['attention_mask'][-1] = [0] * missing_tokens + new_batch['attention_mask'][-1]
            new_batch['input_ids'][-1][-1] = tokenizer.eos_token_id

        return new_batch
    return group_texts


def create_clm_dataset(dataset, tokenizer, max_length):
    print('Dataset Tokenization')
    tokenize_dataset = get_tokenize_fn(tokenizer)
    clm_dataset = dataset.map(
        tokenize_dataset,
        remove_columns=dataset['train'].column_names,
        batched=True,
    )

    print(f'Reshaping dataset into sequences of {max_length}')
    group_texts = get_group_texts_fn(tokenizer, max_length)
    clm_dataset = clm_dataset.map(
        group_texts,
        batched=True,
        remove_columns=clm_dataset['train'].column_names
    )
    return clm_dataset


def load_model_from_config(model_name, tokenizer):
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
