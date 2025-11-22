from datasets import load_dataset, DatasetDict
from itertools import chain
from transformers import AutoTokenizer

import argparse
import os


def _parse_arguments():
    parser = argparse.ArgumentParser(description="Dataset Tokenization")
    parser.add_argument(
        "--tokenizer-name", type=str, required=True,
        help="Name or path of the tokenizer to load."
    )
    parser.add_argument(
        "--save-path", type=str, required=True,
        help="Local path to save CLM ready dataset."
    )
    parser.add_argument(
        "--max-seq-length", type=int, required=True,
        help="Maximum sequence length for training."
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility (default: None)."
    )
    parser.add_argument(
        "--num-proc", type=int, default=None,
        help="Number of process for parallel processing."
    )
    parser.add_argument(
        "--batch-size", type=int, default=1000,
        help="Batch size."
    )
    return parser.parse_args()


def _get_tokenize_fn(tokenizer):
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


def _get_group_texts_fn(tokenizer, max_seq_length):
    def group_texts(batch):
        # convert batch into a single sentence
        concat_batch = {k: list(chain(*batch[k])) for k in ['input_ids', 'attention_mask']}
        new_batch = {
            'input_ids': [],
            'attention_mask': []
        }

        # create chuncks of max_lenght
        total_tokens = len(concat_batch['input_ids'])
        for i in range(0, total_tokens, max_seq_length):
            for k, v in concat_batch.items():
                new_batch[k].append(v[i:i+max_seq_length])

        # the last sentence may not have max_seq_length, and the [SEP]
        # token must be replaced by [EOS] token and padded on the
        # left side as well. [EOS] tokens will be used as sequence
        # representation and must be the last token on the batch.
        last_seq_len = len(new_batch['input_ids'][-1])
        if last_seq_len < max_seq_length:
            pad_token_id = tokenizer.pad_token_id
            missing_tokens = max_seq_length - last_seq_len
            new_batch['input_ids'][-1] = [pad_token_id] * missing_tokens + new_batch['input_ids'][-1]
            new_batch['attention_mask'][-1] = [0] * missing_tokens + new_batch['attention_mask'][-1]
            new_batch['input_ids'][-1][-1] = tokenizer.eos_token_id

        return new_batch
    return group_texts


def log_dataset(dataset):
    print(dataset)
    if isinstance(dataset, DatasetDict):
        print('Dataset Fingerprint:')
        for split in dataset:
            print(split, dataset[split]._fingerprint)
    else:
        print('Dataset Fingerprint:', dataset._fingerprint)


def create_clm_dataset(tokenizer_name, max_seq_length, seed=None, num_proc=None, batch_size=1000):
    print('>>> Loading Raw Dataset')
    raw_dataset = load_dataset(
        'carolina-c4ai/corpus-carolina',
        revision='v2.0.1',
        split='corpus',
    )
    log_dataset(raw_dataset)

    # creating validation split
    raw_dataset = raw_dataset.train_test_split(test_size=0.1, seed=seed)
    val_ds = raw_dataset.pop('test')
    raw_dataset['validation'] = val_ds
    log_dataset(raw_dataset)

    print('>>> Loading Tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    print(tokenizer)

    print('>>> Creating dataset for Causal Language Model')
    print('Dataset Tokenization')
    tokenize_dataset = _get_tokenize_fn(tokenizer)
    clm_dataset = raw_dataset.map(
        tokenize_dataset,
        remove_columns=raw_dataset['train'].column_names,
        batch_size=batch_size,
        num_proc=num_proc,
        batched=True,
    )
    log_dataset(clm_dataset)

    print(f'Reshaping dataset into sequences of {max_seq_length}')
    group_texts = _get_group_texts_fn(tokenizer, max_seq_length)
    clm_dataset = clm_dataset.map(
        group_texts,
        remove_columns=clm_dataset['train'].column_names,
        batch_size=batch_size,
        num_proc=num_proc,
        batched=True,
    )
    log_dataset(clm_dataset)
    return clm_dataset


if __name__ == '__main__':
    args = _parse_arguments()
    dataset = create_clm_dataset(
        tokenizer_name=args.tokenizer_name,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        num_proc=args.num_proc,
        seed=args.seed,
    )

    print('saving dataset at', args.save_path)
    os.makedirs(args.save_path)
    dataset.save_to_disk(args.save_path)
