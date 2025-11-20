from . import trainer
from .utils import create_clm_dataset
from .utils import load_model_from_config
from datasets import load_dataset
from transformers import AutoTokenizer
from datasets import Dataset

import argparse
import gc
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain a new model from scratch")

    parser.add_argument(
        "--tokenizer-name",
        type=str,
        required=True,
        help="Name or path of the tokenizer to load."
    )

    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Name or path of the model config to load."
    )

    parser.add_argument(
        "--max-seq-length",
        type=int,
        required=True,
        help="Maximum sequence length for training."
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        required=True,
        help="Batch size for training."
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        required=True,
        help="Learning rate."
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        required=True,
        help="Weight decay for optimizer."
    )

    parser.add_argument(
        "--warmup-steps",
        type=int,
        required=True,
        help="Number of warmup steps for learning rate scheduler."
    )

    parser.add_argument(
        "--n-epochs",
        type=int,
        required=True,
        help="Number of epochs for training."
    )

    return parser.parse_args()


def main(
    tokenizer_name,
    model_name,
    max_seq_length,
    batch_size,
    learning_rate,
    weight_decay,
    warmup_steps,
    n_epochs,
):
    print('>>> Loading Raw Dataset')
    raw_dataset = load_dataset(
        'carolina-c4ai/corpus-carolina',
        revision='v2.0.1',
        split='corpus',
    )
    assert raw_dataset is Dataset
    raw_dataset = raw_dataset.train_test_split(test_size=0.1, seed=42)
    print(raw_dataset)

    print('>>> Loading Tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    print(tokenizer)

    print('>>> Creating dataset for Causal Language Model')
    clm_dataset = create_clm_dataset(raw_dataset, tokenizer, max_seq_length)
    print(clm_dataset)

    print('>>> Loading model from config')
    model = load_model_from_config(model_name, tokenizer)
    model_size = sum(t.numel() for t in model.parameters())
    print(f"Model Size: {model_size/1000**2:.1f}M parameters")

    print('>>> Training model from scratch')
    model = trainer.train_clm(
        model=model,
        tokenizer=tokenizer,
        train_dataset=clm_dataset['train'],
        validation_dataset=clm_dataset['test'],
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        n_epochs=n_epochs,
    )

    torch.cuda.empty_cache()
    gc.collect()
    


if __name__ == "__main__":
    args = parse_args()
    main(
        tokenizer_name=args.tokenizer_name,
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        n_epochs=args.n_epochs,
    )
