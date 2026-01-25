"""
Creates a new tokenizer from an existing (old) tokenizer
for brazilian portuguese based on Carolina corpus.
"""
from datasets import load_dataset
from transformers import AutoTokenizer

import argparse
import os


def read_arguments():
    """
    Read command-line arguments for model finetuning.

    Returns
    -------
    argparse.Namespace
        An object containing:
            - model_name (str): Path or name of a pretrained tokenizer
            - vocab_size (Optional[int]): maximum vocabulary size (default: 30000)
            - output_dir (Optional[str]): path to save pre-trained tokenizer (default: ./tokenizer)
    """
    parser = argparse.ArgumentParser(
        description="Create a new tokenizer from an already pretrained."
    )

    parser.add_argument(
        "--model-name",
        required=True,
        type=str,
        help="Path or name of the old tokenizer"
    )

    parser.add_argument(
        "--vocab-size",
        type=int,
        default=30000,
        help="Maximum vocabulary size (default: 30000)."
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default='./tokenizer',
        help="Directory to save model pretrained tokenizer (default: ./tokenizer)."
    )

    args = parser.parse_args()
    return args


def get_training_corpus():
    dataset = load_dataset(
        "carolina-c4ai/corpus-carolina",
        split="corpus",
        revision='v2.0.1',
        trust_remote_code=True,
    )
    for example in dataset['text']:
        yield example


def main(
    model_name: str,
    vocab_size: int,
    output_dir: str,
):
    # load dataset
    print("Loading the dataset...")
    dataset = get_training_corpus()

    print(f'Loading pretrained tokenizer: {model_name}')
    old_tokenizer = AutoTokenizer.from_pretrained(model_name)

    print('Training the new tokenizer')
    tokenizer = old_tokenizer.train_new_from_iterator(dataset, vocab_size)

    print('Saving tokenizer at', output_dir)
    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save_pretrained(output_dir)


if __name__ == '__main__':
    args = read_arguments()
    main(
        model_name=args.model_name,
        vocab_size=args.vocab_size,
        output_dir=args.output_dir,
    )
