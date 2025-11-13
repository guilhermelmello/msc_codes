"""
This script creates a Byte-Pair Encoding tokenizer
for brazilian portuguese based on Carolina corpus.
"""
from datasets import load_dataset
from enum import Enum
from transformers import PreTrainedTokenizerFast
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    Tokenizer,
    trainers,
)

import argparse
import os


MIN_TOKEN_FREQ = 3
MAX_LENGTH = 2048

PAD_TOKEN = '[PAD]'
SEP_TOKEN = '[SEP]'
EOS_TOKEN = '[EOS]'
UNK_TOKEN = '[UNK]'
SPECIAL_TOKENS = [PAD_TOKEN, SEP_TOKEN, EOS_TOKEN, UNK_TOKEN]


class ModelType(str, Enum):
    BPE = 'BPE'
    UNIGRAM = 'UNIGRAM'


def read_arguments():
    """
    Read command-line arguments for model finetuning.

    Returns
    -------
    argparse.Namespace
        An object containing:
            - model (str): model type to train (BPE or UNIGRAM)
            - vocab_size (Optional[int]): maximum vocabulary size (default: 30000)
            - output_dir (Optional[str]): path to save pre-trained tokenizer (default: ./tokenizer)
    """
    parser = argparse.ArgumentParser(
        description="Create a new pre-trained byte-level BPE tokenizer."
    )

    parser.add_argument(
        "--model-type",
        required=True,
        type=ModelType,
        choices=list(ModelType),
        help="Tokenizer model type to train."
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


def get_tokenizer_model(model_type: ModelType):
    if model_type == ModelType.BPE:
        return models.BPE()
    elif model_type == ModelType.UNIGRAM:
        return models.Unigram()

    raise ValueError


def get_tokenizer_trainer(model_type: ModelType, vocab_size: int):
    bytes_alphabet = pre_tokenizers.ByteLevel().alphabet()

    if model_type == ModelType.BPE:
        return trainers.BpeTrainer(
            initial_alphabet=bytes_alphabet,
            special_tokens=SPECIAL_TOKENS,
            min_frequency=MIN_TOKEN_FREQ,
            vocab_size=vocab_size,
            show_progress=True,
        )
    elif model_type == ModelType.UNIGRAM:
        return trainers.UnigramTrainer(
            initial_alphabet=bytes_alphabet,
            special_tokens=SPECIAL_TOKENS,
            shrinking_factor=0.05,
            vocab_size=vocab_size,
            unk_token=UNK_TOKEN,
            # n_sub_iterations=,
        )

    raise ValueError


def main(model_type: ModelType, vocab_size: int, output_dir: str):
    # load dataset
    print("Loading the dataset...")
    dataset = load_dataset(
        "carolina-c4ai/corpus-carolina",
        split="corpus",
        revision='v2.0.1',
    )
    print(dataset)

    print(f'Creating the tokenization pipeline for {model_type}')
    print("Model setup.")
    tokenizer_model = get_tokenizer_model(model_type)
    tokenizer = Tokenizer(model=tokenizer_model)
    print('Selected Model:')
    print(tokenizer_model)

    print("Normalizer setup.")
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.Strip(),
        normalizers.NFKD()
    ])

    print("Pre-Tokenizer setup.")
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=True,
        use_regex=True,
    )

    print("Trainer setup.")
    trainer = get_tokenizer_trainer(
        model_type=model_type,
        vocab_size=vocab_size
    )
    print('Selected Trainer:')
    print(trainer)

    print('Training Tokenizer')
    def get_batch_iterator(dataset, batch_size):
        for i in range(0, len(dataset), batch_size):
            yield dataset[i: i+batch_size]['text']

    batch_it = get_batch_iterator(dataset, 1000)
    tokenizer.train_from_iterator(batch_it, trainer=trainer)

    print("Post-Processor setup.")
    tokenizer.post_processor = processors.Sequence([
        processors.ByteLevel(trim_offsets=True),
        processors.TemplateProcessing(
            single=f'$A:0 {EOS_TOKEN}:0',
            pair=f'$A:0 {SEP_TOKEN}:0 $B:0 {EOS_TOKEN}:0',
            special_tokens=[
                (SEP_TOKEN, tokenizer.token_to_id(SEP_TOKEN)),
                (EOS_TOKEN, tokenizer.token_to_id(EOS_TOKEN)),
            ]
        ),
    ])

    print("Enabling truncation.")
    tokenizer.enable_truncation(max_length=MAX_LENGTH)

    print("Decoder setup.")
    tokenizer.decoder = decoders.ByteLevel()

    print('Saving tokenizer at', output_dir)
    os.makedirs(output_dir, exist_ok=True)
    tokenizer_fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        model_max_length=2048,
        add_prefix_space=True,
        pad_token=PAD_TOKEN,
        eos_token=EOS_TOKEN,
        sep_token=SEP_TOKEN,
        unk_token=UNK_TOKEN,
    )
    tokenizer_fast.save_pretrained(output_dir)



if __name__ == '__main__':
    args = read_arguments()
    main(
        model_type=args.model_type,
        vocab_size=args.vocab_size,
        output_dir=args.output_dir,
    )