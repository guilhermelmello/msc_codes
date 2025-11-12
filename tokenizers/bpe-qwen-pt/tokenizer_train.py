"""
This script creates a Byte-Pair Encoding tokenizer
for brazilian portuguese based on Carolina corpus.
"""
from datasets import load_dataset
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
SPECIAL_TOKENS = [PAD_TOKEN, SEP_TOKEN, EOS_TOKEN]


def read_arguments():
    """
    Read command-line arguments for model finetuning.

    Returns
    -------
    argparse.Namespace
        An object containing:
            - vocab_size (Optional[int]): maximum vocabulary size (default: 30000)
            - output_dir (Optional[str]): path to save pre-trained tokenizer (default: ./tokenizer)
    """
    parser = argparse.ArgumentParser(
        description="Create a new pre-trained byte-level BPE tokenizer."
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


def main(vocab_size: int, output_dir: str):
    # load dataset
    print("Loading the dataset...")
    dataset = load_dataset(
        "carolina-c4ai/corpus-carolina",
        split="corpus",
        revision='v2.0.1',
    )
    print(dataset)

    print("Creating the tokenization pipeline...")
    print("Model setup.")
    tokenizer = Tokenizer(model=models.BPE())

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
    alphabet = tokenizer.pre_tokenizer.alphabet()
    trainer = trainers.BpeTrainer(
        initial_alphabet=alphabet,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=MIN_TOKEN_FREQ,
        vocab_size=vocab_size,
        show_progress=True,
    )

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
            single="$A:0 [EOS]:0",
            pair="$A:0 [SEP]:0 $B:0 [EOS]:0",
            special_tokens=[
                ("[SEP]", tokenizer.token_to_id("[SEP]")),
                ("[EOS]", tokenizer.token_to_id("[EOS]")),
            ]
        ),
    ])

    print("Enabling truncation.")
    tokenizer.enable_truncation(max_length=MAX_LENGTH)

    print("Decoder setup.")
    tokenizer.decoder = decoders.ByteLevel()


    os.makedirs(output_dir, exist_ok=True)
    # tokenizer_path = os.path.abspath(os.path.join(output_dir, f'tokenizer_{vocab_size}.json'))
    # print("Saving vocabulary to: ", tokenizer_path)
    # tokenizer.save(tokenizer_path)

    print('Saving tokenizer at', output_dir)
    tokenizer_fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        model_max_length=2048,
        add_prefix_space=True,
        pad_token='[PAD]',
        eos_token='[EOS]',
        sep_token='[SEP]',
    )
    tokenizer_fast.save_pretrained(output_dir)



if __name__ == '__main__':
    args = read_arguments()
    main(
        vocab_size=args.vocab_size,
        output_dir=args.output_dir,
    )