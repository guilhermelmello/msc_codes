"""
This script creates a Byte-Pair Encoding tokenizer
for brazilian portuguese based on Carolina corpus.
"""
from datasets import load_dataset
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    Tokenizer,
    trainers,
)


# TRAINING ARGUMENTS
VOCAB_SIZE = 50000
MIN_TOKEN_FREQ = 3
OUTPUT_DIR = "./results"
SPECIAL_TOKENS = ["<pad>", "<cls>", "<sep>", "<mask>"]
DATASET_STREAMING = False   # to load the dataset in streaming mode


# ------- #
# DATASET #
# ------- #

# load dataset
dataset = load_dataset(
    "carolina-c4ai/corpus-carolina",
    split="corpus",
    revision='v1.2',
    streaming=DATASET_STREAMING
)

# creates batches from a dataset in streaming mode
def get_batch_from_streaming(dataset, batch_size):
    it = iter(dataset)
    try:
        while True:
            batch = [next(it)['text'] for _ in range(batch_size)]
            yield batch
    except StopIteration:
        # produces only full batches,
        # i.e., may drop the last batch.
        pass

# creates batches from a dataset in cache
def get_batch_from_cache(dataset, batch_size):
    for i in range(0, len(dataset), batch_size):
        # does not drop the last batch.
        yield dataset[i: i+batch_size]['text']


# ------------------ #
# TOKENIZER PIPELINE #
# ------------------ #

# Tokenization model
tokenizer = Tokenizer(model=models.BPE())

# Normalization: since it is a BPE tokenizer
# in byte level, only the accents are normalized,
# since accents have an important role in portuguese.
tokenizer.normalizer = normalizers.Sequence([
    normalizers.NFKD()
])

# Pre-tokenization: creates a representation for
# each byte as well as splitting into words. This
# is needed to create a byte level BPE. The argument
# `add_prefix_space` allows the tokenizer to make no
# distinction between a word that occurs in the begining
# or the middle of a sentence. Since words that occur
# in the middle of a sentence keep the preceding space,
# this is set to `True` as an attempt to reduce the size
# of the final vocabulary.
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
    add_prefix_space=True,
)

# Trainer: A Byte-Level BPE does not use an <UNK>
# token since it have a byte level vocabulary with
# all possible bytes. To create a true Byte-Level
# tokenizer, it is needed to pass the initial
# vocabulary. Otherwise, the model will create
# tokens that does not appear in the vocab and
# will need an <UNK> token.
alphabet = tokenizer.pre_tokenizer.alphabet()
trainer = trainers.BpeTrainer(
    special_tokens=SPECIAL_TOKENS,
    vocab_size=VOCAB_SIZE,
    initial_alphabet=alphabet,
    min_frequency=MIN_TOKEN_FREQ)

if DATASET_STREAMING:
    batch_it = get_batch_from_streaming(dataset, 1000)
else:
    batch_it = get_batch_from_cache(dataset, 1000)
tokenizer.train_from_iterator(batch_it, trainer=trainer)

# Post-processor: The tokenizer can receive a single or a
# pair of sentences. BERT tokenizer distinguishes to which
# sentence a token belongs by the 'token_type_id' embedding.
# This embedding have a limited vocabulary size (2) and each
# vector represents one of the sentences in the pair. This
# is used in the NSP (Next Sentence Prediction) training loss.
# Models like RoBERTa does not use this loss and does not need
# this token type id embedding.
# In this script, the token type id is set to 0 for sentences
# A and B because the model will use only MLM loss, but it
# can be changed if needed.
tokenizer.post_processor = processors.TemplateProcessing(
    single="<cls>:0 $A:0 <sep>:0",
    pair="<cls>:0 $A:0 <sep>:0 $B:0 <sep>:0",
    special_tokens=[
        ("<cls>", tokenizer.token_to_id("<cls>")),
        ("<sep>", tokenizer.token_to_id("<sep>")),
    ]
)

# By default, the tokenizer does not include a max length
# parameter and may create a missmatch with the model for
# some scripts that use this information to truncate the
# dataset. The following command enable the tokenizer to
# truncate at a max length.
tokenizer.enable_truncation(max_length=512)

# Decoder: byte level decoder are needed
# when using a byte level pretokenization,
tokenizer.decoder = decoders.ByteLevel()

# save the tokenizer's vocabulary
tokenizer.save(f'{OUTPUT_DIR}/tokenizer.json')


# --------------------------------------- #
#  HOW  TO  UPLOAD  TO  HUGGINGFACE  HUB  #
# --------------------------------------- #
# The tokenizers lib is an API to create
# and manipulate tokenization pipelines.
# Also, it is designed to create (train)
# new vocabularies. To use the tokenizer
# as a layer in a transformer model, it
# is necessary to convert (load) the
# vocab using the transformers lib as a
# PreTrainedTokenizer or
# PreTrainedTokenizerFast. Use the
# following code to do that:
# 
#  
# from transformers import PreTrainedTokenizerFast
# import huggingface_hub
# tokenizer = PreTrainedTokenizerFast(
#     tokenizer_file=f'{OUTPUT_DIR}/tokenizer.json',
#     pad_token='<pad>',
#     cls_token='<cls>',
#     sep_token='<sep>',
#     mask_token='<mask>',
#     model_max_length=512,
# )
# huggingface_hub.notebook_login()  # or .login(TOKEN)
# tokenizer.push_to_hub(HUB_REPO_PATH)
