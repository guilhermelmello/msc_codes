# Qwen3 for Brazilian Portuguese

Scripts to train a Qwen3 model from scratch, using Carolina Corpus (v2.0.1)

1. Download and cache assets for offline mode.
    ```bash
    python scripts/downloads.py
    ```
    - Every model, tokenizer and dataset must be included in this [script](scripts/downloads.py).

1. Prepare and cache the dataset for Causal Language Modeling training.
    ```bash
    python src/dataprep.py \
        --tokenizer-name guilhermelmello/tokenizer-unigram-pt-10k \
        --max-seq-length 1024 \
        --seed 42
    ```

