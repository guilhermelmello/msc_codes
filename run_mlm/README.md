# Run MLM

This folder contains every file and script to train a model with MLM loss. The [run_mlm_no_trainer.py](run_mlm_no_trainer.py) is a modified version of the original script provided by HuggingFace. The modifications were placed to adapt the script to process the Carolina Corpus.

Scripts:
- [From Scratch](script_config.sh): used to train a new MLM model from scratch using a config.json as input.
- [From Checkpoint](script_checkpoint.sh): usedo to train an MLM model from a saved checkpoint.