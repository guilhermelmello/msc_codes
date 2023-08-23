# Run MLM

This folder contains every file and script used to train language models with MLM loss. The original script provided by HuggingFace was modified to process the [Carolina Corpus](https://huggingface.co/datasets/carolina-c4ai/corpus-carolina).

## Training Scripts:

After training the first model, a new version of the  training script was made available by HuggingFace. The latest version can be found [here](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_mlm_no_trainer.py). The scripts used to pretrain the models are listed below:

- [v4.29.0.dev0](run_mlm_no_trainer_4_29.py): requires _transformers_ 4.29.0.dev0 as minimal version;
- [v4.31.0.dev0](run_mlm_no_trainer_4_29.py): requires _transformers_ 4.31.0.dev0 as minimal version.

## Slurm Scripts

Shell sripts used to create and setup a Slurm job. Each shell script correspond to a different pretrained model.

- [mlm_carolina_base.sh](mlm_carolina_base.sh): used to train a new MLM model from scratch using the RoBERTa base configuration file with Carolina Corpus. The [log](logs/mlm_carolina_base.log) shows that the model could not finish the training because of an error. The last checkpoint was used to continue the training with following script.
- [mlm_carolina_base_ckpt.sh](mlm_carolina_base_ckpt.sh): continues the training from last saved checkpoint. The final model is available at [guilhermelmello/roberta_pt_8k_100k](https://huggingface.co/guilhermelmello/roberta_pt_8k_100k). Unfortunately, the log was accidentally overwritten and is not available.