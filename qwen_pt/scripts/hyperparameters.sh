#!/bin/bash
#PBS -N hpsearch
#PBS -q testegpu
#PBS -e logs/hpsearch.err
#PBS -o logs/hpsearch.out

echo "Staring Time: $(date)"
echo "Root directory $PBS_O_WORKDIR"
echo "Node: $(hostname)"
cd $PBS_O_WORKDIR

unset CUDA_VISIBLE_DEVICES
source .venv/bin/activate

# set HF to offline mode
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_EVALUATE_OFFLINE=1
export TRANSFORMERS_OFFLINE=1


echo "Running Hyperparameters Search"
python src/hyperparameters.py \
    --dataset-path datasets/clm-1024-unigram-pt-10k/validation \
    --tokenizer-name guilhermelmello/tokenizer-unigram-pt-10k \
    --model-name Qwen/Qwen3-0.6B \
    --weight-decay 0.1 \
    --warmup-steps 100 \
    --n-trials 3 \
    --n-epochs 5 \
    --seed 42


deactivate
echo "Ending Time: $(date)"
