#!/bin/bash
#PBS -N teste
#PBS -q testegpu
#PBS -e logs/teste.err
#PBS -o logs/teste.out

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


# run python script
echo "Running python script"
python main.py \
    --tokenizer-name guilhermelmello/tokenizer-unigram-pt-10k \
    --model-name Qwen/Qwen3-0.6B-Base \
    --max-seq-length 1024 \
    --batch-size 32 \
    --learning-rate 0.0001 \
    --weight-decay 0.01 \
    --warmup-steps 100 \
    --n-epochs 3 \
    --save-dir ./models/teste-qwen06B-unigram10k


deactivate
echo "Ending Time: $(date)"
