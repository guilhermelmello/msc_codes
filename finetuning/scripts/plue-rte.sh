#!/bin/bash
#PBS -N plue-rte
#PBS -q testegpu
#PBS -e logs/plue-rte/qwen-pt-base-bpe-8k.err
#PBS -o logs/plue-rte/qwen-pt-base-bpe-8k.out

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
    --task-name plue-rte \
    --model-name guilhermelmello/qwen-pt-base-bpe-8k \
    --save-dir models/plue-rte/qwen-pt-base-bpe-8k \
    --num-hp-trials 12 \
    --num-hp-epochs 5 \
    --num-training-epochs 10 \
    --seed 42


deactivate
echo "Ending Time: $(date)"
