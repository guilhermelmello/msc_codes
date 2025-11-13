#!/bin/bash
#PBS -N bpe10k
#PBS -q par128
#PBS -j oe
#PBS -o logs/bpe10k.log

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
python train.py \
    --model BPE \
    --vocab-size 10000 \
    --output-dir models/bpe10k


deactivate
echo "Ending Time: $(date)"
