#!/bin/bash
#PBS -N loss
#PBS -q testegpu
#PBS -e logs/loss-qwen3-06b.err
#PBS -o logs/loss-qwen3-06b.out

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


python src/loss_evaluation.py


deactivate
echo "Ending Time: $(date)"
