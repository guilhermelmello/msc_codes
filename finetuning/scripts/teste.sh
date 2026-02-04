#!/bin/bash
#PBS -N teste
#PBS -q miggpu24h
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
    --task-name assin-rte \
    --model-name guilhermelmello/qwen-pt-bpe-8k \
    --num-hp-trials 3 \
    --num-hp-epochs 5 \
    --num-training-epochs 10 \
    --save-dir models/teste/ \
    --seed 42


deactivate
echo "Ending Time: $(date)"
