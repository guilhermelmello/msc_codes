#!/bin/bash
#PBS -N teste_job
#PBS -q testegpu
#PBS -e logs/assin-rte/bert-base-uncased.err
#PBS -o logs/assin-rte/bert-base-uncased.out

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
    --model-name google-bert/bert-base-uncased \
    --save-dir models/assin-rte/bert-base-uncased/ \
    --n-trials 3 \
    --n-epochs 5 \
    --seed 42


deactivate
echo "Ending Time: $(date)"
