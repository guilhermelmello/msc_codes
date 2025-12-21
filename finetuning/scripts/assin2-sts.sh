#!/bin/bash
#PBS -N a2-sts
#PBS -q miggpu24h
#PBS -e logs/assin2-sts/qwen-pt-bpe-10k.err
#PBS -o logs/assin2-sts/qwen-pt-bpe-10k.out

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
    --task-name assin2-sts \
    --model-name ~/msc_codes/qwen_pt/models/qwen-pt-bpe \
    --save-dir models/assin2-sts/qwen-pt-bpe-10k/ \
    --n-trials 10 \
    --n-epochs 5 \
    --seed 42


deactivate
echo "Ending Time: $(date)"
