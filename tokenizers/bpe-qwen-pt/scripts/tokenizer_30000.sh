#!/bin/bash
#PBS -N tokenizer
#PBS -q par128
#PBS -j oe
#PBS -o logs/tokenizer_30000.log

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
python tokenizer_train.py \
    --vocab-size 30000 \
    --output-dir models/tokenizer-30000


deactivate
echo "Ending Time: $(date)"
