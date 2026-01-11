#!/bin/bash
#PBS -N unigram50k
#PBS -q par128
#PBS -j oe
#PBS -o logs/unigram50k.log

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
    --model-type UNIGRAM \
    --vocab-size 50000 \
    --output-dir models/unigram50k


deactivate
echo "Ending Time: $(date)"
