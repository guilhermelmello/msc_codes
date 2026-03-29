#!/bin/bash
#PBS -N plue-submission
#PBS -q testegpu
#PBS -e logs/plue-submission.err
#PBS -o logs/plue-submission.out

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


MODEL_NAME=ttl-460m

echo "=================================="
echo "Evaluating $MODEL_NAME on RTE task"
python src/plue_submission.py \
    --task rte \
    --model-name models/plue-rte/$MODEL_NAME \
    --save-dir results/plue/$MODEL_NAME

echo "==================================="
echo "Evaluating $MODEL_NAME on WNLI task"
python src/plue_submission.py \
    --task wnli \
    --model-name models/plue-wnli/$MODEL_NAME \
    --save-dir results/plue/$MODEL_NAME


deactivate
echo "Ending Time: $(date)"
