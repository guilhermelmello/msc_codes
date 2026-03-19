#!/bin/bash
#PBS -N plue-wnli
#PBS -q miggpu24h
#PBS -e logs/plue-wnli/ttl-160m.err
#PBS -o logs/plue-wnli/ttl-160m.out

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
    --task-name plue-wnli \
    --model-name nicholasKluge/TeenyTinyLlama-160m \
    --save-dir models/plue-rte/ttl-160m \
    --hp-learning-rate 5e-3 5e-4 5e-5 5e-6 \
    --hp-batch-size 4 8 16 \
    --num-hp-trials 12 \
    --num-hp-epochs 5 \
    --num-training-epochs 10 \
    --skip-test-eval \
    --seed 42


deactivate
echo "Ending Time: $(date)"
