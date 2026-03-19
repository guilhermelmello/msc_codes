#!/bin/bash
#PBS -N plue-rte
#PBS -q miggpu24h
#PBS -e logs/plue-rte/albertina-100m.err
#PBS -o logs/plue-rte/albertina-100m.out

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
    --model-name PORTULAN/albertina-100m-portuguese-ptbr-encoder \
    --save-dir models/plue-rte/albertina-100m \
    --hp-learning-rate 5e-3 5e-4 5e-5 5e-6 \
    --hp-batch-size 4 8 \
    --num-hp-trials 8 \
    --num-hp-epochs 5 \
    --num-training-epochs 10 \
    --skip-test-eval \
    --seed 42


deactivate
echo "Ending Time: $(date)"
