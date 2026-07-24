#!/bin/bash
#PBS -N a2-rte
#PBS -q miggpu24h
#PBS -e logs/assin2-rte/albertina-100m/seed-43.err
#PBS -o logs/assin2-rte/albertina-100m/seed-43.out

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
    --task-name assin2-rte \
    --model-name PORTULAN/albertina-100m-portuguese-ptbr-encoder \
    --save-dir models/assin2-rte/albertina-100m/43 \
    --hp-learning-rate 5e-3 5e-4 5e-5 5e-6 \
    --hp-batch-size 8 16 32 \
    --num-hp-trials 12 \
    --num-hp-epochs 5 \
    --num-training-epochs 10 \
    --seed 43


deactivate
echo "Ending Time: $(date)"
