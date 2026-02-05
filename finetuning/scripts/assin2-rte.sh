#!/bin/bash
#PBS -N a2-rte
#PBS -q miggpu
#PBS -e logs/assin2-rte/albertina-100m.err
#PBS -o logs/assin2-rte/albertina-100m.out

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
    --save-dir models/assin2-rte/albertina-100m \
    --num-hp-trials 10 \
    --num-hp-epochs 3 \
    --num-training-epochs 10 \
    --seed 42


deactivate
echo "Ending Time: $(date)"
