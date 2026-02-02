#!/bin/bash
#PBS -N mviz
#PBS -q umagpu
#PBS -l walltime=00:30:00
#PBS -e logs/gpu_memory_viz.err
#PBS -o logs/gpu_memory_viz.out

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
python src/gpu_memory_viz.py


deactivate
echo "Ending Time: $(date)"
