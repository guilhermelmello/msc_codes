#!/bin/bash
#PBS -N clm
#PBS -q umagpu
#PBS -e logs/train_clm.err
#PBS -o logs/train_clm.out

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


echo "Sending data to /work/gmello"
mkdir -p \
    /work/gmello/datasets/clm-1024-unigram-pt-10k/validation 
rsync -av \
    datasets/clm-1024-unigram-pt-10k/validation/ \
    /work/gmello/datasets/clm-1024-unigram-pt-10k/validation 


echo "Starting Causal Language Model Training."
python src/trainer.py


deactivate
echo "Ending Time: $(date)"
