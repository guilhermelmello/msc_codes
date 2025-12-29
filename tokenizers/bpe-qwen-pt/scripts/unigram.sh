#!/bin/bash
#PBS -N unigram5k
#PBS -q par128
#PBS -j oe
#PBS -o logs/unigram.log

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
echo "Running with 5K"
python train.py \
    --model-type UNIGRAM \
    --vocab-size 5000 \
    --output-dir models/unigram5k-nfc \
    --unicode-norm NFC


echo "Running with 8K"
python train.py \
    --model-type UNIGRAM \
    --vocab-size 8000 \
    --output-dir models/unigram8k-nfc \
    --unicode-norm NFC


echo "Running with 15K"
python train.py \
    --model-type UNIGRAM \
    --vocab-size 15000 \
    --output-dir models/unigram15k-nfc \
    --unicode-norm NFC


echo "Running with 30K"
python train.py \
    --model-type UNIGRAM \
    --vocab-size 30000 \
    --output-dir models/unigram30k-nfc \
    --unicode-norm NFC


echo "Running with 50K"
python train.py \
    --model-type UNIGRAM \
    --vocab-size 50000 \
    --output-dir models/unigram50k-nfc \
    --unicode-norm NFC


deactivate
echo "Ending Time: $(date)"
