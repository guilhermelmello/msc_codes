#!/bin/bash
#PBS -N unigram10k
#PBS -q par128
#PBS -j oe
#PBS -o logs/unigram10k.unicode.log

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

echo "Running with NFC"
python train.py \
    --model-type UNIGRAM \
    --vocab-size 10000 \
    --output-dir models/unigram10k-nfc \
    --unicode-norm NFC


echo "Running with NFD"
python train.py \
    --model-type UNIGRAM \
    --vocab-size 10000 \
    --output-dir models/unigram10k-nfd \
    --unicode-norm NFD


echo "Running with NFKC"
python train.py \
    --model-type UNIGRAM \
    --vocab-size 10000 \
    --output-dir models/unigram10k-nfkc \
    --unicode-norm NFKC


echo "Running with NFKD"
python train.py \
    --model-type UNIGRAM \
    --vocab-size 10000 \
    --output-dir models/unigram10k-nfkd \
    --unicode-norm NFKD


deactivate
echo "Ending Time: $(date)"
