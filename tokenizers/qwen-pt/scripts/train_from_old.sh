#!/bin/bash
#PBS -N tokenizer-dv2
#PBS -q par128
#PBS -j oe
#PBS -o logs/tokenizer-dv2.log

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
python train_from_old.py \
    --model microsoft/deberta-v2-xlarge\
    --output-dir models/deberta-v2-5k \
    --vocab-size 5000

python train_from_old.py \
    --model microsoft/deberta-v2-xlarge\
    --output-dir models/deberta-v2-8k \
    --vocab-size 8000

python train_from_old.py \
    --model microsoft/deberta-v2-xlarge\
    --output-dir models/deberta-v2-10k \
    --vocab-size 10000

# python train_from_old.py \
#     --model microsoft/deberta-v2-xlarge\
#     --output-dir models/deberta-v2-15k \
#     --vocab-size 15000

# python train_from_old.py \
#     --model microsoft/deberta-v2-xlarge\
#     --output-dir models/deberta-v2-30k \
#     --vocab-size 30000

# python train_from_old.py \
#     --model microsoft/deberta-v2-xlarge\
#     --output-dir models/deberta-v2-50k \
#     --vocab-size 50000


deactivate
echo "Ending Time: $(date)"
