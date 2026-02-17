#!/bin/bash
#PBS -N qbase-8k
#PBS -q umagpu
#PBS -e logs/train-qwen-pt-base-8k-full-2.err
#PBS -o logs/train-qwen-pt-base-8k-full-2.out

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


echo "RUNNING QWEN-BASE UNIGRAM 8K"
echo "Sending data to /work/gmello"
DATASET_PATH="datasets/clm-1024-unigram-pt-8k/"
DATASET_WPATH="/work/gmello/$DATASET_PATH"

mkdir -p $DATASET_WPATH
rsync -av $DATASET_PATH $DATASET_WPATH

# echo "Running python script"
# python src/trainer.py \
#     --dataset-path $DATASET_WPATH \
#     --tokenizer-name guilhermelmello/tokenizer-unigram-pt-8k \
#     --init-mode base-config \
#     --model-name Qwen/Qwen3-0.6B \
#     --save-path ./models/qwen-pt-base-unigram8k-full-1 \
#     --batch-size 32 \
#     --num-epochs 3 \
#     --num-workers 16 \
#     --learning-rate 0.001 \
#     --weight-decay 0.1 \
#     --warmup-steps 10000

echo "Running python script"
python src/trainer.py \
    --dataset-path $DATASET_WPATH \
    --init-mode pretrained \
    --tokenizer-name ./models/qwen-pt-base-unigram8k-full-1 \
    --model-name ./models/qwen-pt-base-unigram8k-full-1 \
    --save-path ./models/qwen-pt-base-unigram8k-full-2 \
    --batch-size 32 \
    --num-epochs 3 \
    --num-workers 16 \
    --learning-rate 0.001 \
    --weight-decay 0.1 \
    --warmup-steps 10000



echo "RUNNING QWEN-BASE BPE 8K"
echo "Sending data to /work/gmello"
DATASET_PATH="datasets/clm-1024-bpe-pt-8k/"
DATASET_WPATH="/work/gmello/$DATASET_PATH"

mkdir -p $DATASET_WPATH
rsync -av $DATASET_PATH $DATASET_WPATH

# echo "Running python script"
# python src/trainer.py \
#     --dataset-path $DATASET_WPATH \
#     --tokenizer-name guilhermelmello/tokenizer-bpe-pt-8k \
#     --init-mode base-config \
#     --model-name Qwen/Qwen3-0.6B \
#     --save-path ./models/qwen-pt-base-bpe8k-full-1 \
#     --batch-size 32 \
#     --num-epochs 3 \
#     --num-workers 16 \
#     --learning-rate 0.001 \
#     --weight-decay 0.1 \
#     --warmup-steps 10000

echo "Running python script"
python src/trainer.py \
    --dataset-path $DATASET_WPATH \
    --init-mode pretrained \
    --tokenizer-name ./models/qwen-pt-base-bpe8k-full-1 \
    --model-name ./models/qwen-pt-base-bpe8k-full-1 \
    --save-path ./models/qwen-pt-base-bpe8k-full-2 \
    --batch-size 32 \
    --num-epochs 3 \
    --num-workers 16 \
    --learning-rate 0.001 \
    --weight-decay 0.1 \
    --warmup-steps 10000


deactivate
echo "Ending Time: $(date)"
