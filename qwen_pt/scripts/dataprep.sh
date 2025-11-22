#!/bin/bash
#PBS -N dataprep
#PBS -q testes
#PBS -l nodes=1:ppn=4
#PBS -e logs/dataprep.err
#PBS -o logs/dataprep.out

NUM_PROC=4

echo "Staring Time: $(date)"
echo "Root directory $PBS_O_WORKDIR"
echo "Node: $(hostname)"
cd $PBS_O_WORKDIR

source .venv/bin/activate

# set HF to offline mode
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_EVALUATE_OFFLINE=1
export TRANSFORMERS_OFFLINE=1


echo "Running Unigram Tokenizer"
python src/dataprep.py \
    --save-path datasets/clm-1024-unigram-pt-10k \
    --tokenizer-name guilhermelmello/tokenizer-unigram-pt-10k \
    --max-seq-length 1024 \
    --batch-size 256 \
    --num-proc $NUM_PROC \
    --seed 42

echo "Running BPE Tokenizer"
python src/dataprep.py \
    --save-path datasets/clm-1024-bpe-pt-10k \
    --tokenizer-name guilhermelmello/tokenizer-bpe-pt-10k \
    --max-seq-length 1024 \
    --batch-size 512 \
    --num-proc $NUM_PROC \
    --seed 42


deactivate
echo "Ending Time: $(date)"
