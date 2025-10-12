#!/bin/bash -l

#SBATCH -p arandu
#SBATCH -n 64
#SBATCH --gres=gpu:0
#SBATCH --output=main.log


# Print the name of the worker node to the output file
echo Running on $HOSTNAME
echo "Starting time: $(TZ=America/Sao_Paulo date)"

# move project to output folder
cp -r /home/gmello/msc_codes/tokenizer_bpe /output/gmello

# creates HF cache directory if does not exists
# this directory is intended to be manualy deleted.
mkdir -p /output/gmello/hf_cache

# creates an output directory inside project to save new files
mkdir -p /output/gmello/tokenizer_bpe/results

# Call Docker and run the code
docker run \
    --ipc=host --rm \
    --user "$(id -u):$(id -g)" \
    -v /output/gmello/tokenizer_bpe:/workspace/tokenizer_bpe \
    -v /output/gmello/hf_cache:/workspace/hf_cache \
    -w /workspace/tokenizer_bpe \
    gmello_tokenizers:1.0 \
    python main.py

# move project results back to home directory
cp -r /output/gmello/tokenizer_bpe/results /home/gmello/msc_codes/tokenizer_bpe
rm -r /output/gmello/tokenizer_bpe


echo "Done"
echo "Ending time: $(TZ=America/Sao_Paulo date)"
