#!/bin/bash -l

#SBATCH -p arandu
#SBATCH -n 64
#SBATCH --gres=gpu:7
#SBATCH --output=./logs/mlm_mbert_carolina_base.log


# Print the name of the worker node to the output file
echo Running on $HOSTNAME
echo "Starting time: $(TZ=America/Sao_Paulo date)"
echo "Running on GPU(s)" $CUDA_VISIBLE_DEVICES

# move project to output folder
rsync -av /home/gmello/msc_codes/run_mlm /output/gmello --exclude results

# creates a directory inside the project to save new files
mkdir -p /output/gmello/run_mlm/results/mlm_mbert_carolina_base

# Creates the HF cache directory if it does not exists.
# This directory is intended to persist between runs to
# save downloaded objects. This folder must be manually
# removed when every experiment is finished.
mkdir -p /output/gmello/hf_cache

# Call Docker and run the code
# --low_cpu_mem_usage \     -> Using low cpu usage causes an error.
docker run \
    --ipc=host --rm \
    --user "$(id -u):$(id -g)" \
    --gpus \"device=$CUDA_VISIBLE_DEVICES\" \
    -v /output/gmello/run_mlm:/workspace/run_mlm \
    -v /output/gmello/hf_cache:/workspace/hf_cache \
    -w /workspace/run_mlm \
    gmello_run_mlm:1.0 \
    accelerate launch run_mlm_no_trainer_4_31.py \
    \
    --model_name_or_path bert-base-multilingual-cased \
    --dataset_name carolina-c4ai/corpus-carolina \
    \
    --max_train_steps 100000 \
    --checkpointing_steps 5000 \
    --num_warmup_steps 24000 \
    --gradient_accumulation_steps 24 \
    --per_device_train_batch_size 46 \
    --per_device_eval_batch_size 46 \
    \
    --lr_scheduler_type linear \
    --mlm_probability 0.15 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --adam_beta1 0.90 \
    --adam_beta2 0.98 \
    \
    --preprocessing_num_workers 64 \
    --output_dir ./results/mlm_mbert_carolina_base \
    --seed 42 \
    --fp16


# move project results back to home and clean up the output
# folder. Only the results are saved to keep code consistency.
# Any change made in the /home files will be overwritten by the
# /output files which does not contains the last modifications.
# It includes the current log file.
echo "Saving project results to /home/gmello/msc_codes/run_mlm/results/mlm_mbert_carolina_base"
cp -r /output/gmello/run_mlm/results/mlm_mbert_carolina_base /home/gmello/msc_codes/run_mlm/results
rm -rf /output/gmello/run_mlm


echo "Done"
echo "Ending time: $(TZ=America/Sao_Paulo date)"
