#!/bin/bash
#PBS -N teste_job
#PBS -q testegpu
#PBS -j oe
#PBS -o logs/teste.out

echo "Staring Time: $(date)"
echo "Node: $(hostname)"
echo "CPUs: $(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc 2>/dev/null || echo 'N/D')"
echo "PBS Output Directory: $PBS_O_WORKDIR"
cd ~/msc_codes/finetuning
pwd

# python setup
module load python/3.10.10-gcc-9.4.0
source .venv/bin/activate

# set HF to offline mode
export HF_HUB_OFFLINE=1

# gpu usage
# nvidia-smi pmon >> logs/usogpu.out &

# run python script
echo "Running python script"
python main.py

# virtual environment
deactivate

echo "Ending Time: $(date)"
