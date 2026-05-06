#!/bin/bash -l
#SBATCH --job-name=vggt
#SBATCH --output=/home/woody/iwi9/iwi9146h/logs/log_vggt_%j.log
#SBATCH --error=/home/woody/iwi9/iwi9146h/logs/log_vggt_%j.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

module load python/3.12-conda
conda activate vggt

export PYTHONPATH=/home/woody/iwi9/iwi9146h/vggt:$PYTHONPATH

cd /home/woody/iwi9/iwi9146h/vggt
python /home/woody/iwi9/iwi9146h/run_vggt.py \
    --image_dir /home/woody/iwi9/iwi9146h/data/tree_02/images \
    --output /home/woody/iwi9/iwi9146h/output_vggt/t02 \
    --image_list /home/woody/iwi9/iwi9146h/viewtxts/$1