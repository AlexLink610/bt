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

# Usage:
#   sbatch.tinygpu run_vggt.sh <image_subdir> <output_base> [viewtxt_filename]
#
# Examples:
#   sbatch.tinygpu run_vggt.sh tree_02/images t02 selected_64views_360deg.txt
#   sbatch.tinygpu run_vggt.sh old_room/color old_room
#
# $1 = image subdirectory under /home/woody/iwi9/iwi9146h/data/
# $2 = output base name under /home/woody/iwi9/iwi9146h/output_vggt/
# $3 = (optional) viewtxt filename under /home/woody/iwi9/iwi9146h/viewtxts/
#      if omitted, all images in image_dir are used

if [ -z "$3" ]; then
    python /home/woody/iwi9/iwi9146h/run_vggt.py \
        --image_dir /home/woody/iwi9/iwi9146h/data/$1 \
        --output    /home/woody/iwi9/iwi9146h/output_vggt/$2
else
    python /home/woody/iwi9/iwi9146h/run_vggt.py \
        --image_dir  /home/woody/iwi9/iwi9146h/data/$1 \
        --output     /home/woody/iwi9/iwi9146h/output_vggt/$2 \
        --image_list /home/woody/iwi9/iwi9146h/viewtxts/$3
fi
