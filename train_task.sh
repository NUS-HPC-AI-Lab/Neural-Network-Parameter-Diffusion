#!/bin/bash
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -p normal
#SBATCH --gres=gpu:a100:1
#SBATCH -t 10:00:00
#SBATCH --array=1-1
#SBATCH --mem=30GB
#SBATCH --job-name=4_2_train_tasks
#SBATCH --output=outfiles/logs/%x-%a.log
#SBATCH -e outfiles/errs/%x-%a.txt
#SBATCH --mail-type=END
#SBATCH --mail-user=qiyao@mit.edu

export CUDA_LAUNCH_BLOCKING=1
unset CUDA_VISIBLE_DEVICES

python task_training.py