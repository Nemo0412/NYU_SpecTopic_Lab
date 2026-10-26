#!/bin/bash
#SBATCH --job-name=efficient_ai_lab2
#SBATCH --account=torch_pr_111_general
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/home/ll5914/NYU_Lab/lab2_%j.log
set -euo pipefail
cd /home/ll5914/NYU_Lab
export PYTHONPATH="/home/ll5914/EAI_Course/.lab_deps${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export HF_HOME=/scratch/ll5914/.huggingface
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TIKTOKEN_CACHE_DIR=/home/ll5914/NYU_Lab/.tiktoken_cache
/scratch/ll5914/conda_envs/SVD/bin/python run_lab2.py
