#!/bin/bash
#SBATCH --job-name=lab2_quant
#SBATCH --account=csci_ga_3033-2026fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=lab2_%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
python -m pip install --user -q -r requirements.txt
python run_lab2.py
