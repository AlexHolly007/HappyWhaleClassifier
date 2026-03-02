#!/bin/bash
#SBATCH -A bgmp
#SBATCH -p gpu
#SBATCH -t 08:00:00
#SBATCH -c 6
#SBATCH --mem=32G
#SBATCH -J part1
#SBATCH -o part1_%j.out
#SBATCH -e part1_%j.err

set -euo pipefail

source "$HOME/miniforge3/etc/profile.d/conda.sh" 2>/dev/null || \
source "$HOME/miniconda3/etc/profile.d/conda.sh"

conda activate /gpfs/home/alho/miniforge3/envs/torch_whale

python3 Part1.py
