#!/bin/bash
#SBATCH --job-name=test-1
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --constraint="A100-80"
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH -t 5-0
#SBATCH --mail-type=ALL

## run job
./test.sh $1
