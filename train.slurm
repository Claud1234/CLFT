#!/bin/bash 
#SBATCH --job-name=train-1                ### job name
#SBATCH -n 1
#SBATCH -p gpu                  ### use gpu
#SBATCH --gres=gpu:A100:1               ### specifying the GPU type
#SBATCH --constraint="A100-80"
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH -t 5-0                  ### time limit 
#SBATCH --mail-type=ALL

## run job    
./train.sh $1
