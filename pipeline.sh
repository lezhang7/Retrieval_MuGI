#!/bin/bash
#SBATCH --job-name=mugi
#SBATCH --partition=long                           # Ask for unkillable job
#SBATCH --cpus-per-task=4                             # Ask for 2 CPUs
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100l:1
#SBATCH --ntasks-per-node=1                                  # Ask for 1 GPU
#SBATCH --mem=128G           
#SBATCH --reservation=ubuntu2204                             # Ask for 10 GB of RAM
#SBATCH --time=48:00:00                                   
#SBATCH --output=/home/mila/l/le.zhang/scratch/slurm_logs/mugi-%j.txt
#SBATCH --error=/home/mila/l/le.zhang/scratch/slurm_logs/mugi-%j.txt 

module load miniconda/3
conda init
conda activate openflamingo

for model in  Qwen/Qwen1.5-72B-Chat-AWQ Qwen/Qwen1.5-7B-Chat-AWQ Qwen/Qwen1.5-14B-Chat-AWQ 01-ai/Yi-34B-Chat-4bits 01-ai/Yi-6B-Chat-4bits
do
    python mugi.py --llm $model --irmode mugisparse --mode concat
done