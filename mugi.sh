#!/bin/bash
#SBATCH --job-name=mugi_pipeline
#SBATCH --partition=long                           # Ask for unkillable job
#SBATCH --cpus-per-task=4                             # Ask for 2 CPUs
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100l.3
#SBATCH --ntasks-per-node=1                                  # Ask for 1 GPU
#SBATCH --mem=64G           
#SBATCH --reservation=ubuntu2204                             # Ask for 10 GB of RAM
#SBATCH --time=4:00:00                                   
#SBATCH --output=/home/mila/l/le.zhang/scratch/slurm_logs/muginfc-%j.txt
#SBATCH --error=/home/mila/l/le.zhang/scratch/slurm_logs/muginfc-%j.txt 

module load miniconda/3
conda init
conda activate openflamingo

# 01-ai/Yi-34B-Chat-4bits 01-ai/Yi-6B-Chat-4bits Qwen/Qwen1.5-7B-Chat-AWQ Qwen/Qwen1.5-14B-Chat-AWQ Qwen/Qwen1.5-72B-Chat-AWQ gpt
for irmode in mugisparse
do
    python mugi.py --llm gpt --irmode $irmode
done