#!/bin/bash
#SBATCH -w gnode037
#SBATCH --gres=gpu:1
#SBATCH --mincpus=10
#SBATCH --mem-per-cpu=2G
#SBATCH --output=output.txt
#SBATCH --time=12:00:00

source /home2/akshitsinha28/miniconda3/bin/activate

conda activate unlearn

python /home2/akshitsinha28/AdversarialUnlearning/hyperparam.py