#!/bin/bash
#SBATCH --time=5:0:0
#SBATCH --nodes=1 
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=16 
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-v100

#Environment setup

source ~/.bashrc

cd /work/vision_lab/Cardiac_TAVR_Classification
source .venv/bin/activate
classegTrain -d 1 -f 0 -m sdkj -n long_run -ext tavr