#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=4
#SBATCH --job-name=train_dialoconan_txt
#SBATCH --cpus-per-task=18
#SBATCH --time=01:30:00
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=25G

# This loads the anaconda virtual environment with our packages
source $HOME/.bashrc
conda activate graph-based-hs-cn
cd ./../src

# Run the actual experiment.
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 \
  train_model_text.py \
  --data_root ./../data --dataset DIALOCONAN \
  --output_dir ./../experiments/DIALOCONAN \
  --model declare-lab/flan-alpaca-base \
  --epoch 50 --lr 5e-5 --bs 8 --eval_bs 16 \
  --input_len 512 --output_len 256 \
  --bf16
