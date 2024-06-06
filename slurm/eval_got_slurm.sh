#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=eval_dialoconan_got
#SBATCH --cpus-per-task=18
#SBATCH --time=00:10:00
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=50G

# This loads the anaconda virtual environment with our packages
source $HOME/.bashrc
conda activate graph-based-hs-cn
cd ./../src

# Run the actual experiment.
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 4 \
  train_model.py \
  --data_root ./../data --dataset DIALOCONAN --got_root got/ \
  --output_dir ./../experiments/DIALOCONAN \
  --model declare-lab/flan-alpaca-base \
  --epoch 50 --lr 5e-5 --bs 8 --eval_bs 16 \
  --input_len 512 --output_len 256 \
  --evaluate_dir ./../experiments/DIALOCONAN/declare-lab-flan-alpaca-base_lr5e-05_bs32_op256_ep50_useGTrue_2024-06-01-04-17/checkpoint-3500
