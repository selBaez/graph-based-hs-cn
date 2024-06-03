#!/bin/bash

conda activate graph-based-hs-cn
cd src

python format_dataset.py

python dialogue_to_got.py

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 \
  train_model.py \
  --data_root ./../data --dataset DIALOCONAN --got_root got/ \
  --output_dir experiments/DIALOCONAN \
  --model declare-lab/flan-alpaca-base \
  --epoch 50 --lr 5e-5 --bs 8 --eval_bs 16 \
  --input_len 512 --output_len 256 \
  --use_generate --bf16

CUDA_VISIBLE_DEVICES=0 python \
  evaluate_model.py \
  --data_root ./../data --dataset DIALOCONAN -- got_root got/ \
  --output_dir experiments/DIALOCONAN \
  --model declare-lab/flan-alpaca-base \
  --epoch 50 --lr 5e-5 --bs 8 --eval_bs 16 \
  --input_len 512 --output_len 256 \
  --use_generate --bf16 \
  --evaluate_dir ./../experiments/declare-lab-flan-alpaca-base_lr5e-05_bs0_op64_ep2_useGFalse_2024-05-31-17-50
