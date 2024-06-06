#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=4
#SBATCH --job-name=preprocess_dialoconan_ekg
#SBATCH --cpus-per-task=18
#SBATCH --time=01:00:00
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=50G

# This loads the anaconda virtual environment with our packages
source $HOME/.bashrc
conda activate graph-based-hs-cn

# Run graphdb
cd ./../../graphdb-10.6.3/bin
nohup ./graphdb &

# Run the actual experiment.
cd ./../src
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 \
  dialogue_to_ekg.py \
  --data_root ./../data --dataset DIALOCONAN --splits test \
  --output_dir ./../data/DIALOCONAN/ekg/ \
  --start_step link \
  --stop_step format

# Stop graphdb
kill $(lsof -i:7200)
