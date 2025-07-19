#!/bin/bash

#change to conda install location
CONDA_BASE=~/miniconda3

source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate lightgaussian

python LightGaussian/train_densify_prune.py -s dataset/360/garden -m output/Light_garden --eval --r 1200
python LightGaussian/train_densify_prune.py -s dataset/360/stump -m output/Light_stump --eval --r 1200
python LightGaussian/train_densify_prune.py -s dataset/360/kitchen -m output/Light_kitchen --eval --r 1200

conda deactivate