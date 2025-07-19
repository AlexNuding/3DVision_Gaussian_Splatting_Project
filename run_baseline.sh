#!/bin/bash

#change to conda install location
CONDA_BASE=~/miniconda3

source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate gaussian_splatting

python gaussian-splatting/train.py -s dataset/360/garden -m output/base_garden --eval -r 1200 --data_device cpu --save_iterations 30000 --checkpoint_iterations 30000
python gaussian-splatting/train.py -s dataset/360/stump -m output/base_stump --eval -r 1200 --data_device cpu --save_iterations 30000 --checkpoint_iterations 30000
python gaussian-splatting/train.py -s dataset/360/kitchen -m output/base_kitchen --eval -r 1200 --data_device cpu --save_iterations 30000 --checkpoint_iterations 30000

python gaussian-splatting/render.py -m output/base_garden
python gaussian-splatting/render.py -m output/base_stump
python gaussian-splatting/render.py -m output/base_kitchen

python gaussian-splatting/metrics.py -m output/base_garden
python gaussian-splatting/metrics.py -m output/base_stump
python gaussian-splatting/metrics.py -m output/base_kitchen

conda deactivate