#!/bin/bash

#change to conda install location
CONDA_BASE=~/miniconda3

source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate mini_splatting

python mini-splatting/ms/train.py -s dataset/360/garden -m output/mini_garden -i images_4 --eval --imp_metric outdoor --r 1200 --save_iterations 30000 --checkpoint_iterations 30000
python mini-splatting/ms/train.py -s dataset/360/stump -m output/mini_stump -i images_4 --eval --imp_metric outdoor --r 1200 --save_iterations 30000 --checkpoint_iterations 30000
python mini-splatting/ms/train.py -s dataset/360/kitchen -m output/mini_kitchen -i images_2 --eval --imp_metric indoor --r 1200 --save_iterations 30000 --checkpoint_iterations 30000

python mini-splatting/render.py -m output/mini_garden
python mini-splatting/render.py -m output/mini_stump
python mini-splatting/render.py -m output/mini_kitchen

python mini-splatting/metrics.py -m output/mini_garden
python mini-splatting/metrics.py -m output/mini_stump
python mini-splatting/metrics.py -m output/mini_kitchen


#compressed minisplatting
python mini-splatting/ms_c/run.py -s dataset/360/garden -m output/mini_garden
python mini-splatting/ms_c/run.py -s dataset/360/stump -m output/mini_stump 
python mini-splatting/ms_c/run.py -s dataset/360/kitchen -m output/mini_kitchen

python mini-splatting/metrics_mini_compressed.py -m output/mini_garden
python mini-splatting/metrics_mini_compressed.py -m output/mini_stump
python mini-splatting/metrics_mini_compressed.py -m output/mini_kitchen


conda deactivate