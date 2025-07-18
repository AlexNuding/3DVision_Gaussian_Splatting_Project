#!/bin/bash

conda activate mini_splatting

python mini-splatting/ms/train.py -s dataset/360/garden -m output/mini_garden -i images_4 --eval --imp_metric outdoor --r 1200
python mini-splatting/ms/train.py -s dataset/360/stump -m output/mini_stump -i images_4 --eval --imp_metric outdoor --r 1200
python mini-splatting/ms/train.py -s dataset/360/kitchen -m output/mini_kitchen -i images_2 --eval --imp_metric indoor --r 1200

python mini-splatting/ms_c/run.py -s dataset/360/garden -m output/mini_garden
python mini-splatting/ms_c/run.py -s dataset/360/stump -m output/mini_stump 
python mini-splatting/ms_c/run.py -s dataset/360/kitchen -m output/mini_kitchen

conda deactivate