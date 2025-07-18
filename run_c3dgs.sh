#!/bin/bash

conda activate c3dgs

python c3dgs/compress.py --model_path output/base_garden --data_device "cuda" --output_vq output/c3dgs_garden
python c3dgs/compress.py --model_path output/base_stump --data_device "cuda" --output_vq output/c3dgs_stump
python c3dgs/compress.py --model_path output/base_kitchen --data_device "cuda" --output_vq output/c3dgs_kitchen

conda deactivate