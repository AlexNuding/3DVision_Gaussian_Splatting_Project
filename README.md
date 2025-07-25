# 3DVision Gaussian Splatting Project

A combination of the following Gaussian Splatting Projects for evaluation and comparison:
- 3D Gaussian Splatting - https://github.com/graphdeco-inria/gaussian-splatting
- Mini-Splatting - https://github.com/fatPeter/mini-splatting
- LightGaussian - https://github.com/VITA-Group/LightGaussian
- Compressed 3D Gaussian Splatting - https://github.com/KeKsBoTer/c3dgs
- Compact 3D - https://github.com/UCDvision/compact3d


## How to Use
The repository can be used by following the instructions:

1. Clone the repository with ```shell git clone https://github.com/AlexNuding/3DVision_Gaussian_Splatting_Project.git --recursive```
2. install the environments of each of the gaussian splatting project you want to evaluate
3. open the console in tha base folder of the repository
4. run the setup.sh script3
5. download the MipNeRF dataset from [here](https://jonbarron.info/mipnerf360/) and the Tanks&Temples dataset from [here](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip)
6. unpack the garden, kitchen and stump szene from the MipNeRF dataset, they go in the 360 folder of the created dataset folder. Then unpack the train and truck szene from the Tanks&Temples dataset place them in the tandt folder.
7. execute the run script of the project you want to evaluate, this will train and evaluate the project
8. For visualization follow the instructions of the visualizer in the base Gaussian Splatting repository
9. (optional) run the get_metrics.py file to aggregate all results in a results_combined.json if desired
10. (optional) the plots folder contains a script to generate plots for better evaluation from the results_and_fps.json file. copying the combined metrics in there and adding in the FPS from the visualization is needed.
