#!/bin/bash

#a) DEFINE

#SBATCH --job-name=GaussianSplatting

#resources:

#SBATCH --cpus-per-task=4
# the job can use and see 4 CPUs (from max 24).

#SBATCH --partition=day
# the slurm partition the job is queued to.

#SBATCH --mem-per-cpu=8G
# the job will need 12GB of memory equally distributed on 4 cpus.  (251GB are available in total on one node)

#SBATCH --gres=gpu:A4000:2
#the job can use and see 1 GPUs (4 GPUs are available in total on one node)

#SBATCH --time=2:00:00
# the maximum time the scripts needs to run
# "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"



#SBATCH --error=job.%J.err
# write the error output to job.*jobID*.err

#SBATCH --output=job.%J.out
# write the standard output to job.*jobID*.out


source ~/miniconda3/etc/profile.d/conda.sh

export PATH=$HOME/cuda-11.8/bin:$PATH

export LD_LIBRARY_PATH=$HOME/cuda-11.8/lib64:$LD_LIBRARY_PATH

conda activate gaussian_splatting

python ~/projects/gaussian-splatting/train.py -s ~/projects/datasets/garden/ -m ~/projects/gaussian-splatting/output/base -r 1200 --checkpoint_iterations 8000 15000 30000 --eval
