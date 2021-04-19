#!/bin/bash

### This is from xiangyu

#SBATCH --account=cavs
#SBATCH --time=30:00
#SBATCH --job-name=plannniing
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --partition=debug

 
module purge
module load conda
 
source ~/.bashrc
 
conda activate ray
 
module purge
which python
 
unset LD_PRELOAD
export CARLA_HOME=/projects/cavs/Carla/carla-0.9.10/
export TUNE_RESULT_DIR=/scratch/pgraf/ray_results/
 
python train_path_planning.py



 
