# !/bin/bash

# From a compute node

source $HOME/.bashrc module purge module load gcc conda conda create --prefix /projects/cavs/conda-envs/rllib python=3.6
conda activate /projects/cavs/conda-envs/rllib conda install libgcc -y TMPDIR=$LOCAL_SCRATCH pip install -U
ray[rllib,debug] tensorflow

# Test training

# rllib train --run=PPO --env=CartPole-v0

