# K_Road
A 2-D vehicle simulator for autonomous vehicle research.

`` pip install -r requirements.txt ``

## Directory Structure:
+ command_line_tools/  -- a library for parsing JSON-like configs from the command line
+ coordinated_optimizer/  -- a library of distributed direct policy search methods
+ data_logging/  -- a library for logging experimental data
+ factored_gym/ -- a library for implementing AI gyms as separate components (process, reward function, observation function, terminator, etc)
+ hpc/ -- scripts and utilities for running K_Road on HPC systems
+ k_road/ -- the K_Road simulator
+ scenario/ -- example scenarios for K_Road
+ trainer/ -- Various RL algorithm implementations for use with RLLib and KRoad 
  

## Setting up an environment
```
# create env
conda create --name k_road

# install dependencies
conda install numpy scipy
pip install pygame pymunk

# Optional: 
conda install nb_conda_kernels

# with GPU:
# first, install CUDA drivers: http://www.nvidia.com/getcuda
sudo nvidia-xconfig
# then reboot
glxinfo

conda install tensorflow-gpu

# without GPU:
conda install tensorflow

pip install stable_baselines
 
```
