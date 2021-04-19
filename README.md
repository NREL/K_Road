# cavs-environments
A collection of custom gym environments used for reinforcement learning.

`` pip install -r requirements.txt ``

## Directory Structure:
+ cavs-environments/  -- all environments
  + building/ -- building systems environments
  + grid/ -- electric grid environments
  + vehicle/ -- vehicle environments
    + SimpleGridRoad/ -- simple prototype environment

## Setting up an environment
```
# create env
conda create --name cavs-environments

# install dependencies
conda install numpy scipy
pip install pygame pymunk

# Install carla egg:
easy_install ~/programs/Carla_0.9.5/PythonAPI/carla/dist/carla-0.9.5-py3.5-linux-x86_64.egg

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

## Using this repo:
All environments in this repo can be used with a single installation. With your conda environment activated, follow the commands below for installing and using this package:

```
# Go to the cavs-environments folder.
(cavs) xzhang-32978s:cavs-environments xzhang2$ ls
HPC-CAVs-Code-Style.xml	README.md		__init__.py		cavs_environments	setup.py

# Install it.
(cavs) xzhang-32978s:cavs-environments xzhang2$ conda develop .
added /Users/xzhang2/CodeRepo/cavs-environments
completed operation for: /Users/xzhang2/CodeRepo/cavs-environments

# Use it.
(cavs) xzhang-32978s:cavs-environments xzhang2$ python
Python 3.6.8 |Anaconda, Inc.| (default, Dec 29 2018, 19:04:46) 
[GCC 4.2.1 Compatible Clang 4.0.1 (tags/RELEASE_401/final)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import gym
>>> import cavs_environments
>>> env = gym.make('CarPass-v0')
>>> env.reset()
array([  2.        , -10.51344981])
>>> env = gym.make('GridRoad-v0')
>>> env.reset()
array([ 2.,  2., -1.,  0., -1.,  0., -1.,  0., -1.,  0., -1.,  0., -1.,
        0., -1.,  0., -1.,  0., -1.,  0., -1.,  0.])
```

## Adding new envs to this repo:
**Step 1**: Put the meat of your env into the coresponding folder (building/grid/vehicle), make sure you develop the env in its standard format. [Reference link.](https://github.com/openai/gym/tree/master/gym/envs#how-to-create-new-environments-for-gym)<br>
**Step 2**: Update the init file inside the building/grid/vehicle, for importing your env class. <br>
**Step 3**: Update the init file in cavs_environments folder, for registering your env. <br>

To use the new developed envs (updated repo) in a conda environment where old version of cavs-environment has installed, it is suggested to reinstall again by uninstalling it using `conda develop -u .` (make sure you are in the cavs-environments folder), and installing it using `conda develop .`.
