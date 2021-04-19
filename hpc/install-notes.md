These are notes on how the conda cavs environment was built.

See README.md for how to use the prebuilt conda environment on HPC -- this is what most users should do!

Highly recommend doing this on a COMPUTE node to make sure environment is same as run-time.

## Install ray

(Modified from longer thread here: https://github.nrel.gov/HPC-CAVs/cavs-environments/issues/32)

Install from source using:  https://ray.readthedocs.io/en/latest/installation.html

Dependencies `curl unzip psmisc` referenced in the documentation appear to exist already on eagle. These are the only
steps I needed to build. On an interactive compute node:

```
# srun -Acavs -n1 -t30 --pty $SHELL
# set up conda environment
module purge
module load gcc conda
conda deactivate 
conda env create -n ray python=3.6
conda activate ray
pip install cython==0.29.0
conda install libgcc

# build from source
git clone git@github.com:ray-project/ray.git
ray/ci/travis/install-bazel.sh
cd ray/python
TMPDIR=$LOCAL_SCRATCH pip install -e . --verbose 
```

#### Notes.

* The last command needs any viable temp directory for downloading files (see `pip install --help`). It's not clear
  where it tries to build by default, but wherever it is leads to `[Errno 28] No space left on device`. A supported temp
  dir on Eagle is `/tmp/scratch` which is also mapped to `$LOCAL_SCRATCH`.
* Building the stack with `bazel` is much faster with a compute node than a login node. It also looks like there is a
  lot of multithreading built into bazel so it's probably good etiquette not to eat up cycles on the login node running
  this...

## Installing cavs_environments using prebuilt rllib

```
# Load modules and clone the working (?) conda env for ray/rllib
module purge
module load gcc conda
conda deactivate
conda create -n cavs --clone /projects/cavs/conda-envs/ray_0.8.0.dev6_py36
conda activate cavs

# Install cavs_environments dependencies
conda install requests tabulate psutil
conda install -c conda-forge libjpeg-turbo  # libjpeg.so.8 was missing when i tried, this fixes
conda install -c conda-forge setproctitle
easy_install /projects/cavs/Carla/carla-0.9.5/PythonAPI/carla/dist/carla-0.9.5-py3.5-linux-x86_64.egg 
```

# Install cavs_environments

```
git clone git@github.nrel.gov:HPC-CAVs/cavs-environments.git $CAVS_GITREPO
cd $CAVS_GITREPO
TMPDIR=$LOCAL_SCRATCH pip install -e .
```

# Untested:  use conda-build to build and install package, rather than "develop" it

See [README.md](README.md) for tests.

