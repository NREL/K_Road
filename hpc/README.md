# RLLIB on HPC

What is here:

* Basic setup and examples:  See below
* [examples](examples): Example scripts for running rllib locally and as batch job on HPC.
* [install-notes.md](install-notes.md): Detailed instructions on installing software on HPC.
* [x11.md](x11.md): Enabling X11 forwarding if you want to render environments.
* [rllib-install.md](rllib-install.md): *NEW* installation of rllib on Eagle using pip

__Note:__ Enabling rendering in gym, whether explicitly, e.g.,

__Note:__ rllib did not previously install on eagle using pip, but it does now (???)

```
env.render()
```

or implicitly, e.g.,

```
from gym.envs.classic_control import rendering
```

will crash your code unless X11 is enabled. It will almost certainly crash batch jobs if you run them in the standard
way. Make sure you write code that doesn't enable rendering by default!

## Basic setup and examples

Here are instructions on how to set up your environment and run some examples on HPC.

Highly recommended to create the conda environment and run all tests on a COMPUTE node to ensure that your environment
matches runtime environment.

### Installing cavs_environments with preconfigured RLLIB

Installation on HPC has two steps:

1. Clone the conda environment containing packages that enable cavs_environments stuff to be parallelized across nodes
   on HPC.

        # srun -Acavs -n1 -t30 --pty $SHELL
        module purge
        module load gcc conda
        conda deactivate
        conda create -n cavs --clone /projects/cavs/conda-envs/cavs_deps_py36
        conda activate cavs

2. Install cavs_environments from source in this environment. This is a finnicky process, the install script should "
   just work" but it is very ad hoc.

        source install.sh

The second step designates `cavs_environments` as being in a `<develop>` channel, meaning it is not actually intalled as
a package but is on the conda path. This allows you to test out code changes without having to continually rebuild the
package.

### Simple tests and examples:

Load the new environment on a COMPUTE node:

```
module purge
module load gcc conda
conda deactivate
conda activate cavs
```

Minimal test of dependency installation:

```
ray start --head   # should not crash
# ray stop
```

Minimal test of cavs_environments installation:

```
# In Python REPL
>>> import cavs_environments
Registered cavs_environments!
```

Minimal test of rllib training on a single COMPUTE node:

```
git clone git@github.nrel.gov:HPC-CAVs/cavs-environments.git $CAVS_GITREPO
cd $CAVS_GITREPO/hpc/examples
python test_conda.py  # should start training
```

Simple example of multinode rllib ("parallel sleeping") on HPC. Do from LOGIN node. You might also need to change your
environment setup at top of script.

```
cd $CAVS_GITREPO/hpc/examples
sbatch ./trainer.sh
```

Simple test of multinode training on HPC. Do from LOGIN node. You might also need to change your environment setup at
top of script.

```
cd $CAVS_GITREPO/hpc/examples
sbatch ./path_tracking_test_standard.sh
```

