#!/bin/bash
# This script was only used to test for environment variables running
# batch jobs on HPC/slurm.  It is not useful for running rllib!

#SBATCH --job-name=slurm-test
#SBATCH --cpus-per-task=35
#SBATCH --mem-per-cpu=1GB
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --time=5
#SBATCH --account=cavs

env

echo "num_nodes: "$SLURM_JOB_NUM_NODES
echo "cpus_per_node: "$SLURM_JOB_CPUS_PER_NODE

total_cpus=$(($SLURM_JOB_NUM_NODES * $SLURM_JOB_CPUS_PER_NODE))
echo "total cpus: "$total_cpus
