#!/bin/bash
# Run the path_tracking_test_standard.py script on multiple nodes.
# Simple modification from trainer.sh.

#SBATCH --job-name=pt-2
#SBATCH --cpus-per-task=36
#SBATCH --mem-per-cpu=1GB
#SBATCH --nodes=2
#SBATCH --tasks-per-node=1
#SBATCH --time=30
#SBATCH --account=cavs

# MODIFY HERE according to your environment setup
source $HOME/.bashrc
module purge
module load gcc conda
conda deactivate
conda activate cavs

#worker_num=2 # Must be one less that the total number of nodes
worker_num=$(($SLURM_JOB_NUM_NODES - 1))
total_cpus=$(($SLURM_JOB_NUM_NODES * $SLURM_CPUS_ON_NODE))
echo "worker_num="$worker_num
echo "total_cpus="$total_cpus

nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST) # Getting the node names
nodes_array=($nodes)

echo "nodes: "$nodes
echo "nodes_array: "$nodes_array

node1=${nodes_array[0]}
echo "node1: "$node1

ip_prefix=$(srun --nodes=1 --ntasks=1 -w $node1 hostname --ip-address) # Making address
suffix=':6379'
ip_head=$ip_prefix$suffix
redis_password=$(uuidgen)
echo "ip_prefix: "$ip_pref
echo "suffix: "$suffix
echo "ip_head: "$ip_head
echo "redis_password: "$redis_password

export ip_head # Exporting for latter access by trainer.py

echo "starting head"
srun --nodes=1 --ntasks=1 -w $node1 ray start --block --head --redis-port=6379 --redis-password=$redis_password &# Starting the head
sleep 5

echo "starting workers"
for ((i = 1; i <= $worker_num; i++)); do
  node2=${nodes_array[$i]}
  echo "i=${i}, node2=${node2}"
  srun --nodes=1 --ntasks=1 -w $node2 ray start --block --address=$ip_head --redis-password=$redis_password &# Starting the workers
  sleep 5
done

echo "calling trainer.py"
python -u road_test_standard.py $redis_password $total_cpus
