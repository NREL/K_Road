#!/bin/bash
# Run the path_tracking_test_standard.py script on multiple nodes.
# Simple modification from trainer.sh.

#SBATCH --job-name=srunpy
#SBATCH --cpus-per-task=36
#SBATCH --mem-per-cpu=1GB
#SBATCH --nodes=6
#SBATCH --tasks-per-node=1
#SBATCH --time=15
#SBATCH --account=cavs
#SBATCH --cpu-freq=high-high:Performance

# MODIFY HERE according to your environment setup
source ~/acavs


#source $HOME/.bashrc
unset LD_PRELOAD
#module purge
#module load gcc conda
#module load gcc/7.3.0
#module load cuda/10.0.130
#module load cudnn/7.4.2/cuda-10.0
#module load boost/1.69.0/gcc-7.3.0
#module load openmpi/3.1.3/gcc-7.3.0
#
#conda deactivate
#conda activate cavs

memory_per_node=$((36*1024*1024*1024))
object_store_memory=$((18*1024*1024*1024))
driver_object_store_memory=$((18*1024*1024*1024))
code
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

export ip_head # Exporting for latter access

echo "starting head"
srun --nodes=1 --ntasks=1 -w $node1 ray start --block --head --port=6379 --redis-password=$redis_password --memory=$memory_per_node --object-store-memory=$object_store_memory --num-cpus=36 &# Starting the head
sleep 60

echo "starting workers"
for ((i = 1; i <= $worker_num; i++)); do
  node2=${nodes_array[$i]}
  echo "i=${i}, node2=${node2}"
  srun --nodes=1 --ntasks=1 -w $node2 ray start --block --address=$ip_head --redis-password=$redis_password  --memory=$memory_per_node --object-store-memory=$object_store_memory --num-cpus=36 &# Starting the workers
#  srun --nodes=1 --ntasks=1 -w $node2 ray start --block --address=$ip_head --redis-password=$redis_password  --memory=$memory_per_node --object-store-memory=$object_store_memory --num-cpus=36 &# Starting the workers
  sleep 5
done

echo "executing command... python -u "$@" ray " "$redis_password" "$total_cpus"

python -u "$@" ray "$redis_password" "$total_cpus"
