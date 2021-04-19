#!/bin/bash
source ~/.bashrc
module purge
module load gcc conda
conda deactivate
conda activate cavs
