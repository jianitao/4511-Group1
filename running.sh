#!/bin/bash
#SBATCH --job-name=baseline
#SBATCH --output=%x_%o.out
#SBATCH --error=%x_%e.err
#SBATCH --partition matador
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10   #You can request up to 40 CPU cores per node
#SBATCH --gpus-per-node=1      #You can request up to 2 GPUs per node

. $HOME/conda/etc/profile.d/conda.sh
conda activate tf-gpu-hpcc
# pip install torch_geometric
# conda install conda-forge::transformers

time python second_model.py