#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=20:00:00   # walltime
#SBATCH --ntasks=100   # number of processor cores (i.e. tasks)
#SBATCH --nodes=10  # number of nodes
#SBATCH --mem-per-cpu=15G   # memory per CPU core
#SBATCH -J "parSim"   # job name
#SBATCH --qos=normal


# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
module load python3/3.6.4
python3 parSim.py
