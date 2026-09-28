#!/bin/bash
#SBATCH --nodes=2                  
#SBATCH --ntasks-per-node=2          
#SBATCH --output=malicious.out     
#SBATCH --job-name=karimi_malicious

#SBATCH --mem-per-cpu=512m

# module load GCC Python OpenMPI

time srun python malicious.py --num_rounds 10 --num_epochs 1