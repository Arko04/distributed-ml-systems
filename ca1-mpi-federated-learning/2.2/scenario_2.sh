#!/bin/bash
#SBATCH --nodes=2                  
#SBATCH --ntasks-per-node=2          
#SBATCH --output=scenario_2.out     
#SBATCH --job-name=karimi_fed_learning_2   

#SBATCH --mem-per-cpu=512m

# module load GCC Python OpenMPI

time srun python logreg_fedavg.py --num_rounds 10 --num_epochs 1
