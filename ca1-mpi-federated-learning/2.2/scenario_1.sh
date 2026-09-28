#!/bin/bash
#SBATCH --nodes=2                  
#SBATCH --ntasks-per-node=2     
#SBATCH --output=scenario_1.out      
#SBATCH --job-name=karimi_fed_learning_1  

#SBATCH --mem-per-cpu=512m

# module load GCC Python OpenMPI

time srun python logreg_fedavg.py --num_rounds 1 --num_epochs 10
