#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=Video_gen         #Set the job name to "JobExample4"
#SBATCH --time=72:00:00              #Set the wall clock limit to 1hr and 30min
#SBATCH --nodes=1
#SBATCH --ntasks=1                   #Request 1 task (CPU)
#SBATCH --cpus-per-task=8            #Request exactly (or max) of 28 tasks per node     
#SBATCH --gres=gpu:a100:2            #Request 1 GPU per node can be 1 or 2, type=k80 or v100
#SBATCH -p gpu-research
#SBATCH --qos=olympus-research-gpu
#SBATCH -o Video_gen_%j.log          #Send stdout/err to "Example4Out.[jobID]"
#SBATCH -e Video_gen_%j.err

##OPTIONAL JOB SPECIFICATIONS
##SBATCH --account=122768702865            #Set billing account to Terra:122768702865	Grace:132768702708
#SBATCH --mail-type=FAIL              	   #Send email on all job events
#SBATCH --mail-user=chaoyi_he@tamu.edu    #Send all emails to email_address 

#First Executable Line

cd /mnt/shared-scratch/Katehi_L/chaoyi_he/Video_gen/
source /mnt/shared-scratch/Katehi_L/chaoyi_he/conda/miniconda/bin/activate
conda activate 3_9_env

torchrun --nproc_per_node=2 --master_port=42247 main.py
