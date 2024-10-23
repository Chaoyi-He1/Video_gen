#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=Video_gen         		#Set the job name to "JobExample4"
#SBATCH --time=72:00:00              		#Set the wall clock limit to 1hr and 30min
#SBATCH --ntasks=8                   		#Request 1 task (CPU)
#SBATCH --ntasks-per-node=8          		#Request exactly (or max) of 28 tasks per node
#SBATCH --mem=64G                    		#Request 64GB per node
#SBATCH --gres=gpu:a100:2            		#Request 1 GPU per node can be 1 or 2, type=k80 or v100
#SBATCH --partition=gpu              		#Request the GPU partition/queue
#SBATCH -o Video_gen_DiT_L_4_%j.log    		#Send stdout/err to "Example4Out.[jobID]"
#SBATCH -e Video_gen_%j.err

##OPTIONAL JOB SPECIFICATIONS
##SBATCH --account=122768702865            	#Set billing account to Terra:122768702865	Grace:132768702708
##SBATCH --mail-type=ALL              	   	#Send email on all job events
##SBATCH --mail-user=chaoyi_he@tamu.edu    	#Send all emails to email_address 
##module load CUDA/12.4

#First Executable Line

module load GCCcore/11.2.0
module load Python/3.9.6
module load CUDA/12.4
cd /scratch/user/chaoyi_he
source Python_3_9/bin/activate
export HF_HOME=/scratch/user/chaoyi_he/torch_cache
cd Video_gen/

torchrun --nproc_per_node=2 main.py
