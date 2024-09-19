#!/bin/sh
#SBATCH --job-name=Gpu_Batch             # Job name
#SBATCH -o video_gen.log       	   	 # output file name    
#SBATCH -e video_gen.err          	 # error file name
#SBATCH --mail-type=NONE                 # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=chaoyi_he@tamu.edu   # Where to send mail
#SBATCH --nodes=1                        # Use one node
#SBATCH --ntasks=1                       # Run a single task
#SBATCH --cpus-per-task=8                # Number of CPU cores per task
#SBATCH --gres=gpu:a100:2                # Type and number of GPUs 
#SBATCH --partition=gpu-research         # Partition/Queue to run in
#SBATCH --qos=olympus-research-gpu-sh    # Set QOS to use
#SBATCH --time=04:00:00                  # Time limit hrs:min:sec - set to 1 hour

# enter your commands below
# set woring directory if different than current directory
cd /mnt/shared-scratch/Katehi_L/chaoyi_he/Video_gen/

# Start anaconda shell (if needed)

# Activate conda environment
source /mnt/shared-scratch/Katehi_L/chaoyi_he/Conda_env/bin/activate
conda activate 3_9_env

# Run Python 
torchrun --nproc_per_node=2 main.py 2>&1 output.log