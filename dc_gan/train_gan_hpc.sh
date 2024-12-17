#!/bin/bash -l
#SBATCH -J train_dc_gan
#SBATCH --mem=150G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:3
#SBATCH --time=15-00:00:00
#SBATCH --partition=gpu-qi
#SBATCH --mail-user=aetaffe@ucdavis.edu
#SBATCH --mail-type=ALL
#SBATCH -o jobs/train_gan-%j.output
#SBATCH -e jobs/train_gan-%j.error

# Run the Python script with the input file
module load conda3/4.X
conda activate alex-synthetic-data
python train_dc_gan.py