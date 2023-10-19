#!/bin/sh

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=LMEM-Brown
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=20:00:00
#SBATCH --mem=32G
#SBATCH --output=slurm_output/%A.out
# https://servicedesk.surfsara.nl/wiki/display/WIKI/Lisa+usage+and+accounting
echo "${SLURM_JOB_ID}"
date

module purge
module load 2022
module load Anaconda3/2022.05
source "$(conda info --base)/etc/profile.d/conda.sh"
conda deactivate
conda activate stat-gen-eval-3.10

ROOT="${HOME}/projects/surprise"
cd "${ROOT}" || exit

python3 code/notebooks/surprise_ppo/surprise_ppo_brown.py \
	  --data_dir "${ROOT}/data/surprise" \
	  --out_name "${ROOT}/data/modelling_results"

date
