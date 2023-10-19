#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=xlargeBLL
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


FNAME="processed_ratings"
DATASET="BLL2018"
#MODEL_NAME="EleutherAI/gpt-j-6B"  #"microsoft/DialoGPT-medium" # distilgpt2, gpt2, gpt2-medium, microsoft/DialoGPT-medium
N_SAMPLES_PER_RUN=5
N_RUNS=20
MAX_LENGTH=86

ROOT="${HOME}/projects/surprise"
DATA_PATH="${ROOT}/data/psychometric/${DATASET}/${FNAME}.jsonl"
cd ${ROOT}

for MODEL_NAME in "gpt2-xl" "facebook/opt-2.7b" "EleutherAI/gpt-neo-2.7B";
do
  OUT_DIR="${ROOT}/data/alternatives/${DATASET}/${MODEL_NAME}"
  python3 code/generate_alternatives.py \
    --data_path "${DATA_PATH}" \
    --model_name "${MODEL_NAME}" \
    --out_dir "${OUT_DIR}" \
    --n_samples_per_run ${N_SAMPLES_PER_RUN} \
    --n_sampling_runs ${N_RUNS} \
    --max_generation_length "${MAX_LENGTH}" \
    --context_separator "none"
  date
done