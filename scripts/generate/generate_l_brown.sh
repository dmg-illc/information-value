#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=largeBrown
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00
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


DATASET="brown"
FNAME="${DATASET}_rt"
#MODEL_NAME="EleutherAI/gpt-j-6B"  #"microsoft/DialoGPT-medium" # distilgpt2, gpt2, gpt2-medium, microsoft/DialoGPT-medium
N_SAMPLES_PER_RUN=10
N_RUNS=10
MAX_LENGTH=63

ROOT="${HOME}/projects/surprise"
DATA_PATH="${ROOT}/data/psychometric/RTs/${FNAME}.jsonl"
cd ${ROOT}

for MODEL_NAME in "gpt2-large" "facebook/opt-1.3b" "EleutherAI/gpt-neo-1.3B";
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