#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=generate
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
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
MODEL_NAME="gpt2"  #"microsoft/DialoGPT-medium" # distilgpt2, gpt2, gpt2-medium, microsoft/DialoGPT-medium
N_SAMPLES=1000
MAX_LENGTH=41

ROOT="${HOME}/projects/surprise"
DATA_PATH="${ROOT}/data/${DATASET}/${FNAME}.jsonl"
OUT_PATH="${ROOT}/data/${DATASET}/${DATASET}_${MODEL_NAME}_${N_SAMPLES}samples.jsonl"

cd ${ROOT}

python3 code/generate_alternatives.py \
  --data_path="${DATA_PATH}" \
  --model_name="${MODEL_NAME}" \
  --do_sample \
  --out_path="${OUT_PATH}" \
  --n_samples "${N_SAMPLES}" \
  --max_length "${MAX_LENGTH}" \
  --context_key "context" 

date
