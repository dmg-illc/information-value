#!/bin/sh

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=surprDailyDialog
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

ROOT="${HOME}/information-value"

cd "${ROOT}" || exit

CACHE_DIR="/../information-value/.cache"
MODEL_DIR="/../information-value/models"

for DATASET in "dailydialog";
do
    for MODEL_NAME in "${MODEL_DIR}/gpt2/${DATASET}/checkpoint-1075/" "${MODEL_DIR}/gpt2-medium/${DATASET}/checkpoint-1435/" "${MODEL_DIR}/gpt2-large/${DATASET}/checkpoint-1290/" "${MODEL_DIR}/microsoft/DialoGPT-small/${DATASET}/checkpoint-1075/" "${MODEL_DIR}/microsoft/DialoGPT-medium/${DATASET}/checkpoint-1435/" "${MODEL_DIR}/microsoft/DialoGPT-large/${DATASET}/checkpoint-1720/" "${MODEL_DIR}/EleutherAI/gpt-neo-125m/${DATASET}/checkpoint-1075/" "${MODEL_DIR}/EleutherAI/gpt-neo-1.3B/${DATASET}/checkpoint-860/";
    do
    DATA_PATH="${ROOT}/data/psychometric/${DATASET}/${DATASET}_results_is.jsonl"
    OUT_DIR="${ROOT}/data/surprisal/${DATASET}/${MODEL_NAME}"
    python3 code/compute_surprisal.py \
      --corpus_path "${DATA_PATH}" \
      --model_name "${MODEL_NAME}" \
      --output_path "${OUT_DIR}" \
      --separator "<\s> <s>" \
      --random_contexts
  done
done

date
