#!/bin/sh

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=surprSwitchboard
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

ROOT="${HOME}/projects/surprise"
cd "${ROOT}" || exit

MODEL_DIR="/scratch-shared/mariog/surprise/models"

for DATASET in "switchboard";
do
    for MODEL_NAME in "${MODEL_DIR}/gpt2/${DATASET}/checkpoint-1945" "${MODEL_DIR}/gpt2-medium/${DATASET}/checkpoint-1554" "${MODEL_DIR}/gpt2-large/${DATASET}/checkpoint-1554" "${MODEL_DIR}/microsoft/DialoGPT-small/${DATASET}/checkpoint-1945" "${MODEL_DIR}/microsoft/DialoGPT-medium/${DATASET}/checkpoint-2072" "${MODEL_DIR}/microsoft/DialoGPT-large/${DATASET}/checkpoint-1554" "${MODEL_DIR}/EleutherAI/gpt-neo-125m/${DATASET}/checkpoint-1556" "${MODEL_DIR}/EleutherAI/gpt-neo-1.3B/${DATASET}/checkpoint-777";
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
