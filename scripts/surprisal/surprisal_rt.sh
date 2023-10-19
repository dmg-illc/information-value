#!/bin/sh

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=surprRT
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
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
for MODEL_NAME in "gpt2" "EleutherAI/gpt-neo-125m" "facebook/opt-125m" "gpt2-medium" "gpt2-large" "EleutherAI/gpt-neo-1.3B" "facebook/opt-1.3b" "facebook/opt-350m";
  do
  for DATASET in "provo" "ns" "brown";
    do 
    DATA_PATH="${ROOT}/data/psychometric/RTs/${DATASET}_rt.jsonl"
    OUT_DIR="${ROOT}/data/surprisal/${DATASET}/${MODEL_NAME}"
    python3 code/compute_surprisal.py \
      --corpus_path "${DATA_PATH}" \
      --model_name "${MODEL_NAME}" \
      --output_path "${OUT_DIR}" \
      --random_contexts
  done
done

date
