#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=pplDailyDialog
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --mem=64G
#SBATCH --output=slurm_output/%A.out
# https://servicedesk.surfsara.nl/wiki/display/WIKI/Lisa+usage+and+accounting
echo "${SLURM_JOB_ID}"
date


source ${HOME}/.bashrc
module purge
module load 2022
module load Anaconda3/2022.05
source "$(conda info --base)/etc/profile.d/conda.sh"
conda deactivate
conda activate stat-gen-eval-3.10


ROOT="${HOME}/information-value"

cd "${ROOT}" || exit

CACHE_DIR="/../information-value/.cache"
DATASET_DIR="${ROOT}/data/corpora/dailydialog/train/"

for MODEL in "gpt2" "gpt2-medium" "gpt2-large" "gpt2-xl" \
              "facebook/opt-125m" "facebook/opt-350m" "facebook/opt-1.3b" "facebook/opt-2.7b" \
              "EleutherAI/gpt-neo-125m" "EleutherAI/gpt-neo-1.3b" "EleutherAI/gpt-neo-2.7b";
do
  echo ${MODEL}
  for FILE in "${DATASET_DIR}"/*;
  do
    echo ${FILE}
    FILENAME="$(basename -- "$FILE" .txt)"
    SAVE="${ROOT}/eval/ppl/${MODEL}/${FILENAME}"
    python3 "${ROOT}"/code/run_clm.py \
      --model_name_or_path ${MODEL} \
      --do_eval \
      --validation_file "${FILE}" \
      --output_dir "${SAVE}" \
      --cache_dir ${CACHE_DIR} \
      --per_device_eval_batch_size=16 \
      --save_total_limit=1 \
      --bf16=False \
      --num_train_epochs=1
  done
done
