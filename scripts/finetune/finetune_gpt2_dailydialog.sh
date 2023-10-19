#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=ppl
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
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
OUTPUT_DIR_ROOT="/../information-value/models"
DATASET="dailydialog"
TRAIN_PATH="${ROOT}/data/corpora/${DATASET}/train/${DATASET}_train_BOS_EOS_blender.txt"
VAL_PATH="${ROOT}/data/corpora/${DATASET}/val/${DATASET}_validation_BOS_EOS_blender.txt"


for MODEL in "gpt2";  # "gpt2-medium" "gpt2-large" "gpt2-xl";
#              "facebook/opt-125m" "facebook/opt-350m" "facebook/opt-1.3b" "facebook/opt-2.7b" "facebook/opt-6.7b" \
#              "EleutherAI/gpt-neo-125m" "EleutherAI/gpt-neo-1.3b" "EleutherAI/gpt-neo-2.7b" "EleutherAI/gpt-j-6b";
do
  echo ${MODEL}
  echo "${DATASET_PATH}"
  python3 "${ROOT}"/code/run_clm.py \
    --model_name_or_path ${MODEL} \
    --do_train \
    --do_eval \
    --train_file "${TRAIN_PATH}" \
    --validation_file "${VAL_PATH}" \
    --output_dir "${OUTPUT_DIR_ROOT}/${MODEL}/${DATASET}" \
    --cache_dir ${CACHE_DIR} \
    --per_device_train_batch_size=16 \
    --per_device_eval_batch_size=16 \
    --bf16=False \
    --save_total_limit=1 \
    --evaluation_strategy=epoch \
    --logging_strategy=epoch \
    --save_strategy=epoch \
    --load_best_model_at_end=True \
    --label_smoothing_factor 0.01 \
    --num_train_epochs=3
done

