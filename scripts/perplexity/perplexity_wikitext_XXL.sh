#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=pplWikiXXL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
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


ROOT="${HOME}/projects/surprise"

cd "${ROOT}" || exit

CACHE_DIR="/scratch-shared/jbaan/stat-gen-eval/.cache"
DATASET="wikitext"


# Loop over files in DATASET_DIR

for MODEL in "facebook/opt-6.7b" "EleutherAI/gpt-j-6b";
do
  echo ${MODEL}
  SAVE="${ROOT}/eval/ppl/${MODEL}/${DATASET}"
  python3 "${ROOT}"/code/run_clm.py \
    --model_name_or_path ${MODEL} \
    --do_eval \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-v1 \
    --output_dir "${SAVE}" \
    --cache_dir ${CACHE_DIR} \
    --per_device_eval_batch_size=4 \
    --save_total_limit=1 \
    --bf16=False \
    --num_train_epochs=1
done
