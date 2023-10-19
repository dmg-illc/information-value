#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=sProvoGPT2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0:30:00
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

DATASET="provo"

ROOT="${HOME}/projects/surprise"
DATA_PATH="${ROOT}/data/psychometric/RTs/${DATASET}_rt.jsonl"

cd "${ROOT}" || exit

GEN_DIR="${ROOT}/data/alternatives/${DATASET}/openAI/gpt2-large"

nfiles=$(find ${GEN_DIR} -type f -name '*nucleus_085*.jsonl'| wc -l)
echo "${nfiles} files"
count=0
for f in $(find "${GEN_DIR}" -name '*nucleus_085*.jsonl');
do
  (( count ++ ))
  echo "${count}/${nfiles}: ${f}"
  OUT_DIR="${ROOT}/data/surprise/${DATASET}"
  mkdir -p ${OUT_DIR}
  OUT_PATH="${OUT_DIR}/${f//\//-}.csv"
  python3 code/compute_surprise.py \
    --corpus_path "${DATA_PATH}" \
    --alternatives_path "${f}" \
    --output_path "${OUT_PATH}" \
    --separator "spacy" \
    --max_samples 100 \
    --step_size_samples 10 \
    --random_contexts
done

date
