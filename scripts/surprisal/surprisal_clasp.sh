# #!/bin/sh

# #SBATCH --partition=gpu
# #SBATCH --gres=gpu:1
# #SBATCH --job-name=surprClasp
# #SBATCH --ntasks=1
# #SBATCH --cpus-per-task=1
# #SBATCH --time=01:00:00
# #SBATCH --mem=32G
# #SBATCH --output=slurm_output/%A.out
# # https://servicedesk.surfsara.nl/wiki/display/WIKI/Lisa+usage+and+accounting
# echo "${SLURM_JOB_ID}"
# date

# module purge
# module load 2022
# module load Anaconda3/2022.05
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda deactivate
# conda activate stat-gen-eval-3.10

# ROOT="${HOME}/projects/surprise"
# cd "${ROOT}" || exit

# SARENNE
source /disk/scratch/swallbridge/miniconda3/bin/activate utt_pred 
python --version

# export TRANSFORMERS_OFFLINE=1
export PIP_DOWNLOAD_CACHE=/disk/scratch/swallbridge/BERT_FP/temp_cache
export TORCH_HOME=/disk/scratch/swallbridge/BERT_FP/temp_cache
export TRANSFORMERS_CACHE=/disk/scratch/swallbridge/BERT_FP/temp_cache

ROOT="/disk/scratch/swallbridge/utterance-predictability"

for MODEL_NAME in "gpt2" "EleutherAI/gpt-neo-125m" "facebook/opt-125m" "gpt2-medium" "gpt2-large" "EleutherAI/gpt-neo-1.3B" "facebook/opt-1.3b" "facebook/opt-350m";
do
  DATASET="BLL2018" 
  DATA_PATH="${ROOT}/data/psychometric/${DATASET}/processed_ratings.jsonl"
  OUT_DIR="${ROOT}/data/surprisal/${DATASET}/${MODEL_NAME}"
  python3 code/compute_surprisal.py \
	  --corpus_path "${DATA_PATH}" \
	  --model_name "${MODEL_NAME}" \
	  --output_path "${OUT_DIR}" \
	  --random_contexts
done

date
