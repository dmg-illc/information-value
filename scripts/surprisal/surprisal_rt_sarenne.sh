#!/bin/bash
# source env_utterance_predictability/bin/activate
source /disk/scratch/swallbridge/miniconda3/bin/activate utt_pred 
python --version

# export TRANSFORMERS_OFFLINE=1
export PIP_DOWNLOAD_CACHE=/disk/scratch/swallbridge/BERT_FP/temp_cache
export TORCH_HOME=/disk/scratch/swallbridge/BERT_FP/temp_cache
export TRANSFORMERS_CACHE=/disk/scratch/swallbridge/BERT_FP/temp_cache

ROOT="/disk/scratch/swallbridge/utterance-predictability"

cd ${ROOT}

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
      --separator "none" \
      --random_contexts
  done
done

date
