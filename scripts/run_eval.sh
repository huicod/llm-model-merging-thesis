#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: bash scripts/run_eval.sh /path/to/model /path/to/output_dir"
  exit 1
fi

MODEL_PATH="$1"
OUTPUT_DIR="$2"
BATCH_SIZE="${BATCH_SIZE:-1}"

mkdir -p "${OUTPUT_DIR}"

HF_ALLOW_CODE_EVAL=1 lm_eval \
  --model hf \
  --model_args "pretrained=${MODEL_PATH},trust_remote_code=True" \
  --tasks humaneval,sciq \
  --batch_size "${BATCH_SIZE}" \
  --num_fewshot 0 \
  --gen_kwargs '{"max_new_tokens":512}' \
  --confirm_run_unsafe_code \
  --output_path "${OUTPUT_DIR}/humaneval_sciq"

lm_eval \
  --model hf \
  --model_args "pretrained=${MODEL_PATH},trust_remote_code=True" \
  --tasks gsm8k \
  --batch_size "${BATCH_SIZE}" \
  --num_fewshot 0 \
  --limit 0.5 \
  --gen_kwargs '{"max_new_tokens":512}' \
  --output_path "${OUTPUT_DIR}/gsm8k_0shot"
