#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL="${BASE_MODEL:-/path/to/models/Qwen2-1.5B}"
CODE_MODEL="${CODE_MODEL:-/path/to/exported/qwen1.5b_code}"
MATH_MODEL="${MATH_MODEL:-/path/to/exported/qwen1.5b_math}"
SCIQ_MODEL="${SCIQ_MODEL:-/path/to/exported/qwen1.5b_sciq}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/path/to/merged_models}"

python scripts/merge_models.py \
  --base-model "${BASE_MODEL}" \
  --models "${CODE_MODEL}" "${MATH_MODEL}" "${SCIQ_MODEL}" \
  --output-dir "${OUTPUT_ROOT}/merge_CMS_maskrankmean" \
  --base-mask-rate 0.9 \
  --alpha 0.5 \
  --mask-strategy magnitude \
  --dtype bfloat16 \
  --trust-remote-code

python scripts/merge_models.py \
  --base-model "${BASE_MODEL}" \
  --models "${CODE_MODEL}" "${MATH_MODEL}" \
  --output-dir "${OUTPUT_ROOT}/merge_CM_maskrankmean" \
  --base-mask-rate 0.9 \
  --alpha 0.5 \
  --mask-strategy magnitude \
  --dtype bfloat16 \
  --trust-remote-code

python scripts/merge_models.py \
  --base-model "${BASE_MODEL}" \
  --models "${CODE_MODEL}" "${SCIQ_MODEL}" \
  --output-dir "${OUTPUT_ROOT}/merge_CS_maskrankmean" \
  --base-mask-rate 0.9 \
  --alpha 0.5 \
  --mask-strategy magnitude \
  --dtype bfloat16 \
  --trust-remote-code

python scripts/merge_models.py \
  --base-model "${BASE_MODEL}" \
  --models "${MATH_MODEL}" "${SCIQ_MODEL}" \
  --output-dir "${OUTPUT_ROOT}/merge_MS_maskrankmean" \
  --base-mask-rate 0.9 \
  --alpha 0.5 \
  --mask-strategy magnitude \
  --dtype bfloat16 \
  --trust-remote-code
