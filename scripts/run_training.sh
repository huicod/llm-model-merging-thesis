#!/usr/bin/env bash
set -euo pipefail

CONFIG_DIR="${CONFIG_DIR:-configs}"

llamafactory-cli train "${CONFIG_DIR}/qwen2_1_5b_math_lora.yaml"
llamafactory-cli train "${CONFIG_DIR}/qwen2_1_5b_code_lora.yaml"
llamafactory-cli train "${CONFIG_DIR}/qwen2_1_5b_sciq_lora.yaml"
