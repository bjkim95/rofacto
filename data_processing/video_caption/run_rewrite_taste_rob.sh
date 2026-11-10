#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ ! -f "${PROJECT_ROOT}/data_processing/video_caption/caption_rewrite.py" ]]; then
  echo "ERROR: failed to resolve project root from ${SCRIPT_DIR}" >&2
  exit 1
fi

export PYTHONPATH="${PYTHONPATH:-}:${PROJECT_ROOT}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false

ROOT_DIR="${ROOT_DIR:-${PROJECT_ROOT}/data/taste_rob_fps16_480_832}"
PROMPT_SUBDIR="${PROMPT_SUBDIR:-prompts}"
OUTPUT_FOLDER_NAME="${OUTPUT_FOLDER_NAME:-prompts_rewrite}"
PROMPT_FILE="${PROMPT_FILE:-${PROJECT_ROOT}/data_processing/video_caption/prompt/rewrite.txt}"
MODEL_NAME="${MODEL_NAME:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
NUM_SPLITS="${NUM_SPLITS:-8}"
MAX_RETRY_COUNT="${MAX_RETRY_COUNT:-10}"
TEMPERATURE="${TEMPERATURE:-0.7}"
MAX_TOKENS="${MAX_TOKENS:-1024}"

if [[ ! -d "${ROOT_DIR}" ]]; then
  echo "ERROR: root dir not found: ${ROOT_DIR}" >&2
  exit 1
fi

if [[ ! -f "${PROMPT_FILE}" ]]; then
  echo "ERROR: prompt file not found: ${PROMPT_FILE}" >&2
  exit 1
fi

if [[ -n "${SLURM_ARRAY_TASK_ID+set}" ]]; then
  ARRAY_ARGS=(--array_index "${SLURM_ARRAY_TASK_ID}" --num_splits "${NUM_SPLITS}")
  echo "[INFO] Running array split: index=${SLURM_ARRAY_TASK_ID}, num_splits=${NUM_SPLITS}"
else
  ARRAY_ARGS=()
  echo "[INFO] Running full dataset (no array split)"
fi

python3 "${SCRIPT_DIR}/caption_rewrite.py" \
  --root_dir "${ROOT_DIR}" \
  --prompt_subdir "${PROMPT_SUBDIR}" \
  --output_folder_name "${OUTPUT_FOLDER_NAME}" \
  --prompt_file "${PROMPT_FILE}" \
  --model_name "${MODEL_NAME}" \
  --max_retry_count "${MAX_RETRY_COUNT}" \
  --prefix '"rewritten description": ' \
  --answer_template 'your rewritten description here' \
  --temperature "${TEMPERATURE}" \
  --max_tokens "${MAX_TOKENS}" \
  --skip_existing \
  "${ARRAY_ARGS[@]}"
