#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

source ~/.bashrc
CONDA_BIN="${CONDA_EXE:-$(command -v conda || true)}"
if [[ -z "${CONDA_BIN}" && -x "${HOME}/anaconda3/bin/conda" ]]; then
  CONDA_BIN="${HOME}/anaconda3/bin/conda"
fi
if [[ -z "${CONDA_BIN}" ]]; then
  echo "ERROR: conda not found" >&2
  exit 1
fi
source "$("${CONDA_BIN}" info --base)/etc/profile.d/conda.sh"
conda activate "${ENCODE_ENV:-rofacto}"

export PYTHONPATH="${PYTHONPATH:-}:${REPO_ROOT}"

ROOT_DIR="${ROOT_DIR:-data/droid/render_v2}"
TARGET_HEIGHT="${TARGET_HEIGHT:-256}"
TARGET_WIDTH="${TARGET_WIDTH:-448}"
VIDEOS_DIR_NAME="${VIDEOS_DIR_NAME:-videos_clips81}"
VIDEOS_OUTPUT_DIR_STEM="${VIDEOS_OUTPUT_DIR_STEM:-video_latents_clips81_${TARGET_HEIGHT}x${TARGET_WIDTH}}"
IMAGE_LATENTS_OUTPUT_DIR_STEM="${IMAGE_LATENTS_OUTPUT_DIR_STEM:-image_latents_clips81_${TARGET_HEIGHT}x${TARGET_WIDTH}}"
FUN_INP_I2V_OUTPUT_DIR_STEM="${FUN_INP_I2V_OUTPUT_DIR_STEM:-fun_inp_i2v_latents_clips81_${TARGET_HEIGHT}x${TARGET_WIDTH}}"
CLIP_IMAGE_EMBEDS_OUTPUT_DIR_STEM="${CLIP_IMAGE_EMBEDS_OUTPUT_DIR_STEM:-clip_image_embeds_clips81_${TARGET_HEIGHT}x${TARGET_WIDTH}}"
IMAGE_ENCODER_SUBPATH="${IMAGE_ENCODER_SUBPATH:-models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth}"
ENCODE_I2V_IMAGE_LATENTS="${ENCODE_I2V_IMAGE_LATENTS:-0}"
ENCODE_FUN_INP_I2V_LATENTS="${ENCODE_FUN_INP_I2V_LATENTS:-0}"
SHARD_PENDING_AFTER_SKIP_EXISTING="${SHARD_PENDING_AFTER_SKIP_EXISTING:-0}"
SHARD_JOBS_BY_KEY="${SHARD_JOBS_BY_KEY:-0}"
STATIC_VIDEOS_DIR_NAME="${STATIC_VIDEOS_DIR_NAME:-static_videos_clips81}"
STATIC_VIDEOS_OUTPUT_DIR_STEM="${STATIC_VIDEOS_OUTPUT_DIR_STEM:-static_video_latents_clips81_${TARGET_HEIGHT}x${TARGET_WIDTH}}"
STATIC_VIDEOS_FALLBACK_DIR_NAME="${STATIC_VIDEOS_FALLBACK_DIR_NAME:-videos_clips81}"
HAND_VIDEOS_DIR_NAME="${HAND_VIDEOS_DIR_NAME:-robot_mesh_videos_action_clips81}"
HAND_VIDEOS_OUTPUT_DIR_STEM="${HAND_VIDEOS_OUTPUT_DIR_STEM:-robot_mesh_videos_action_clips81_latents_${TARGET_HEIGHT}x${TARGET_WIDTH}}"
STATIC_DISPARITY_DIR_NAME="${STATIC_DISPARITY_DIR_NAME:-static_depth_videos_clips81}"
STATIC_DISPARITY_OUTPUT_DIR_STEM="${STATIC_DISPARITY_OUTPUT_DIR_STEM:-static_depth_videos_clips81_latents_${TARGET_HEIGHT}x${TARGET_WIDTH}}"
STATIC_DISPARITY_IMAGE_LATENTS_OUTPUT_DIR_STEM="${STATIC_DISPARITY_IMAGE_LATENTS_OUTPUT_DIR_STEM:-static_depth_image_latents_clips81_${TARGET_HEIGHT}x${TARGET_WIDTH}}"
HAND_DISPARITY_DIR_NAME="${HAND_DISPARITY_DIR_NAME:-robot_eef_action_depth_videos_clips81}"
HAND_DISPARITY_OUTPUT_DIR_STEM="${HAND_DISPARITY_OUTPUT_DIR_STEM:-robot_eef_action_depth_videos_clips81_latents_${TARGET_HEIGHT}x${TARGET_WIDTH}}"
PROMPT_DIR_NAME="${PROMPT_DIR_NAME:-videos_clips81_prompts_qwen3vl}"
PROMPT_OUTPUT_DIR_STEM="${PROMPT_OUTPUT_DIR_STEM:-prompt_embeds_qwen3vl_general}"
PROMPT_JSON_FIELDS="${PROMPT_JSON_FIELDS:-general.text}"
WAN_VERSION="${WAN_VERSION:-2.1}"
DTYPE="${DTYPE:-bf16}"
WAN_DATASET_FILES="${WAN_DATASET_FILES-data/dataset_files/droid_render_v2_date_split/train_videos_clips81.txt,data/dataset_files/droid_render_v2_date_split/test_videos_clips81.txt}"

if [[ $# -eq 0 ]]; then
  MODALITIES=(videos prompts hand_videos static_videos static_disparity hand_disparity)
else
  MODALITIES=("$@")
fi

has_modality() {
  local needle="$1"
  local modality
  for modality in "${MODALITIES[@]}"; do
    if [[ "${modality}" == "${needle}" ]]; then
      return 0
    fi
  done
  return 1
}

echo "=== run_encode_with_wan_droid.sh ==="
echo "repo_root=${REPO_ROOT}"
echo "root_dir=${ROOT_DIR}"
echo "modalities=${MODALITIES[*]}"
echo "target_size=${TARGET_HEIGHT}x${TARGET_WIDTH}"
echo "videos_dir_name=${VIDEOS_DIR_NAME}"
echo "videos_output_dir_stem=${VIDEOS_OUTPUT_DIR_STEM}"
echo "image_latents_output_dir_stem=${IMAGE_LATENTS_OUTPUT_DIR_STEM}"
echo "fun_inp_i2v_output_dir_stem=${FUN_INP_I2V_OUTPUT_DIR_STEM}"
echo "clip_image_embeds_output_dir_stem=${CLIP_IMAGE_EMBEDS_OUTPUT_DIR_STEM}"
echo "image_encoder_subpath=${IMAGE_ENCODER_SUBPATH}"
echo "encode_i2v_image_latents=${ENCODE_I2V_IMAGE_LATENTS}"
echo "encode_fun_inp_i2v_latents=${ENCODE_FUN_INP_I2V_LATENTS}"
echo "shard_pending_after_skip_existing=${SHARD_PENDING_AFTER_SKIP_EXISTING}"
echo "shard_jobs_by_key=${SHARD_JOBS_BY_KEY}"
echo "static_videos_dir_name=${STATIC_VIDEOS_DIR_NAME}"
echo "static_videos_output_dir_stem=${STATIC_VIDEOS_OUTPUT_DIR_STEM}"
echo "static_videos_fallback_dir_name=${STATIC_VIDEOS_FALLBACK_DIR_NAME:-<none>}"
echo "hand_videos_dir_name=${HAND_VIDEOS_DIR_NAME}"
echo "hand_videos_output_dir_stem=${HAND_VIDEOS_OUTPUT_DIR_STEM}"
echo "static_disparity_dir_name=${STATIC_DISPARITY_DIR_NAME}"
echo "static_disparity_output_dir_stem=${STATIC_DISPARITY_OUTPUT_DIR_STEM}"
echo "static_disparity_image_latents_output_dir_stem=${STATIC_DISPARITY_IMAGE_LATENTS_OUTPUT_DIR_STEM}"
echo "hand_disparity_dir_name=${HAND_DISPARITY_DIR_NAME}"
echo "hand_disparity_output_dir_stem=${HAND_DISPARITY_OUTPUT_DIR_STEM}"
echo "wan_version=${WAN_VERSION}"
echo "dtype=${DTYPE}"
echo "dataset_files=${WAN_DATASET_FILES:-<none>}"
if has_modality prompts; then
  echo "prompt_dir_name=${PROMPT_DIR_NAME}"
  echo "prompt_output_dir_stem=${PROMPT_OUTPUT_DIR_STEM}"
  echo "prompt_json_fields=${PROMPT_JSON_FIELDS:-<none>}"
fi

ARRAY_ARGS=()
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  SPLITS="${SLURM_ARRAY_TASK_COUNT:-1}"
  ARRAY_ARGS=(--rank "${SLURM_ARRAY_TASK_ID}" --world_size "${SPLITS}")
  echo "array_mode=on rank=${SLURM_ARRAY_TASK_ID}/${SPLITS}"
else
  echo "array_mode=off"
fi

DATASET_ARGS=()
if [[ -n "${WAN_DATASET_FILES}" ]]; then
  IFS=', ' read -r -a DATASET_FILES_ARRAY <<< "${WAN_DATASET_FILES}"
  if [[ ${#DATASET_FILES_ARRAY[@]} -gt 0 ]]; then
    DATASET_ARGS=(--dataset_files "${DATASET_FILES_ARRAY[@]}")
  fi
fi

PROMPT_JSON_ARGS=()
if [[ -n "${PROMPT_JSON_FIELDS}" ]]; then
  IFS=', ' read -r -a PROMPT_JSON_FIELDS_ARRAY <<< "${PROMPT_JSON_FIELDS}"
  if [[ ${#PROMPT_JSON_FIELDS_ARRAY[@]} -gt 0 ]]; then
    PROMPT_JSON_ARGS=(--prompt_json_fields "${PROMPT_JSON_FIELDS_ARRAY[@]}")
  fi
fi

I2V_ARGS=()
case "${ENCODE_I2V_IMAGE_LATENTS}" in
  1|true|TRUE|yes|YES|on|ON)
    I2V_ARGS+=(--encode_i2v_image_latents)
    ;;
esac
case "${ENCODE_FUN_INP_I2V_LATENTS}" in
  1|true|TRUE|yes|YES|on|ON)
    I2V_ARGS+=(--encode_fun_inp_i2v_latents)
    ;;
esac

PENDING_SHARD_ARGS=()
case "${SHARD_PENDING_AFTER_SKIP_EXISTING}" in
  1|true|TRUE|yes|YES|on|ON)
    PENDING_SHARD_ARGS+=(--shard_pending_after_skip_existing)
    ;;
esac
case "${SHARD_JOBS_BY_KEY}" in
  1|true|TRUE|yes|YES|on|ON)
    PENDING_SHARD_ARGS+=(--shard_jobs_by_key)
    ;;
esac

python3 -u data_processing/encode_with_wan.py \
  --root_dir "${ROOT_DIR}" \
  --dataset_type droid \
  --modalities "${MODALITIES[@]}" \
  --target_height "${TARGET_HEIGHT}" \
  --target_width "${TARGET_WIDTH}" \
  --videos_dir_name "${VIDEOS_DIR_NAME}" \
  --videos_output_dir_stem "${VIDEOS_OUTPUT_DIR_STEM}" \
  --image_latents_output_dir_stem "${IMAGE_LATENTS_OUTPUT_DIR_STEM}" \
  --fun_inp_i2v_output_dir_stem "${FUN_INP_I2V_OUTPUT_DIR_STEM}" \
  --clip_image_embeds_output_dir_stem "${CLIP_IMAGE_EMBEDS_OUTPUT_DIR_STEM}" \
  --image_encoder_subpath "${IMAGE_ENCODER_SUBPATH}" \
  --static_videos_dir_name "${STATIC_VIDEOS_DIR_NAME}" \
  --static_videos_output_dir_stem "${STATIC_VIDEOS_OUTPUT_DIR_STEM}" \
  --static_videos_fallback_dir_name "${STATIC_VIDEOS_FALLBACK_DIR_NAME}" \
  --hand_videos_dir_name "${HAND_VIDEOS_DIR_NAME}" \
  --hand_videos_output_dir_stem "${HAND_VIDEOS_OUTPUT_DIR_STEM}" \
  --static_disparity_dir_name "${STATIC_DISPARITY_DIR_NAME}" \
  --static_disparity_output_dir_stem "${STATIC_DISPARITY_OUTPUT_DIR_STEM}" \
  --static_disparity_image_latents_output_dir_stem "${STATIC_DISPARITY_IMAGE_LATENTS_OUTPUT_DIR_STEM}" \
  --hand_disparity_dir_name "${HAND_DISPARITY_DIR_NAME}" \
  --hand_disparity_output_dir_stem "${HAND_DISPARITY_OUTPUT_DIR_STEM}" \
  --prompt_dir_name "${PROMPT_DIR_NAME}" \
  --prompt_output_dir_stem "${PROMPT_OUTPUT_DIR_STEM}" \
  --wan_version "${WAN_VERSION}" \
  --dtype "${DTYPE}" \
  --skip_existing \
  "${I2V_ARGS[@]}" \
  "${PENDING_SHARD_ARGS[@]}" \
  "${PROMPT_JSON_ARGS[@]}" \
  "${DATASET_ARGS[@]}" \
  "${ARRAY_ARGS[@]}"
