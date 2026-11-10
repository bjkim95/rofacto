#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# RoFacto external asset fetcher.
#
# The repository ships code only. Everything downloaded here is an upstream
# artifact that is redistributable by its owner, not by us:
#
#   1. DROID annotations (calibration / language / filter JSONs)  ~215 MB
#      → huggingface.co/KarlP/droid
#   2. Robot meshes + third-party code used by the render / retarget pipelines
#      → pinned upstream git commits
#
# Model weights (Wan base, LightX2V LoRA, SAM3 .pt, MiniMax-Remover,
# Qwen3-VL, trained DWM checkpoints) are NOT fetched here; see README_ENV.md.
#
# Usage:  bash fetch_assets.sh [--droid-only|--third-party-only]
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DP="$HERE/data_processing"
WHAT="${1:-all}"

have() { command -v "$1" >/dev/null 2>&1; }

# ── 1. DROID annotations ─────────────────────────────────────────────────────
# Consumed by data_processing/droid/* (intrinsics + cam2base extrinsics are
# required for every render_v2 export; the rest are used by dataset builders).
DROID_REPO="KarlP/droid"
DROID_DEST="$DP/droid/assets"
DROID_FILES=(
  intrinsics.json
  cam2base_extrinsics.json
  cam2base_extrinsic_superset.json
  camera_serials.json
  keep_ranges_1_0_1.json
  droid_language_annotations.json
  episode_id_to_path.json
)

fetch_droid() {
  echo "== DROID annotations → $DROID_DEST =="
  mkdir -p "$DROID_DEST"
  local missing=()
  for f in "${DROID_FILES[@]}"; do
    [[ -s "$DROID_DEST/$f" ]] || missing+=("$f")
  done
  if [[ ${#missing[@]} -eq 0 ]]; then
    echo "  all ${#DROID_FILES[@]} files already present; skipping"
    return
  fi
  echo "  missing: ${missing[*]}"
  if have huggingface-cli; then
    for f in "${missing[@]}"; do
      huggingface-cli download "$DROID_REPO" "$f" --local-dir "$DROID_DEST" >/dev/null
      echo "  fetched $f"
    done
  else
    for f in "${missing[@]}"; do
      curl -fL --progress-bar -o "$DROID_DEST/$f" \
        "https://huggingface.co/$DROID_REPO/resolve/main/$f"
      echo "  fetched $f"
    done
  fi
}

# ── 2. Third-party code / meshes at pinned commits ───────────────────────────
# name|url|commit|subpath-to-keep (empty = whole repo)
THIRD_PARTY=(
  "robosuite|https://github.com/ARISE-Initiative/robosuite.git|51cc01785bab80ffeed20da15e67d7dd4140e76a|"
  "sam3|https://github.com/facebookresearch/sam3.git|8e451d5eb43c817b64ae7577fb7b9ae223db88a9|"
  "dex-urdf|https://github.com/dexsuite/dex-urdf.git||"
  "dex-retargeting|https://github.com/dexsuite/dex-retargeting.git||"
)

fetch_third_party() {
  echo "== third-party (pinned) → $DP/third_party =="
  mkdir -p "$DP/third_party"
  for row in "${THIRD_PARTY[@]}"; do
    IFS='|' read -r name url commit _sub <<<"$row"
    local dest="$DP/third_party/$name"
    if [[ -d "$dest" ]] && [[ -n "$(ls -A "$dest" 2>/dev/null)" ]]; then
      echo "  $name already present; skipping"
      continue
    fi
    echo "  cloning $name"
    git clone --quiet "$url" "$dest"
    if [[ -n "$commit" ]]; then
      git -C "$dest" checkout --quiet "$commit"
      echo "    pinned at ${commit:0:12}"
    fi
    rm -rf "$dest/.git"
  done
  cat <<'NOTE'
  NOTE: only a small subset of these trees is referenced by our URDFs and
  retarget configs. A full clone is larger than what the pipeline needs; that is
  expected and harmless. Licenses stay with each upstream project (robosuite:
  MIT, sam3: SAM License, SPIDER-derived work: CC BY-NC 4.0).
NOTE
}

case "$WHAT" in
  --droid-only)        fetch_droid ;;
  --third-party-only)  fetch_third_party ;;
  all|"")              fetch_droid; fetch_third_party ;;
  *) echo "unknown option: $WHAT" >&2; exit 2 ;;
esac

echo ""
echo "Done. Verify with:  bash fetch_assets.sh   (idempotent; re-run is a no-op)"
