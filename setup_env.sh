#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# RoFacto reproducible conda environment builder.
# Creates a clean, minimal env for the rofacto pipeline (config / dataset /
# train / inference / dexycb render+assemble / eval), pinned to the versions
# used by the verified Wan inference path.
#
# Usage:   bash setup_env.sh [ENV_NAME]        (default ENV_NAME=rofacto)
# Requires: conda on PATH, network access to pypi + the pytorch cu128 index,
#           and a CUDA 12.x capable node for actual GPU runs (build itself is CPU-safe).
#
# NOT covered here: `spider` and `pyroki` envs (separate specialized envs for the
# DexYCB retarget/IK steps — see README_ENV.md).
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

ENV_NAME="${1:-rofacto}"
PYVER="3.10.19"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CU_INDEX="https://download.pytorch.org/whl/cu128"
FLASH_WHL="$HERE/flash_attn-2.8.3+cu128torch2.9-cp310-cp310-linux_x86_64.whl"

# conda activation
if [[ -z "${CONDA_EXE:-}" ]]; then
  source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null \
    || { echo "ERROR: conda not found on PATH" >&2; exit 1; }
fi
source "$(conda info --base)/etc/profile.d/conda.sh"

echo "== [1/5] create env $ENV_NAME (python $PYVER) =="
conda create -y -n "$ENV_NAME" "python=$PYVER"
conda activate "$ENV_NAME"
python -m pip install --upgrade pip

echo "== [2/5] torch stack (cu128 index) =="
pip install --index-url "$CU_INDEX" \
  torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0

echo "== [3/5] pinned requirements =="
# torch trio is re-listed in the lock but already satisfied above (no-op).
# pyrender 0.1.45 hard-pins PyOpenGL==3.1.0; the verified env uses 3.1.4, so
# install pyrender WITHOUT deps and let the lock's PyOpenGL==3.1.4 satisfy it.
REQ_TMP=$(mktemp)
trap 'rm -f "$REQ_TMP"' EXIT
grep -vE '^\s*pyrender==' "$HERE/requirements-lock.txt" > "$REQ_TMP"
pip install -r "$REQ_TMP"
pip install --no-deps pyrender==0.1.45

echo "== [4/5] flash-attn (local prebuilt wheel; optional but matches verified path) =="
if [[ -f "$FLASH_WHL" ]]; then
  pip install "$FLASH_WHL"
else
  echo "WARN: flash-attn wheel not found at $FLASH_WHL — skipping (attention falls back)." >&2
fi

echo "== [5/5] verify imports (CPU) =="
cd "$HERE"
python - <<'PY'
import importlib, sys
ok = True
for m in ["torch","torchvision","diffusers","transformers","accelerate","peft",
          "safetensors","deepspeed","decord","imageio","cv2","pyrender","yourdfpy",
          "trimesh","torchmetrics","skimage","omegaconf","h5py","einops"]:
    try:
        importlib.import_module(m); print(f"  ok  {m}")
    except Exception as e:
        ok = False; print(f"  FAIL {m}: {e}")
import torch
print(f"  torch {torch.__version__} cuda-built={torch.version.cuda}")
try:
    import flash_attn; print(f"  ok  flash_attn {flash_attn.__version__}")
except Exception as e:
    print(f"  warn flash_attn: {e}")
sys.exit(0 if ok else 1)
PY

echo ""
echo "DONE. Activate with:  conda activate $ENV_NAME"
echo "Check the entrypoint: python -m training.wan.inference --help"
