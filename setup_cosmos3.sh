#!/usr/bin/env bash
# Separate environment: does not modify the existing Wan environment.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$ROOT/requirements/cosmos3.lock"
DEST=${1:-"$ROOT/.deps/cosmos-framework"}
command -v uv >/dev/null || { echo 'Install uv >=0.11.3 first.' >&2; exit 1; }
command -v ffmpeg >/dev/null || { echo 'Install ffmpeg and ffprobe first.' >&2; exit 1; }
if [[ ! -e "$DEST" ]]; then
  mkdir -p "$(dirname "$DEST")"
  git clone "$framework_url" "$DEST"
  git -C "$DEST" checkout --detach "$framework_revision"
fi
[[ $(git -C "$DEST" rev-parse HEAD) == "$framework_revision" ]] || {
  echo 'Existing framework has the wrong revision. Use a new destination.' >&2; exit 1;
}
[[ -z $(git -C "$DEST" status --porcelain --untracked-files=no) ]] || {
  echo 'Framework has local changes; refusing to reuse it.' >&2; exit 1;
}
printf '%s  %s\n' "$framework_lock_sha256" "$DEST/uv.lock" | sha256sum --check
export PYTHONNOUSERSITE=1
uv sync --project "$DEST" --python "$python_version" --frozen --group "$dependency_group" --extra train --no-dev
uv pip install --python "${UV_PROJECT_ENVIRONMENT:-$DEST/.venv}/bin/python" --no-deps "torchdata==$torchdata_version" "torchtitan==$torchtitan_version"
printf 'Activate with: source %q/bin/activate\n' "${UV_PROJECT_ENVIRONMENT:-$DEST/.venv}"
printf 'Then run commands from %q (no experiment PYTHONPATH needed).\n' "$ROOT"
