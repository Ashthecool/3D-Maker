#!/usr/bin/env bash
# Idempotent Colab setup for 3D-Maker (issue #3).
#   bash scripts/setup_colab.sh [--skip-models] [--skip-blender]
# Installs: ComfyUI (pinned release), this package, model files, Blender 4.5 LTS.
# Safe to re-run: every step skips work that is already done.
set -euo pipefail

SKIP_MODELS=0
SKIP_BLENDER=0
for arg in "$@"; do
  case "$arg" in
    --skip-models) SKIP_MODELS=1 ;;
    --skip-blender) SKIP_BLENDER=1 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cfg() { python -c "import maker3d.config as c; print(c.$1)"; }
step() { printf '\n=== %s ===\n' "$*"; }
t0=$(date +%s)

step "GPU"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
  echo "WARNING: no GPU attached (Runtime > Change runtime type > A100 GPU)."
fi

step "3D-Maker package"
pip install -q -e "$REPO_DIR[colab]"

COMFY_DIR="$(cfg COMFYUI_DIR)"
COMFY_REPO="$(cfg COMFYUI_REPO)"
COMFY_REF="$(cfg COMFYUI_REF)"

step "ComfyUI $COMFY_REF -> $COMFY_DIR"
if [ ! -d "$COMFY_DIR/.git" ]; then
  git -c advice.detachedHead=false clone -q --depth 1 --branch "$COMFY_REF" "$COMFY_REPO" "$COMFY_DIR"
else
  current="$(git -C "$COMFY_DIR" describe --tags --exact-match 2>/dev/null || echo unknown)"
  if [ "$current" != "$COMFY_REF" ]; then
    git -C "$COMFY_DIR" fetch -q --depth 1 origin tag "$COMFY_REF"
    git -C "$COMFY_DIR" -c advice.detachedHead=false checkout -q "$COMFY_REF"
  fi
fi
# Keep Colab's preinstalled torch/torchvision/torchaudio: they already satisfy ComfyUI.
grep -vE '^(torch|torchvision|torchaudio)([<>=!~ ]|$)' "$COMFY_DIR/requirements.txt" \
  > /tmp/comfy-reqs.txt
pip install -q -r /tmp/comfy-reqs.txt
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| gpu', torch.cuda.is_available())"

if [ "$SKIP_MODELS" = 0 ]; then
  step "Models"
  python -m maker3d.download --comfy-dir "$COMFY_DIR"
fi

if [ "$SKIP_BLENDER" = 0 ]; then
  BLENDER_DIR="$(cfg BLENDER_DIR)"
  BLENDER_URL="$(cfg BLENDER_URL)"
  step "Blender -> $BLENDER_DIR"
  if [ ! -x "$BLENDER_DIR/blender" ]; then
    # Shared libraries Blender needs even in background mode.
    apt-get -qq update >/dev/null
    apt-get -qq install -y libxi6 libxkbcommon0 libxrender1 libxxf86vm1 libxfixes3 \
      libgl1 libegl1 libsm6 >/dev/null
    curl -sSL "$BLENDER_URL" | tar -xJ -C /content
  fi
  ln -sf "$BLENDER_DIR/blender" /usr/local/bin/blender
  blender --version | head -1
fi

step "Done in $(( $(date +%s) - t0 ))s"
