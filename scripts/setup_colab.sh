#!/usr/bin/env bash
# Idempotent Colab setup (issue #3): ComfyUI (pinned, >=0.34.0), model downloads, Blender 4.5 LTS.
set -euo pipefail

echo "setup_colab.sh: not implemented yet - see issue #3"
nvidia-smi || echo "WARNING: no GPU attached. Runtime > Change runtime type > L4 GPU"
