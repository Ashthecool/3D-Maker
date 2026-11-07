# 3D-Maker: notes for Claude

## What this is
A Google Colab Pro tool: prompt → Anima Turbo v1.1 (ComfyUI) → BiRefNet + TRELLIS.2 (native ComfyUI nodes, ≥0.34.0) → Blender 4.5 LTS headless toon render that looks 2D. Outputs go to a job folder on Google Drive. The owner opens `toon.blend` locally.

## Hard constraints
- **The owner is in the EU.** Use only TRELLIS.2 for image-to-3D. Do **not** add Hunyuan3D (license excludes EU/UK/KR) or Pixal3D (HF card blocks EU downloads).
- **Anima is non-commercial.** Never commit model weights. Keep the license notes in README/LICENSE.
- **Engine-agnostic toon shading.** Headless EEVEE on Colab is unreliable, so render with Cycles GPU there. Don't use EEVEE-only nodes such as Shader to RGB. Cel bands come from dot(N, L) → constant ColorRamp; outlines come from an inverted hull masked by Backfacing.
- **GPU-only code runs on Colab.** Local tests and CI are CPU-only. Keep `tests/` free of torch, ComfyUI and bpy imports.

## Layout
- `src/maker3d/config.py`: model files (HF repo + path + ComfyUI subfolder), defaults.
- `src/maker3d/comfy_client.py`: ComfyUI HTTP API (`/prompt`, `/history`, `/view`).
- `src/maker3d/workflows/*.api.json`: API-format workflows exported from the pinned ComfyUI.
- `src/maker3d/blender/toon_scene.py`: runs inside Blender (`blender -b -P ... -- args`).
- `notebooks/3D_Maker.ipynb`: thin launcher. Logic lives in `src/`, not in cells.

## Commands
- `uv sync --dev && uv run ruff check . && uv run ruff format --check . && uv run pytest`
- Local Blender: `C:\Program Files\Blender Foundation\Blender 4.5\blender.exe`

## Tools
- MCP `colab-mcp` (`.mcp.json`): drives a Colab tab in the owner's browser. Call `open_colab_browser_connection` first.
- MCP `blender`: blender-mcp. Needs Blender open with the addon's server started.
- Work is tracked in GitHub issues and the "3D-Maker Roadmap" project. Reference issue numbers in commits (`Fixes #N`).
