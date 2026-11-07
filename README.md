# 3D-Maker

Type a prompt, get an anime character as a **3D model that looks 2D**.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ashthecool/3D-Maker/blob/main/notebooks/3D_Maker.ipynb)
[![CI](https://github.com/Ashthecool/3D-Maker/actions/workflows/ci.yml/badge.svg)](https://github.com/Ashthecool/3D-Maker/actions/workflows/ci.yml)

> **Status:** early development. Track progress on the [project board](https://github.com/users/Ashthecool/projects/1) and in [issues](https://github.com/Ashthecool/3D-Maker/issues).

## How it works

| Stage | What runs | Output |
|---|---|---|
| 1. Image | **Anima Turbo v1.1** in ComfyUI (CFG 1, ~10 steps) | 4 candidate character images |
| 2. 3D | **BiRefNet** background removal + **TRELLIS.2** (native ComfyUI nodes) | textured `model.glb` |
| 3. 2D look | **Blender 4.5 LTS** headless: cel bands, ink outlines, flat colours, turntable on twos | stills, `turntable.mp4`, `toon.blend` |

Everything runs on a single Colab GPU (L4 recommended). Results are saved to
`MyDrive/3D-Maker/jobs/<timestamp>/`. Open `toon.blend` in Blender 4.5 on your own PC to
pose, tweak and re-render in EEVEE.

## Use it (once v1 is done)

1. Click **Open in Colab** above.
2. Runtime → Change runtime type → **L4 GPU**.
3. Runtime → Run all, then open the Gradio link that appears.

## Licenses

The code here is MIT. The models are downloaded at runtime and keep their own licenses:

- **Anima / Anima Turbo v1.1**: CircleStone Labs **Non-Commercial** License. Use the tool for personal, non-commercial work.
- **TRELLIS.2**: MIT.
- **Blender**: GPL.

## Development

```bash
uv sync --dev
uv run ruff check . && uv run pytest
```

See [CLAUDE.md](CLAUDE.md) for project conventions.
