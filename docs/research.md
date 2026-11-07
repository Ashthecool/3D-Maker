# Research notes (2026-11-07)

## Anima Turbo v1.1
- Official file: `circlestone-labs/Anima` → `split_files/diffusion_models/anima-turbo-v1.1.safetensors` (4.18 GB).
- Text encoder: `split_files/text_encoders/qwen_3_06b_base.safetensors`. VAE: `split_files/vae/qwen_image_vae.safetensors`.
- Architecture: derivative of NVIDIA Cosmos-Predict2-2B with a Qwen3-0.6B text encoder. Native in ComfyUI only (no official diffusers pipeline).
- Turbo settings (model card): CFG 1, 8–12 steps, euler, 512²–1536². Distillation gives a strong default style but less diversity.
- Prompts: Danbooru tags (lowercase, spaces) and/or natural language. Prefix `masterpiece, best quality, score_7, safe`. Artist tags need `@`.
- License: CircleStone Labs Non-Commercial (the model card says outputs are not restricted).
- Low-VRAM GGUF quantizations: `vanes430/Anima-Turbo-V1.1-GGUF` (needs ComfyUI-GGUF, recommends euler + sgm_uniform, 4–8 steps).

## Image → 3D
- **TRELLIS.2** (Microsoft, MIT): native ComfyUI core nodes since 0.34.0 (PR #14718, Aug 2026). No CUDA compilation.
  - Files in `Comfy-Org/TRELLIS.2`: `trellis_2_int8_convrot`, shape and texture VAEs, DINOv3 for clip_vision. Background removal: `Comfy-Org/BiRefNet`.
  - Output: GLB with baked base color, normal and AO. Template: "Pixal3D & TRELLIS.2: Image to Model".
  - Upstream repo needs ≥24 GB VRAM and Linux. The int8 native build's VRAM is still unmeasured (issue #4).
- **Pixal3D**: closer alignment to the input image, but the HF card gates out EU users. **Not used.**
- **Hunyuan3D 2.1**: license excludes the EU, UK and South Korea. **Not used.**
- **StdGEN** (anime-specific, semantic parts): an option to revisit if TRELLIS.2 struggles with anime input.

## 2D look in Blender
- Reference: Vertex Arcade, "Making 3D Look Like 2D in Blender" (https://www.youtube.com/watch?v=eHddD5OfuwI).
- Headless EEVEE on Linux needs OpenGL/EGL and is unreliable in containers. Cycles GPU works headless, so the shader avoids EEVEE-only nodes.

## Colab Pro
- L4 (24 GB) ≈ 1.7–2 CU/h; A100 ≈ 11–15 CU/h; Pro = 100 CU/month. GPU type is not guaranteed.

## Tooling
- Colab MCP (official): `uvx git+https://github.com/googlecolab/colab-mcp`. Proxies to a browser Colab tab; one client at a time.
- Blender MCP: `uvx blender-mcp` plus `addon.py` in Blender (socket server).
