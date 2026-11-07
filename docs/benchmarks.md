# Benchmarks

Measured 2026-11-07 on Colab Pro, issues #3, #4 and #10.

**Machine:** NVIDIA A100-SXM4-80GB, driver 580.82.07, CUDA 13.0. 12 vCPU, 167 GB RAM.

**Software:** Python 3.13.16, torch 2.14.1+cu130, ComfyUI v0.44.0, Blender 4.5.10 LTS.

## Setup (`scripts/setup_colab.sh`)

| Step | Time |
|---|---|
| Fresh VM: ComfyUI, package, models, Blender | **138 s** |
| … of which the model download (14 GB, no HF token) | ≈ 65 s |
| Re-run (everything already present) | 10 s |
| Start ComfyUI until `/system_stats` answers | 30 s |

## Stages on A100

Peak VRAM is the maximum `nvidia-smi memory.used` for the GPU as a whole.

| Stage | Settings | First run | Repeat run | Peak VRAM |
|---|---|---|---|---|
| Anima Turbo v1.1 | 832×1216, 10 steps, euler, CFG 1, batch 4 | 18.1 s | **5.1 s** | 11.1 GB |
| BiRefNet + TRELLIS.2 | upsample 1536, 700k faces, 4096² texture (defaults) | 116.9 s | **102.2 s** | 13.6 GB (19.0 GB with Anima still loaded) |
| BiRefNet + TRELLIS.2 | upsample 1024 | — | 72.2 s | 11.7 GB |
| Blender toon render (Cycles OptiX) | 1024², 16 spp, 4 stills, 48-frame turntable, MP4, packed `.blend` | ≈ 11 min (≈ 8 min GPU kernel JIT, see below) | **42 s** | 3.9 GB |

Repeat-run timings for TRELLIS.2 and Blender (102 s and 42 s) are with models cached in RAM and kernels compiled.

Output sizes: GLB ≈ 49 MB, `toon.blend` ≈ 52 MB (textures packed), `turntable.mp4` ≈ 0.8 MB.

**A full job takes about 150 s on a warm runtime:** 5 s Anima + 102 s TRELLIS.2 + 42 s Blender.

## Cost

- The A100 costs about 11–15 compute units per hour. Check the Colab resources panel for the exact rate.
- A warm job therefore costs about **0.5–0.6 CU**. Colab Pro's 100 CU a month buys about 7–9 A100 hours.
- An idle connected runtime also uses units, so disconnect when you are done.

## Findings

- All model files in `config.py` load in ComfyUI v0.44.0. That includes `dino_v3_vit_l.safetensors` as the clip_vision model; no `naf` variant is needed.
- TRELLIS.2 is mostly CPU-bound after sampling: UV unwrap takes about 22 s and texture baking about 21 s. Lowering the upsample resolution to 1024 saves about 30 s per model. Quality at 1024 vs 1536 is still to compare (#9).
- **Cycles GPU kernel JIT:** Blender 4.5 ships no CUDA cubin for sm_80 (A100), and OptiX kernels are PTX on every GPU. So the NVIDIA driver compiles them on the first render of a fresh VM.
  - This took about 8 minutes, and the caches are `~/.nv/ComputeCache` (48 MB) and `/var/tmp/OptixCache_root` (1 MB).
  - `setup_colab.sh` now runs this compile in the background (`src/maker3d/blender/warm_gpu.py`) and copies the cache to Drive. The copy happens only when Drive is mounted, so it can be restored on the next VM.
- Two fixes to the Blender render cut its time from 182 s to 42 s:
  - `emission_sampling = 'NONE'`: the emission-only toon shader was turning every face into a light, which rebuilt the light tree each frame.
  - `use_persistent_data`: textures were being reloaded every frame.
- Blocking `subprocess.run(..., capture_output=True)` cells can hang after Blender finishes. Run long jobs with `Popen` and a log file instead.

## Still to measure

- L4 (24 GB): TRELLIS.2 peaks at 13.6 GB, so it should fit. Blender has an sm_89 cubin, so CUDA won't need the JIT there; OptiX still will.
- Default GPU: **A100**, the owner's choice. The upsample resolution stays at 1536 until #9 compares quality.
