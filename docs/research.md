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

### Guilty Gear Xrd (reference for "perfect anime" 3D)
Videos the owner shared: Noggi, "I Studied How Guilty Gear Makes 3D Look Perfectly 2D" (https://www.youtube.com/watch?v=tWcaQ3gCbUU). Bran Sculpts, "Making an Anime character in 3D!" (https://www.youtube.com/watch?v=AWocxbVD3CE).

The primary source is Junya Motomura's GDC 2015 talk "GuiltyGearXrd's Art Style: The X Factor Between 2D and 3D". The handout PDF on ggxrd.com refused connections (2026-11-07), so the points below come from a summary of the talk (videohighlight.com/v/yhGjCzxJV3E):
- "Kill everything 3D". A 2D look comes from hand-made decisions, not from physically correct results.
- **Edited vertex normals**: automatic normals don't give clean anime shadow shapes, faces above all. Normals are kept at unit length.
- **Two colour textures**: base (lit) and tint (shadow), so every pixel has its own shadow colour. No specular map.
- **Vertex colours (RGBA)** drive the lighting threshold (shadow bias) and the outline. Self-shadowing was dropped as uncontrollable; shadow areas are painted instead.
- **Inverted-hull outline**, controlled per vertex. **Inner lines** are drawn into textures, with the UV layout set up so the lines stay sharp up close.
- **Limited animation**: every frame is a key pose (no smooth in-betweens). Models are deformed per pose.
- **Fixed character light rig**, independent of stage lighting. Textures are point sampled.

Bran Sculpts' Blender workflow (80.lv interview, 2024-03-12):
- Emission material.
- Grease Pencil **Line Art** modifier for the outlines.
- Edges marked with "Mark Freestyle" for extra inner lines.
- Hand-painted ink lines mixed into the emission colour.

What this pipeline can automate (it has no hand-authored data):
- Smooth or transfer shading normals from a simplified copy, for cleaner cel shapes. This stands in for hand-edited normals.
  - Done (#23): vertex normals blurred along the surface (seams welded first) go into a `toon_normal` attribute; ToonShade's "Smooth Normals" input blends between them and the mesh normals. `--normal-smooth` sets the blur radius as a fraction of the model height. Blurring the normals directly avoids the shrinking of a smoothed copy (fingers and hair strands collapse) and needs no Data Transfer modifier.
- Derive a per-pixel shadow colour from the base texture (a hue-shifted, more saturated darker tone) instead of one flat tint.
  - Done (#21): one cool tint turned skin shadows mauve. The tint now fades out as the base colour gets more saturated: colourful parts darken in their own hue (skin goes rosy, red goes crimson), greyish parts (silver hair, white, black) take the cool tint. Then the saturation is raised ×1.35. Tint, saturation, hue shift and strength are ToonShade inputs.
- Turntable timing on twos or threes.
  - Done (#22): `--hold 2|3` keys a new pose every N frames (constant interpolation) and `--ease` lingers on the front and back views.

![Toon shading before and after #21 and #23: Cycles stills of two TRELLIS.2 characters](img/toon-shading-before-after.jpg)
- Optional Freestyle lines for creases. Freestyle is CPU-based, so it works with headless Cycles. Grease Pencil Line Art needs the GPU draw engine, like EEVEE.
- The Anima image's own line art already survives into the TRELLIS.2 texture bake.

## Body parts and extra views (2026-11-08)
The question: would splitting the character into body parts, and texturing each part separately, fix what goes wrong when TRELLIS.2 does everything in one pass? The problems are skin colour on the back of the head, dark and smudged backs, and colours bleeding from one part into the next. Test characters: the girl and boy jobs from 2026-11-07. Scripts and renders stayed in the session scratchpad.

### Generating each part as its own mesh
| Model | Built on | Textured | Licence / EU | Status (2026-11-08) |
|---|---|---|---|---|
| KaiNinja (Alaya Lab, arXiv 2609.15659) | TRELLIS.2 | yes (TRELLIS.2 materials) | MIT weights, not gated | Weights out 2026-10-23, inference code not yet. Parts come out automatically (no masks); about 24 s per object on an H100. **The one to watch.** |
| OmniPart (HKU, SIGGRAPH Asia 2025) | TRELLIS v1 | yes | MIT | Needs a 2D part mask. Lower resolution than TRELLIS.2, own CUDA stack, community ComfyUI node only. |
| StdGEN (CVPR 2025) | own multi-view model + LRM | yes | Apache-2.0 code; checkpoints "for research purposes only" | Made for anime (separate body, clothes and hair), but old stack (torch 2.1, cu118) and lower texture quality. |
| PartCrafter, PartPacker | TripoSG / own | no | MIT / NVIDIA licence | Geometry only. |
| Hunyuan3D-Part (P3-SAM, X-Part) | Hunyuan3D | — | Tencent licence excludes the EU | **Not used.** |

Splitting the geometry does not fix the texture by itself: each part is still textured from the one front image. Also, the native TRELLIS.2 nodes in ComfyUI v0.44.0 cannot texture a mesh you give them, because there is no node that encodes a mesh into a shape latent. The upstream repo has a texturing pipeline that can.

### Texturing each part (tested)
- **Parts in the 2D image:** See-through's body-part parser (SIGGRAPH 2026). It is SAM-HQ ViT-H fine-tuned on anime, with 19 classes. Weights `24yearsold/l2d_sam_iter2` (1.3 GB, Apache-2.0, not gated); about 20 s per image on a CPU.
  - Clean results on both test images for hair, face, eyes, jacket body, shorts, legs and shoes.
  - Its "handwear" class covers sleeves and hands together; bare thighs and boots can land in "legwear".
  - The alternative, SAM 3.1 with text prompts, is native in ComfyUI (`SAM3_Detect`; `Comfy-Org/sam3.1` is not gated, and the SAM licence covers the EEA). Not tested on anime.
- **Labels onto the mesh:** the colour match's bilateral grid (x/z position plus TRELLIS colour) carries the labels over well.
  - A walk along the surface (each vertex takes the nearest labelled point) leaks wherever TRELLIS.2 fuses parts together: jacket into the back hair, sleeves onto the torso.
  - One rule helps: a face is only on surfaces that point forward. That removes the skin colour on the back of the boy's head.
- **Colours per part:** stops colours bleeding between parts, but 8 colours per part make flat areas blotchy. Use fewer colours per part.

### Other angles of the same character (tested)
- **IP-Adapters for Anima** (community):
  - Anima Ref Adapter v2: ArcAnima node (Apache-2.0) plus ElectronGirl weights (CircleStone non-commercial). Trained on Anima base v1.0; a turbo LoRA works; the full Turbo model is untested.
  - Also LuciferTC's and Wenaka2004's IP-Adapters.
  - Not tested yet: installing a custom node on Colab was not approved in this session.
- **Native, tested: in-context outpainting.**
  - Setup: the picked image goes on the left half of a 1664×1216 canvas, with `SetLatentNoiseMask` on the right half. Prompt: "reference sheet, multiple views, two views, front view and back view, from behind".
  - Result: the same character in 4 of 4 seeds for both test characters, 12–15 s per batch of 4 on an A100 with Anima Turbo v1.1.
  - Colours the prompt leaves open can drift: black boots instead of red, red trim on white sneakers.
- **Projecting the back view onto the back of the mesh** (like the front projection, with the model turned 180°):
  - The best of the 4 back views fits the mesh silhouette with IoU 0.80–0.85, so it can be picked automatically.
  - It brings drawn hair strands, seams and folds to the back.
  - Problems: the colour drift above, and misalignment at the silhouette edges.
- **Turnaround sheet from text alone:** draws a new character each time, so it is not useful for a picked image.

![Back of the model: part labels, colours per part, and a generated back view projected onto the back](img/parts-and-back-view.jpg)

### Plan
1. Back view by in-context outpainting (native nodes); keep the best of 4 by silhouette IoU and project it onto back-facing surfaces (#18).
2. Part labels for the front image and the back view. Snap each part's colours in the back view to the front image's palette for that part (fixes the drift), and use per-part palettes for the sides.
3. Watch for KaiNinja's inference code, which would give separate meshes for hair and clothes.

Found on the way: `toon_scene.py --yaw` did not turn the model for the projection, because glTF imports use quaternion rotation. Fixed by turning the world matrix.

## Colab Pro
- L4 (24 GB) ≈ 1.7–2 CU/h; A100 ≈ 11–15 CU/h; Pro = 100 CU/month. GPU type is not guaranteed.

## Tooling
- Colab MCP (official): `uvx git+https://github.com/googlecolab/colab-mcp`. Proxies to a browser Colab tab; one client at a time.
- Blender MCP: `uvx blender-mcp` plus `addon.py` in Blender (socket server).
