"""Central configuration: model files, folders and generation defaults.

``verified=True`` means the file was confirmed in the Hugging Face repo listing
(2026-11-07). Whether each one loads in the pinned ComfyUI is checked in issue #4.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelFile:
    """A single file downloaded from the Hugging Face Hub into a ComfyUI models folder."""

    repo_id: str
    path_in_repo: str
    comfy_subdir: str
    verified: bool = False

    @property
    def filename(self) -> str:
        return self.path_in_repo.rsplit("/", 1)[-1]

    @property
    def url(self) -> str:
        return f"https://huggingface.co/{self.repo_id}/resolve/main/{self.path_in_repo}"


# --- Stage 1: Anima Turbo v1.1 (CircleStone Labs Non-Commercial License) ---
ANIMA_MODELS: tuple[ModelFile, ...] = (
    ModelFile(
        "circlestone-labs/Anima",
        "split_files/diffusion_models/anima-turbo-v1.1.safetensors",
        "diffusion_models",
        verified=True,
    ),
    ModelFile(
        "circlestone-labs/Anima",
        "split_files/text_encoders/qwen_3_06b_base.safetensors",
        "text_encoders",
        verified=True,
    ),
    ModelFile(
        "circlestone-labs/Anima",
        "split_files/vae/qwen_image_vae.safetensors",
        "vae",
        verified=True,
    ),
)

# --- Stage 2: TRELLIS.2 native ComfyUI nodes (MIT) + BiRefNet background removal ---
TRELLIS2_MODELS: tuple[ModelFile, ...] = (
    ModelFile(
        "Comfy-Org/TRELLIS.2",
        "diffusion_models/trellis_2_int8_convrot.safetensors",
        "diffusion_models",
        verified=True,
    ),
    ModelFile(
        "Comfy-Org/TRELLIS.2",
        "vae/trellis_2_shape_vae_bf16.safetensors",
        "vae",
        verified=True,
    ),
    ModelFile(
        "Comfy-Org/TRELLIS.2",
        "vae/trellis_2_texture_vae_bf16.safetensors",
        "vae",
        verified=True,
    ),
    ModelFile(
        "Comfy-Org/TRELLIS.2",
        "clip_vision/dino_v3_vit_l.safetensors",
        "clip_vision",
        verified=True,
    ),
    ModelFile(
        "Comfy-Org/BiRefNet",
        "background_removal/birefnet.safetensors",
        "background_removal",
        verified=True,
    ),
)

ALL_MODELS: tuple[ModelFile, ...] = ANIMA_MODELS + TRELLIS2_MODELS

# --- Runtime locations on Colab ---
COMFYUI_DIR = "/content/ComfyUI"
COMFYUI_URL = "http://127.0.0.1:8188"
COMFYUI_MIN_VERSION = "0.34.0"  # first release with native TRELLIS.2 nodes
COMFYUI_REPO = "https://github.com/Comfy-Org/ComfyUI"
COMFYUI_REF = "v0.44.0"  # pinned release (2026-11-04)
DRIVE_JOBS_DIR = "/content/drive/MyDrive/3D-Maker/jobs"
BLENDER_VERSION = "4.5.10"
BLENDER_DIR = f"/content/blender-{BLENDER_VERSION}-linux-x64"
BLENDER_URL = (
    f"https://download.blender.org/release/Blender4.5/blender-{BLENDER_VERSION}-linux-x64.tar.xz"
)


@dataclass(frozen=True)
class AnimaDefaults:
    """Model card: Turbo uses CFG 1 and 8-12 steps; euler works well with Turbo."""

    steps: int = 10
    cfg: float = 1.0
    sampler: str = "euler"
    scheduler: str = "simple"
    width: int = 832
    height: int = 1216
    batch_size: int = 4


TOON_STYLES: tuple[str, ...] = ("flat", "cel", "cel_rim")
DEFAULT_TOON_STYLE = "cel"
