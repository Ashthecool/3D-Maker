"""Central configuration: model files, folders and generation defaults.

Model paths marked ``verified=False`` come from docs and still need checking
on Colab (issue #4).
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
    ),
    ModelFile("Comfy-Org/TRELLIS.2", "vae/trellis_2_shape_vae_bf16.safetensors", "vae"),
    ModelFile("Comfy-Org/TRELLIS.2", "vae/trellis_2_texture_vae_bf16.safetensors", "vae"),
    ModelFile("Comfy-Org/TRELLIS.2", "clip_vision/dino_v3_vit_l.safetensors", "clip_vision"),
    ModelFile(
        "Comfy-Org/BiRefNet",
        "background_removal/birefnet.safetensors",
        "background_removal",
    ),
)

ALL_MODELS: tuple[ModelFile, ...] = ANIMA_MODELS + TRELLIS2_MODELS

# --- Runtime locations on Colab ---
COMFYUI_DIR = "/content/ComfyUI"
COMFYUI_URL = "http://127.0.0.1:8188"
COMFYUI_MIN_VERSION = "0.34.0"  # first release with native TRELLIS.2 nodes
DRIVE_JOBS_DIR = "/content/drive/MyDrive/3D-Maker/jobs"
BLENDER_VERSION = "4.5.10"


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
