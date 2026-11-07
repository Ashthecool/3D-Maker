"""ComfyUI API-format workflows built in Python (issues #6 and #8).

Node types, input names and default values come from ComfyUI v0.44.0
(``/object_info``) and the official templates ``image_anima_base_v1.json`` and
``3d_pixal3d_trellis2_image_to_model.json`` (TRELLIS.2 branch only: the
Pixal3D/MoGe nodes are left out on purpose, see CLAUDE.md).

A workflow is ``{node_id: {"class_type": ..., "inputs": {...}}}``. A link to
another node's output is ``[node_id, output_index]``.
"""

from __future__ import annotations

from dataclasses import dataclass

from maker3d.config import AnimaDefaults
from maker3d.prompts import NEGATIVE

Workflow = dict[str, dict]


class _Graph:
    def __init__(self) -> None:
        self.nodes: Workflow = {}

    def add(self, class_type: str, title: str = "", **inputs) -> str:
        node_id = str(len(self.nodes) + 1)
        node: dict = {"class_type": class_type, "inputs": inputs}
        if title:
            node["_meta"] = {"title": title}
        self.nodes[node_id] = node
        return node_id


def out(node_id: str, index: int = 0) -> list:
    """Reference output ``index`` of ``node_id``."""
    return [node_id, index]


# --------------------------------------------------------------------------------------
# Stage 1: Anima Turbo v1.1 text-to-image
# --------------------------------------------------------------------------------------
def anima_t2i(
    prompt: str,
    seed: int,
    negative: str = NEGATIVE,
    params: AnimaDefaults | None = None,
    filename_prefix: str = "maker3d/anima",
) -> Workflow:
    p = params or AnimaDefaults()
    g = _Graph()
    unet = g.add("UNETLoader", unet_name="anima-turbo-v1.1.safetensors", weight_dtype="default")
    clip = g.add(
        "CLIPLoader",
        clip_name="qwen_3_06b_base.safetensors",
        type="stable_diffusion",  # as in the official Anima template
        device="default",
    )
    vae = g.add("VAELoader", vae_name="qwen_image_vae.safetensors")
    pos = g.add("CLIPTextEncode", "prompt", text=prompt, clip=out(clip))
    neg = g.add("CLIPTextEncode", "negative", text=negative, clip=out(clip))
    latent = g.add(
        "EmptyLatentImage", "latent", width=p.width, height=p.height, batch_size=p.batch_size
    )
    sample = g.add(
        "KSampler",
        "sampler",
        model=out(unet),
        seed=seed,
        steps=p.steps,
        cfg=p.cfg,
        sampler_name=p.sampler,
        scheduler=p.scheduler,
        positive=out(pos),
        negative=out(neg),
        latent_image=out(latent),
        denoise=1.0,
    )
    decoded = g.add("VAEDecode", samples=out(sample), vae=out(vae))
    g.add("SaveImage", "save", images=out(decoded), filename_prefix=filename_prefix)
    return g.nodes


# --------------------------------------------------------------------------------------
# Stage 2: BiRefNet + TRELLIS.2 image-to-model
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Trellis2Settings:
    """Defaults copied from the official template's TRELLIS.2 branch."""

    remove_background: bool = True
    crop_size: int = 1024
    crop_pad: float = 1.1
    upsample_resolution: int = 1536  # 1024..2048
    remesh_resolution: int = 768
    decimate_faces: int = 700_000
    texture_size: int = 4096
    dino: str = "dino_v3_vit_l.safetensors"


def trellis2_i2m(
    image_name: str,
    seed: int,
    settings: Trellis2Settings | None = None,
    filename_prefix: str = "maker3d/model",
) -> Workflow:
    s = settings or Trellis2Settings()
    g = _Graph()

    # Loaders
    unet = g.add(
        "UNETLoader", unet_name="trellis_2_int8_convrot.safetensors", weight_dtype="default"
    )
    shape_vae = g.add("VAELoader", vae_name="trellis_2_shape_vae_bf16.safetensors")
    tex_vae = g.add("VAELoader", vae_name="trellis_2_texture_vae_bf16.safetensors")
    dino = g.add("CLIPVisionLoader", clip_name=s.dino)

    # Input image -> mask -> centred square crop
    image = g.add("LoadImage", "image", image=image_name)
    if s.remove_background:
        bg_model = g.add("LoadBackgroundRemovalModel", bg_removal_name="birefnet.safetensors")
        mask = out(g.add("RemoveBackground", bg_removal_model=out(bg_model), image=out(image)))
        # Saved for the Blender front projection (toon_scene.py --source-mask).
        mask_img = g.add("MaskToImage", mask=mask)
        g.add("SaveImage", "mask", images=out(mask_img), filename_prefix=f"{filename_prefix}_mask")
    else:
        mask = out(image, 1)  # the PNG's own alpha channel
    crop = g.add(
        "ImageCropToMask",
        images=out(image),
        masks=mask,
        width=s.crop_size,
        height=s.crop_size,
        pad_factor=s.crop_pad,
        grow_mask=0,
        background="#000000",
    )
    cond = g.add("Trellis2Conditioning", clip_vision_model=out(dino), image=out(crop))

    # Model variants (CFG schedule matches the original TRELLIS.2 pipeline)
    m_struct = g.add("CFGOverride", model=out(unet), cfg=1.0, start_percent=0.667, end_percent=1.0)
    m_struct = g.add("RescaleCFG", model=out(m_struct), multiplier=0.7)
    m_struct = g.add("ModelSamplingSD3", model=out(m_struct), shift=5.0)
    m_shape = g.add("CFGOverride", model=out(unet), cfg=1.0, start_percent=0.769, end_percent=1.0)
    m_shape = g.add("RescaleCFG", model=out(m_shape), multiplier=0.5)

    def ksampler(title, model, pos, neg, latent, seed_, steps, cfg, scheduler):
        return g.add(
            "KSampler",
            title,
            model=out(model),
            seed=seed_,
            steps=steps,
            cfg=cfg,
            sampler_name="euler",
            scheduler=scheduler,
            positive=pos,
            negative=neg,
            latent_image=latent,
            denoise=1.0,
        )

    # 1) sparse structure
    empty = g.add("EmptyTrellis2LatentStructure", batch_size=1)
    s1 = ksampler(
        "structure", m_struct, out(cond, 0), out(cond, 1), out(empty), seed, 12, 7.5, "normal"
    )
    voxel = g.add(
        "VaeDecodeStructureTrellis2", samples=out(s1), vae=out(shape_vae), resolution="32"
    )

    # 2) shape, then upsample to the target resolution
    shape = g.add(
        "Trellis2ShapeStage", positive=out(cond, 0), negative=out(cond, 1), voxel=out(voxel)
    )
    s2 = ksampler(
        "shape", m_shape, out(shape, 0), out(shape, 1), out(shape, 2), seed, 20, 7.5, "normal"
    )
    up = g.add(
        "Trellis2UpsampleStage",
        positive=out(shape, 0),
        negative=out(shape, 1),
        shape_latent=out(s2),
        vae=out(shape_vae),
        target_resolution=s.upsample_resolution,
    )
    s3 = ksampler("upsample", m_shape, out(up, 0), out(up, 1), out(up, 2), seed, 12, 7.5, "simple")
    hi_mesh = g.add("VaeDecodeShapeTrellis", samples=out(s3), vae=out(shape_vae))

    # 3) texture voxels
    tex = g.add(
        "Trellis2TextureStage", positive=out(up, 0), negative=out(up, 1), shape_latent=out(s3)
    )
    s4 = ksampler(
        "texture", unet, out(tex, 0), out(tex, 1), out(tex, 2), seed + 1, 12, 1.0, "normal"
    )
    voxel_colors = g.add(
        "VaeDecodeTextureTrellis",
        samples=out(s4),
        vae=out(tex_vae),
        shape_subdivides=out(hi_mesh, 1),
    )

    # 4) clean low-poly mesh + UVs
    remesh = g.add(
        "RemeshMesh",
        mesh=out(hi_mesh),
        resolution=s.remesh_resolution,
        sign_mode="solid",
        **{"sign_mode.fill": 20.0, "sign_mode.qef": False},
        band=1.0,
        project_back=0.0,
        fix_poles=False,
        smooth_iters=20,
        drop_small_components=0.01,
        precluster_max_verts=20_000_000,
    )
    low = g.add(
        "DecimateMesh",
        mesh=out(remesh),
        target_face_count=s.decimate_faces,
        placement_mode="midpoint",
    )
    low = g.add("MeshSmoothNormals", mesh=out(low), crease_angle=180.0)
    uv = g.add(
        "UnwrapMesh",
        mesh=out(low),
        segmenter="pec",
        resolution=s.texture_size,
        padding=1,
        weld_distance=0.0002,
    )

    # 5) bake textures and save
    bake = g.add(
        "BakeTextureFromVoxel",
        mesh=out(uv),
        voxel_colors=out(voxel_colors),
        texture_size=s.texture_size,
        reference_mesh=out(hi_mesh),
    )
    normal = g.add(
        "BakeNormalMapFromMesh",
        low_poly=out(uv),
        high_poly=out(remesh),
        resolution=2048,
        cage_distance=0.05,
        ignore_backfaces=True,
    )
    ao = g.add(
        "BakeAmbientOcclusion",
        low_poly=out(uv),
        high_poly=out(remesh),
        resolution=1024,
        samples=64,
        max_distance=0.71,
        strength=1.0,
        bias=0.01,
    )
    textured = g.add(
        "ApplyTextureToMesh",
        mesh=out(uv),
        base_color=out(bake, 0),
        metallic=out(bake, 1),
        roughness=out(bake, 2),
        occlusion=out(ao),
        normal_map=out(normal),
    )
    final = g.add("MeshSmoothNormals", mesh=out(textured), crease_angle=180.0)
    g.add("SaveGLB", "save", mesh=out(final), filename_prefix=filename_prefix)
    return g.nodes
