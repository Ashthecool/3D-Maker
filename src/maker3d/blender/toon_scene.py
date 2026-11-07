"""Blender script that turns a TRELLIS.2 GLB into a 2D-looking toon scene (issues #11-#13).

Run inside Blender (not regular Python):
    blender -b --factory-startup -P toon_scene.py -- --glb model.glb --out job_dir --style cel

Writes to ``--out``: ``stills/{front,three_quarter,side,back}.png``, ``turntable.mp4``
(if ffmpeg is on PATH; the PNG frames are kept in ``frames/``) and a packed ``toon.blend``.

Design (engine-agnostic, so Cycles can render headless on Colab and EEVEE works locally):
- Cel bands: dot(normal, light_dir) -> constant ColorRamp, multiplied with the base-color
  texture and output as Emission. Avoids EEVEE-only Shader-to-RGB.
- Outline: inverted hull (Solidify, flipped normals) with a black material masked by
  Geometry > Backfacing instead of the backface-culling flag.
- Color management: Standard view transform (not AgX) so flat colours stay flat.
- Camera: orthographic; turntable keyed on twos (constant interpolation every 2 frames).

Everything that does not need ``bpy`` is importable from plain Python for tests.
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

STYLES = ("flat", "cel", "cel_rim")
LIGHT_DIR = (-0.45, -0.65, 0.62)  # from camera-left, front and above (Blender: -Y is front)
SHADOW_TINT = (0.62, 0.58, 0.78)  # bluish-purple shadow multiplier
STILLS = {"front": 0.0, "three_quarter": 45.0, "side": 90.0, "back": 180.0}


# --------------------------------------------------------------------------------------
# Pure helpers (no bpy)
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Band:
    position: float  # 0..1 on the half-Lambert factor
    value: tuple[float, float, float]  # RGB multiplier for the base colour


def cel_bands(style: str, bands: int) -> list[Band]:
    """Constant ColorRamp stops for ``style``. ``flat`` has a single full-bright band."""
    if style not in STYLES:
        raise ValueError(f"unknown style {style!r}; choose from {STYLES}")
    if style == "flat":
        return [Band(0.0, (1.0, 1.0, 1.0))]
    if bands not in (2, 3):
        raise ValueError("bands must be 2 or 3")
    lit = (1.0, 1.0, 1.0)
    if bands == 2:
        return [Band(0.0, SHADOW_TINT), Band(0.5, lit)]
    mid = tuple(round((s + 1.0) / 2, 4) for s in SHADOW_TINT)
    return [Band(0.0, SHADOW_TINT), Band(0.42, mid), Band(0.62, lit)]


def normalized(v: tuple[float, float, float]) -> tuple[float, float, float]:
    n = math.sqrt(sum(c * c for c in v)) or 1.0
    return tuple(c / n for c in v)


def outline_thickness(height: float, factor: float) -> float:
    """Outline shell thickness in scene units, proportional to the model height."""
    return max(height, 1e-6) * factor


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse the arguments after ``--`` on Blender's command line."""
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    p = argparse.ArgumentParser(prog="toon_scene.py", description=__doc__.split("\n")[0])
    p.add_argument("--glb", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--style", choices=STYLES, default="cel")
    p.add_argument("--bands", type=int, choices=(2, 3), default=2)
    p.add_argument("--outline", type=float, default=0.004, help="fraction of model height")
    p.add_argument("--no-outline", action="store_true")
    p.add_argument("--res", type=int, default=1024)
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--frames", type=int, default=96, help="turntable length at 24 fps")
    p.add_argument("--no-turntable", action="store_true")
    p.add_argument(
        "--yaw", type=float, default=0.0, help="extra rotation (deg) so the model faces -Y"
    )
    p.add_argument("--transparent", action="store_true", help="transparent background")
    p.add_argument("--background", default="#ffffff")
    p.add_argument("--engine", choices=("CYCLES", "EEVEE"), default="CYCLES")
    p.add_argument("--cpu", action="store_true", help="render Cycles on the CPU")
    p.add_argument(
        "--source-image",
        type=Path,
        help="the front image TRELLIS.2 got; projected onto front-facing surfaces",
    )
    p.add_argument("--source-mask", type=Path, help="background-removal mask of --source-image")
    return p.parse_args(argv)


# --------------------------------------------------------------------------------------
# Front projection helpers (numpy only, no bpy). Image arrays use Blender's row order:
# row 0 is the BOTTOM of the picture, so pixel y grows upwards like world Z.
# --------------------------------------------------------------------------------------
def erode(mask, px: int):
    """Binary erosion with a 4-neighbourhood, ``px`` times."""
    m = mask.astype(bool)
    for _ in range(max(px, 0)):
        e = m.copy()
        e[1:, :] &= m[:-1, :]
        e[:-1, :] &= m[1:, :]
        e[:, 1:] &= m[:, :-1]
        e[:, :-1] &= m[:, 1:]
        m = e
    return m


def dilate(mask, px: int):
    return ~erode(~mask.astype(bool), px)


def splat(px_x, px_y, shape):
    """Boolean image with a True pixel under every point (points outside are dropped)."""
    import numpy as np

    h, w = shape
    img = np.zeros(shape, bool)
    xi, yi = np.round(px_x).astype(int), np.round(px_y).astype(int)
    ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    img[yi[ok], xi[ok]] = True
    return img


def background_mask(rgba, thresh: float = 0.94):
    """Foreground mask for a picture on a white background: near-white pixels that are
    connected to the border are background. Fallback when no segmentation mask is given."""
    import numpy as np

    cand = rgba[..., :3].min(axis=2) > thresh
    bg = np.zeros_like(cand)
    for sl in (np.s_[0, :], np.s_[-1, :], np.s_[:, 0], np.s_[:, -1]):
        bg[sl] = cand[sl]
    while True:
        grown = dilate(bg, 1) & cand
        if (grown == bg).all():
            return ~bg
        bg = grown


def fit_projection(xz, mask, down: int = 4, scales=None, shift: int = 6):
    """Fit ``pixel = s * (x, z) + (ox, oy)`` so the mesh's front silhouette matches ``mask``.

    ``xz``: (N, 2) mesh vertex positions seen from the front (x right, z up).
    Starts from the bounding boxes (uniform scale from the height), then searches scale and
    shift for the best silhouette IoU on a ``down``-times smaller grid.
    Returns ``(s, ox, oy, iou)`` in full-resolution pixels.
    """
    import numpy as np

    if scales is None:
        scales = np.linspace(0.94, 1.06, 13)
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0:
        raise ValueError("empty mask")
    x, z = xz[:, 0], xz[:, 1]
    s0 = (rows[-1] - rows[0] + 1) / max(float(z.max() - z.min()), 1e-9)
    cx_img, cy_img = (cols[0] + cols[-1]) / 2, (rows[0] + rows[-1]) / 2
    cx_mesh, cz_mesh = (x.min() + x.max()) / 2, (z.min() + z.max()) / 2
    small = mask[::down, ::down].astype(bool)
    best = (-1.0, s0, cx_img - s0 * cx_mesh, cy_img - s0 * cz_mesh)
    for k in scales:
        s = s0 * k
        ox, oy = cx_img - s * cx_mesh, cy_img - s * cz_mesh
        # closing (dilate then erode) fills gaps between points without growing the outline
        sil = erode(dilate(splat((s * x + ox) / down, (s * z + oy) / down, small.shape), 1), 1)
        for dy in range(-shift, shift + 1):
            for dx in range(-shift, shift + 1):
                moved = np.roll(sil, (dy, dx), axis=(0, 1))
                union = (moved | small).sum()
                iou = (moved & small).sum() / union if union else 0.0
                if iou > best[0]:
                    best = (iou, s, ox + dx * down, oy + dy * down)
    iou, s, ox, oy = best
    return s, ox, oy, float(iou)


def smoothstep(e0: float, e1: float, x):
    import numpy as np

    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def hex_to_rgb(value: str) -> tuple[float, float, float]:
    value = value.lstrip("#")
    if len(value) != 6:
        raise ValueError(f"expected #rrggbb, got {value!r}")
    srgb = [int(value[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    # sRGB -> linear, since Blender colour inputs are scene-linear
    return tuple(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in srgb)


# --------------------------------------------------------------------------------------
# Blender side
# --------------------------------------------------------------------------------------
def main(argv: list[str]) -> None:
    import bpy
    from mathutils import Vector

    args = parse_args(argv)
    out = args.out.resolve()
    (out / "stills").mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene

    # --- clean scene + import -----------------------------------------------------
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    bpy.ops.import_scene.gltf(filepath=str(args.glb.resolve()))
    meshes = [o for o in scene.objects if o.type == "MESH"]
    if not meshes:
        raise SystemExit(f"no meshes in {args.glb}")
    proj_image = None
    if args.source_image:
        rotate_yaw = math.radians(args.yaw)
        for o in [o for o in scene.objects if o.parent is None]:
            o.rotation_euler.z += rotate_yaw  # project in the pose the camera sees at yaw
        bpy.context.view_layer.update()
        proj_image = add_front_projection(bpy, meshes, args.source_image, args.source_mask)
        for o in [o for o in scene.objects if o.parent is None]:
            o.rotation_euler.z -= rotate_yaw
        bpy.context.view_layer.update()

    # Pivot empty: everything rotates around the model's vertical axis.
    corners = [o.matrix_world @ Vector(c) for o in meshes for c in o.bound_box]
    lo = Vector((min(c.x for c in corners), min(c.y for c in corners), min(c.z for c in corners)))
    hi = Vector((max(c.x for c in corners), max(c.y for c in corners), max(c.z for c in corners)))
    size = hi - lo
    height = size.z
    pivot = bpy.data.objects.new("ToonPivot", None)
    scene.collection.objects.link(pivot)
    # Stand the model on the origin, centred on its vertical axis, then parent it to the
    # pivot (which sits at the origin with an identity transform, so no inverse is needed).
    offset = Vector(((lo.x + hi.x) / 2, (lo.y + hi.y) / 2, lo.z))
    for o in [o for o in scene.objects if o.parent is None and o is not pivot]:
        o.location -= offset
        o.parent = pivot
    base_yaw = math.radians(args.yaw)
    pivot.rotation_euler = (0.0, 0.0, base_yaw)

    # --- materials ------------------------------------------------------------------
    toon_group = build_toon_group(bpy, args.style, args.bands)
    for mat in {s.material for o in meshes for s in o.material_slots if s.material}:
        make_toon_material(bpy, mat, toon_group)
        if proj_image is not None:
            insert_projection(bpy, mat, proj_image)
    if not args.no_outline:
        outline_mat = build_outline_material(bpy)
        thickness = outline_thickness(height, args.outline)
        for o in meshes:
            add_outline(o, outline_mat, thickness)

    # --- world, colour management, camera -------------------------------------------
    world = bpy.data.worlds.new("ToonWorld")
    scene.world = world
    world.use_nodes = True
    bg = next(n for n in world.node_tree.nodes if n.type == "BACKGROUND")
    bg.inputs["Color"].default_value = (*hex_to_rgb(args.background), 1.0)
    bg.inputs["Strength"].default_value = 1.0
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.render.film_transparent = args.transparent

    cam_data = bpy.data.cameras.new("ToonCam")
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = max(height, size.x, size.y) * 1.12
    cam = bpy.data.objects.new("ToonCam", cam_data)
    scene.collection.objects.link(cam)
    dist = max(size) * 4 + 1.0
    cam.location = (0.0, -dist, height / 2)
    cam.rotation_euler = (math.radians(90), 0.0, 0.0)
    cam_data.clip_end = dist * 3
    scene.camera = cam

    # --- render settings ------------------------------------------------------------
    r = scene.render
    r.resolution_x = r.resolution_y = args.res
    r.resolution_percentage = 100
    r.fps = 24
    r.image_settings.file_format = "PNG"
    r.image_settings.color_mode = "RGBA" if args.transparent else "RGB"
    setup_engine(bpy, scene, args)

    # --- stills ---------------------------------------------------------------------
    for name, deg in STILLS.items():
        pivot.rotation_euler.z = base_yaw + math.radians(deg)
        r.filepath = str(out / "stills" / f"{name}.png")
        bpy.ops.render.render(write_still=True)
        print(f"[toon] still {name} -> {r.filepath}", flush=True)
    pivot.rotation_euler.z = base_yaw

    # --- turntable on twos ----------------------------------------------------------
    scene.frame_start, scene.frame_end = 1, args.frames
    key_turntable(pivot, base_yaw, args.frames)
    if not args.no_turntable:
        frames_dir = out / "frames"
        frames_dir.mkdir(exist_ok=True)
        scene.frame_step = 2  # only the frames that change
        r.filepath = str(frames_dir / "f_")
        bpy.ops.render.render(animation=True)
        scene.frame_step = 1
        encode_mp4(frames_dir, out / "turntable.mp4", args.background)

    # --- save -----------------------------------------------------------------------
    scene.frame_set(1)
    r.filepath = "//render/"
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(out / "toon.blend"), compress=True)
    print(f"[toon] saved {out / 'toon.blend'}", flush=True)


def build_toon_group(bpy, style: str, bands: int):
    """Node group: Color in -> toon-shaded Color out. Light direction lives here once."""
    g = bpy.data.node_groups.new("ToonShade", "ShaderNodeTree")
    g.interface.new_socket("Color", in_out="INPUT", socket_type="NodeSocketColor")
    g.interface.new_socket("Color", in_out="OUTPUT", socket_type="NodeSocketColor")
    n, links = g.nodes, g.links
    gin, gout = n.new("NodeGroupInput"), n.new("NodeGroupOutput")
    gin.location, gout.location = (-900, 0), (500, 0)

    geo = n.new("ShaderNodeNewGeometry")
    geo.location = (-900, -250)
    light = n.new("ShaderNodeCombineXYZ")
    light.label = "Light direction (world)"
    light.location = (-900, -450)
    for axis, value in zip("XYZ", normalized(LIGHT_DIR), strict=True):
        light.inputs[axis].default_value = value
    dot = n.new("ShaderNodeVectorMath")
    dot.operation = "DOT_PRODUCT"
    dot.location = (-650, -300)
    links.new(geo.outputs["Normal"], dot.inputs[0])
    links.new(light.outputs["Vector"], dot.inputs[1])
    half = n.new("ShaderNodeMapRange")  # half-Lambert: -1..1 -> 0..1
    half.location = (-450, -300)
    half.inputs["From Min"].default_value = -1.0
    links.new(dot.outputs["Value"], half.inputs["Value"])

    ramp = n.new("ShaderNodeValToRGB")
    ramp.label = "Cel bands"
    ramp.location = (-250, -300)
    ramp.color_ramp.interpolation = "CONSTANT"
    stops = cel_bands(style, bands)
    elems = ramp.color_ramp.elements
    while len(elems) > 1:
        elems.remove(elems[-1])
    elems[0].position, elems[0].color = stops[0].position, (*stops[0].value, 1.0)
    for band in stops[1:]:
        e = elems.new(band.position)
        e.color = (*band.value, 1.0)
    links.new(half.outputs["Result"], ramp.inputs["Fac"])

    mul = n.new("ShaderNodeMix")
    mul.data_type = "RGBA"
    mul.blend_type = "MULTIPLY"
    mul.location = (50, 0)
    mul.inputs["Factor"].default_value = 1.0
    links.new(gin.outputs["Color"], mul.inputs["A"])
    links.new(ramp.outputs["Color"], mul.inputs["B"])
    result = mul.outputs["Result"]

    if style == "cel_rim":
        rim = n.new("ShaderNodeLayerWeight")
        rim.location = (-450, 250)
        rim.inputs["Blend"].default_value = 0.35
        rim_ramp = n.new("ShaderNodeValToRGB")
        rim_ramp.label = "Rim"
        rim_ramp.location = (-250, 250)
        rim_ramp.color_ramp.interpolation = "CONSTANT"
        rim_ramp.color_ramp.elements[0].color = (0, 0, 0, 1)
        rim_ramp.color_ramp.elements[1].position = 0.6
        rim_ramp.color_ramp.elements[1].color = (0.35, 0.35, 0.35, 1)
        links.new(rim.outputs["Facing"], rim_ramp.inputs["Fac"])
        lit_only = n.new("ShaderNodeMix")  # rim only on the lit side
        lit_only.data_type = "RGBA"
        lit_only.blend_type = "MULTIPLY"
        lit_only.location = (50, 250)
        lit_only.inputs["Factor"].default_value = 1.0
        links.new(rim_ramp.outputs["Color"], lit_only.inputs["A"])
        links.new(ramp.outputs["Color"], lit_only.inputs["B"])
        add = n.new("ShaderNodeMix")
        add.data_type = "RGBA"
        add.blend_type = "ADD"
        add.location = (250, 0)
        add.inputs["Factor"].default_value = 1.0
        links.new(result, add.inputs["A"])
        links.new(lit_only.outputs["Result"], add.inputs["B"])
        result = add.outputs["Result"]

    links.new(result, gout.inputs["Color"])
    return g


def make_toon_material(bpy, mat, toon_group) -> None:
    """Replace ``mat``'s shading with base colour -> ToonShade -> Emission, keeping the texture."""
    nt = mat.node_tree
    if nt is None:
        mat.use_nodes = True
        nt = mat.node_tree
    bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
    color_src = None
    base_value = (0.8, 0.8, 0.8, 1.0)
    if bsdf is not None:
        base = bsdf.inputs["Base Color"]
        base_value = tuple(base.default_value)
        if base.is_linked:
            color_src = base.links[0].from_socket
    output = next((n for n in nt.nodes if n.type == "OUTPUT_MATERIAL"), None)
    if output is None:
        output = nt.nodes.new("ShaderNodeOutputMaterial")
    grp = nt.nodes.new("ShaderNodeGroup")
    grp.node_tree = toon_group
    grp.location = (output.location.x - 400, output.location.y - 300)
    if color_src is not None:
        nt.links.new(color_src, grp.inputs["Color"])
    else:
        grp.inputs["Color"].default_value = base_value
    emit = nt.nodes.new("ShaderNodeEmission")
    emit.location = (output.location.x - 200, output.location.y - 300)
    emit.inputs["Strength"].default_value = 1.0
    nt.links.new(grp.outputs["Color"], emit.inputs["Color"])
    nt.links.new(emit.outputs["Emission"], output.inputs["Surface"])
    output.is_active_output = True
    no_light_sampling(mat)


def no_light_sampling(mat) -> None:
    """Emission here is just flat colour; don't make Cycles treat every face as a light."""
    cycles_settings = getattr(mat, "cycles", None)
    if cycles_settings is not None and hasattr(cycles_settings, "emission_sampling"):
        cycles_settings.emission_sampling = "NONE"


def build_outline_material(bpy):
    """Black where the hull faces away from its (flipped) normals, transparent elsewhere."""
    mat = bpy.data.materials.new("ToonOutline")
    mat.use_nodes = True
    mat.use_backface_culling = True  # EEVEE viewport nicety; Backfacing mask does the real work
    nt = mat.node_tree
    for node in list(nt.nodes):
        nt.nodes.remove(node)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (300, 0)
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    geo.location = (-300, 200)
    ink = nt.nodes.new("ShaderNodeEmission")
    ink.location = (-300, 0)
    ink.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    clear = nt.nodes.new("ShaderNodeBsdfTransparent")
    clear.location = (-300, -150)
    mix = nt.nodes.new("ShaderNodeMixShader")
    mix.location = (50, 0)
    # Backfacing = 1 (near side of the flipped hull) -> transparent; 0 -> ink.
    nt.links.new(geo.outputs["Backfacing"], mix.inputs["Fac"])
    nt.links.new(ink.outputs["Emission"], mix.inputs[1])
    nt.links.new(clear.outputs["BSDF"], mix.inputs[2])
    nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])
    if hasattr(mat, "surface_render_method"):
        mat.surface_render_method = "DITHERED"
    no_light_sampling(mat)
    return mat


def add_outline(obj, outline_mat, thickness: float) -> None:
    obj.data.materials.append(outline_mat)
    mod = obj.modifiers.new("ToonOutline", "SOLIDIFY")
    mod.thickness = thickness
    mod.offset = 1.0
    mod.use_flip_normals = True
    mod.use_rim = False
    # Even offset spikes at sharp/degenerate vertices of generated meshes; clamp instead.
    mod.use_even_offset = False
    mod.use_quality_normals = True
    mod.thickness_clamp = 1.0
    mod.material_offset = len(obj.data.materials) - 1


def add_front_projection(
    bpy,
    meshes,
    image_path,
    mask_path=None,
    facing=(0.2, 0.55),
    erode_px: int = 3,
):
    """Project the source image onto the front of ``meshes`` (current world transforms).

    TRELLIS.2 generates its own texture, which blurs small details such as eyes. The
    character faces the camera in the source image, so for surfaces that face the front
    and are not hidden behind other geometry, the original pixels are better. Adds a
    ``front_proj`` UV map and a ``front_weight`` point attribute to each mesh, and returns
    a packed RGBA image (alpha = eroded mask) for ``insert_projection``.
    """
    import numpy as np
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree

    src = bpy.data.images.load(str(image_path), check_existing=True)
    w, h = src.size
    rgba = np.empty(w * h * 4, np.float32)
    src.pixels.foreach_get(rgba)
    rgba = rgba.reshape(h, w, 4)
    if mask_path:
        mimg = bpy.data.images.load(str(mask_path), check_existing=True)
        if tuple(mimg.size) != (w, h):
            raise ValueError(f"mask size {tuple(mimg.size)} != image size {(w, h)}")
        mpx = np.empty(w * h * 4, np.float32)
        mimg.pixels.foreach_get(mpx)
        mask = mpx.reshape(h, w, 4)[..., 0] > 0.5
    else:
        mask = background_mask(rgba)
    inner = erode(mask, erode_px)

    # World-space geometry of all meshes, for fitting and occlusion rays.
    parts = []
    for o in meshes:
        me = o.data
        n = len(me.vertices)
        co = np.empty(n * 3, np.float32)
        me.vertices.foreach_get("co", co)
        no = np.empty(n * 3, np.float32)
        me.vertices.foreach_get("normal", no)
        m = np.array(o.matrix_world, dtype=np.float64)
        rot = m[:3, :3]
        wco = co.reshape(n, 3) @ rot.T + m[:3, 3]
        wno = no.reshape(n, 3) @ np.linalg.inv(rot)  # inverse-transpose for normals
        wno /= np.linalg.norm(wno, axis=1, keepdims=True) + 1e-12
        me.calc_loop_triangles()
        tris = np.empty(len(me.loop_triangles) * 3, np.int32)
        me.loop_triangles.foreach_get("vertices", tris)
        parts.append((o, wco, wno, tris.reshape(-1, 3)))

    all_co = np.concatenate([p[1] for p in parts])
    s, ox, oy, iou = fit_projection(all_co[:, [0, 2]], mask)
    print(f"[toon] front projection fit: IoU {iou:.3f}", flush=True)

    tri_offset, all_tris = 0, []
    for _, wco, _, tris in parts:
        all_tris.append(tris + tri_offset)
        tri_offset += len(wco)
    bvh = BVHTree.FromPolygons(all_co.tolist(), np.concatenate(all_tris).tolist())
    height = float(all_co[:, 2].max() - all_co[:, 2].min())
    eps = 1e-3 * height
    toward_camera = Vector((0.0, -1.0, 0.0))  # the camera looks along +Y from -Y

    for o, wco, wno, _ in parts:
        me = o.data
        px, py = s * wco[:, 0] + ox, s * wco[:, 2] + oy
        weight = smoothstep(facing[0], facing[1], -wno[:, 1])
        xi = np.clip(np.round(px).astype(int), 0, w - 1)
        yi = np.clip(np.round(py).astype(int), 0, h - 1)
        weight *= inner[yi, xi]
        for i in np.flatnonzero(weight > 0):
            origin = Vector((wco[i] + wno[i] * eps).tolist())
            if bvh.ray_cast(origin, toward_camera)[0] is not None:
                weight[i] = 0.0  # something is in front of this point
        attr = me.attributes.get("front_weight") or me.attributes.new(
            "front_weight", "FLOAT", "POINT"
        )
        attr.data.foreach_set("value", weight.astype(np.float32))

        render_uv = next((uv for uv in me.uv_layers if uv.active_render), None)
        active_index = me.uv_layers.active_index
        uvl = me.uv_layers.get("front_proj") or me.uv_layers.new(name="front_proj", do_init=False)
        loop_v = np.empty(len(me.loops), np.int32)
        me.loops.foreach_get("vertex_index", loop_v)
        uv = np.stack([(px + 0.5) / w, (py + 0.5) / h], axis=1)[loop_v].astype(np.float32)
        uvl.uv.foreach_set("vector", uv.ravel())
        me.uv_layers.active_index = active_index
        if render_uv is not None:
            render_uv.active_render = True
        print(
            f"[toon] {o.name}: {int((weight > 0).sum())} vertices use the source image", flush=True
        )

    proj = bpy.data.images.new("FrontProjection", w, h, alpha=True)
    out = rgba.copy()
    out[..., 3] = inner.astype(np.float32)
    proj.pixels.foreach_set(out.ravel())
    proj.alpha_mode = "STRAIGHT"
    proj.pack()
    return proj


def insert_projection(bpy, mat, proj_image) -> bool:
    """Blend ``proj_image`` (via ``front_proj`` UVs) into the colour feeding ToonShade."""
    nt = mat.node_tree
    grp = next(
        (
            n
            for n in nt.nodes
            if n.type == "GROUP" and n.node_tree and n.node_tree.name.startswith("ToonShade")
        ),
        None,
    )
    if grp is None:
        return False
    sock = grp.inputs["Color"]
    base = sock.links[0].from_socket if sock.is_linked else None
    x, y = grp.location.x, grp.location.y
    uvn = nt.nodes.new("ShaderNodeUVMap")
    uvn.uv_map = "front_proj"
    uvn.location = (x - 900, y - 250)
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = proj_image
    tex.extension = "CLIP"
    tex.location = (x - 700, y - 250)
    attr = nt.nodes.new("ShaderNodeAttribute")
    attr.attribute_type = "GEOMETRY"
    attr.attribute_name = "front_weight"
    attr.location = (x - 700, y - 550)
    fac = nt.nodes.new("ShaderNodeMath")
    fac.operation = "MULTIPLY"
    fac.location = (x - 420, y - 400)
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.blend_type = "MIX"
    mix.label = "Front projection"
    mix.location = (x - 220, y - 150)
    nt.links.new(uvn.outputs["UV"], tex.inputs["Vector"])
    nt.links.new(attr.outputs["Fac"], fac.inputs[0])
    nt.links.new(tex.outputs["Alpha"], fac.inputs[1])
    nt.links.new(fac.outputs["Value"], mix.inputs["Factor"])
    if base is not None:
        nt.links.new(base, mix.inputs["A"])
    else:
        mix.inputs["A"].default_value = tuple(sock.default_value)
    nt.links.new(tex.outputs["Color"], mix.inputs["B"])
    nt.links.new(mix.outputs["Result"], sock)
    return True


def setup_engine(bpy, scene, args) -> None:
    if args.engine == "EEVEE":
        for name in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
            try:
                scene.render.engine = name
                break
            except TypeError:
                continue
        scene.eevee.taa_render_samples = args.samples
        return
    scene.render.engine = "CYCLES"
    scene.render.use_persistent_data = True  # keep geometry/textures between turntable frames
    cyc = scene.cycles
    cyc.samples = args.samples
    cyc.use_denoising = False
    cyc.use_adaptive_sampling = False
    cyc.max_bounces = 2
    cyc.transparent_max_bounces = 16
    cyc.caustics_reflective = cyc.caustics_refractive = False
    cyc.device = "CPU"
    if args.cpu:
        return
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for backend in ("OPTIX", "CUDA", "HIP", "METAL", "ONEAPI"):
        try:
            prefs.compute_device_type = backend
        except TypeError:
            continue
        prefs.get_devices()
        gpus = [d for d in prefs.devices if d.type == backend]
        if gpus:
            for d in prefs.devices:
                d.use = d.type == backend
            cyc.device = "GPU"
            print(f"[toon] Cycles {backend}: {', '.join(d.name for d in gpus)}", flush=True)
            return
    print("[toon] no GPU backend found, rendering Cycles on the CPU", flush=True)


def key_turntable(pivot, base_yaw: float, frames: int) -> None:
    """One full turn over ``frames`` frames, keyed every 2 frames with constant interpolation."""
    for f in range(1, frames + 1, 2):
        pivot.rotation_euler.z = base_yaw + 2 * math.pi * (f - 1) / frames
        pivot.keyframe_insert("rotation_euler", index=2, frame=f)
    action = pivot.animation_data.action
    curves = action.fcurves if hasattr(action, "fcurves") else []
    for fc in curves:
        for kp in fc.keyframe_points:
            kp.interpolation = "CONSTANT"


def encode_mp4(frames_dir: Path, dest: Path, background: str) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print("[toon] ffmpeg not found; turntable frames kept in", frames_dir, flush=True)
        return
    # Rendered frames are the odd ones (1, 3, 5, ...): play them at 12 fps, output 24 fps.
    cmd = [
        ffmpeg, "-y", "-loglevel", "error", "-framerate", "12", "-pattern_type", "glob",
        "-i", str(frames_dir / "f_*.png"), "-r", "24",
        "-vf", f"pad=ceil(iw/2)*2:ceil(ih/2)*2:color={background}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(dest),
    ]  # fmt: skip
    subprocess.run(cmd, check=True)
    print(f"[toon] turntable -> {dest}", flush=True)


if __name__ == "__main__":
    main(sys.argv)
