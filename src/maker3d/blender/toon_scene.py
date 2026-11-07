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
    return p.parse_args(argv)


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
    return mat


def add_outline(obj, outline_mat, thickness: float) -> None:
    obj.data.materials.append(outline_mat)
    mod = obj.modifiers.new("ToonOutline", "SOLIDIFY")
    mod.thickness = thickness
    mod.offset = 1.0
    mod.use_flip_normals = True
    mod.use_rim = False
    mod.use_even_offset = True
    mod.material_offset = len(obj.data.materials) - 1


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
