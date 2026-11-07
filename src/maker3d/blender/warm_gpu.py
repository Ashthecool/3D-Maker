"""Compile Cycles' GPU kernels ahead of time (issue #10).

Blender 4.5 ships no CUDA cubin for the A100 (sm_80), and OptiX kernels are PTX on every
GPU. So the NVIDIA driver JIT-compiles them on the first Cycles GPU render of a fresh
Colab VM, which took ~8 minutes on an A100. ``setup_colab.sh`` runs this script in
the background right after installing Blender, so the toon render later finds a warm cache
(``~/.nv/ComputeCache`` and ``/var/tmp/OptixCache_<user>``).

    blender -b --factory-startup -P warm_gpu.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bpy  # noqa: E402
import toon_scene as ts  # noqa: E402


def main() -> None:
    t0 = time.time()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    bpy.ops.mesh.primitive_cube_add()
    cube = bpy.context.object
    # Same shader features as the toon scene: emission, transparency, backfacing.
    cube.data.materials.append(ts.build_outline_material(bpy))
    cam = bpy.data.objects.new("Cam", bpy.data.cameras.new("Cam"))
    scene.collection.objects.link(cam)
    cam.location = (0.0, -6.0, 0.0)
    cam.rotation_euler = (1.5708, 0.0, 0.0)
    scene.camera = cam
    scene.render.resolution_x = scene.render.resolution_y = 32
    args = ts.parse_args(["--glb", "-", "--out", "-", "--samples", "1"])
    ts.setup_engine(bpy, scene, args)
    bpy.ops.render.render()
    print(
        f"[warm] Cycles {scene.cycles.device} kernels ready in {time.time() - t0:.0f}s", flush=True
    )


main()
