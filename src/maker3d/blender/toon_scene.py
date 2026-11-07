"""Blender script that turns a TRELLIS.2 GLB into a 2D-looking toon scene (issues #11-#13).

Run inside Blender (not regular Python):
    blender -b --factory-startup -P toon_scene.py -- --glb model.glb --out job_dir --style cel

Design (engine-agnostic, so Cycles can render headless on Colab and EEVEE works locally):
- Cel bands: dot(normal, light_dir) -> constant ColorRamp, multiplied with the base-color
  texture. Avoids EEVEE-only Shader-to-RGB.
- Outline: inverted hull (Solidify, flipped normals) with a black material masked by
  Geometry > Backfacing instead of the backface-culling flag.
- Color management: Standard view transform (not AgX) so flat colours stay flat.
- Camera: long lens or orthographic; turntable animated on twos.
"""
