"""Job orchestration across the three stages (issue #15).

stage 1: Anima Turbo v1.1   prompt -> candidate PNGs        (ComfyUI)
stage 2: BiRefNet+TRELLIS.2 chosen PNG -> textured GLB      (ComfyUI)
stage 3: Blender toon       GLB -> stills, turntable, .blend (blender -b)

Every stage writes into one job folder on Google Drive so a disconnected
Colab session can resume from the last finished stage.
"""
