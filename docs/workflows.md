# ComfyUI workflows

The workflows are built in Python by [`src/maker3d/workflows.py`](../src/maker3d/workflows.py), not stored as
JSON, so every parameter (prompt, seed, sizes, texture resolution) is a plain argument.

| Builder | Issue | Source template (ComfyUI v0.44.0) |
|---|---|---|
| `anima_t2i(prompt, seed, ...)` | #6 | `image_anima_base_v1.json` (turbo branch) |
| `trellis2_i2m(image_name, seed, ...)` | #8 | `3d_pixal3d_trellis2_image_to_model.json` (TRELLIS.2 branch only; Pixal3D/MoGe removed) |

Both pass ComfyUI v0.44.0's `/prompt` validation. Verified on Colab on 2026-11-07.

To export one as JSON for the ComfyUI UI:

```python
import json
from maker3d.workflows import trellis2_i2m

print(json.dumps(trellis2_i2m("input.png", seed=1), indent=2))
```
