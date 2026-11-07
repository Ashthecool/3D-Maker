"""Pure-Python parts of the Blender toon script (bpy is only imported inside main())."""

import pytest

from maker3d.blender import toon_scene as ts


def test_bands():
    assert ts.cel_bands("flat", 2) == [ts.Band(0.0, (1.0, 1.0, 1.0))]
    two = ts.cel_bands("cel", 2)
    assert [b.position for b in two] == [0.0, 0.5]
    assert two[0].value == ts.SHADOW_TINT and two[-1].value == (1.0, 1.0, 1.0)
    three = ts.cel_bands("cel_rim", 3)
    assert [b.position for b in three] == sorted(b.position for b in three)
    assert all(lo <= mid <= 1.0 for lo, mid in zip(three[0].value, three[1].value, strict=True))
    with pytest.raises(ValueError):
        ts.cel_bands("anime", 2)
    with pytest.raises(ValueError):
        ts.cel_bands("cel", 4)


def test_helpers():
    assert ts.normalized((3.0, 0.0, 4.0)) == (0.6, 0.0, 0.8)
    assert ts.outline_thickness(2.0, 0.005) == pytest.approx(0.01)
    assert ts.hex_to_rgb("#ffffff") == (1.0, 1.0, 1.0)
    assert ts.hex_to_rgb("000000") == (0.0, 0.0, 0.0)
    assert ts.hex_to_rgb("#808080")[0] == pytest.approx(0.2158, abs=1e-3)  # sRGB -> linear
    with pytest.raises(ValueError):
        ts.hex_to_rgb("#fff")


def test_parse_args_after_double_dash():
    argv = ["blender", "-b", "-P", "toon_scene.py", "--", "--glb", "m.glb", "--out", "job"]
    args = ts.parse_args(argv)
    assert str(args.glb) == "m.glb" and args.style == "cel" and args.engine == "CYCLES"
    assert not args.no_turntable and args.frames == 96
    args = ts.parse_args(["--glb", "m.glb", "--out", "o", "--style", "flat", "--no-outline"])
    assert args.style == "flat" and args.no_outline


# --- front projection fitting (numpy only) -------------------------------------------
np = pytest.importorskip("numpy")


def _inside(x, z):
    """A simple 'character': a body ellipse plus a head disc (x right, z up)."""
    body = (x / 0.25) ** 2 + ((z - 0.4) / 0.4) ** 2 <= 1
    head = (x - 0.05) ** 2 + (z - 0.9) ** 2 <= 0.12**2
    return body | head


def _figure(n=40000, seed=0):
    pts = np.random.default_rng(seed).uniform([-0.3, 0.0], [0.3, 1.05], (n, 2))
    return pts[_inside(pts[:, 0], pts[:, 1])]


def test_fit_projection_recovers_transform():
    pts = _figure()
    s, ox, oy = 400.0, 230.0, 60.0  # pixel = s * (x, z) + (ox, oy)
    py, px = np.mgrid[0:520, 0:460]
    mask = _inside((px - ox) / s, (py - oy) / s)  # row 0 = bottom, like Blender pixels
    fs, fox, foy, iou = ts.fit_projection(pts, mask)
    assert iou > 0.85
    assert fs == pytest.approx(s, rel=0.03)
    assert fox == pytest.approx(ox, abs=6) and foy == pytest.approx(oy, abs=6)


def test_masks():
    m = np.zeros((20, 20), bool)
    m[5:15, 5:15] = True
    assert ts.erode(m, 2).sum() == 36 and ts.dilate(m, 1).sum() == 100 + 40
    img = np.ones((20, 20, 4), np.float32)
    img[6:14, 6:14, :3] = 0.2  # dark square on white
    img[9:11, 9:11, :3] = 1.0  # white hole inside it is not background
    fg = ts.background_mask(img)
    assert fg[6:14, 6:14].all() and not fg[0].any()
    assert ts.smoothstep(0.2, 0.6, np.array([0.0, 0.4, 1.0])) == pytest.approx([0.0, 0.5, 1.0])
