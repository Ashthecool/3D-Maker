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


def test_sampling_helpers():
    img = np.zeros((4, 4, 1))
    img[:, 2:] = 1.0
    assert ts.bilinear(img, np.array([1.5, 0.0, 9.0]), np.array([0.0, 0.0, 9.0]))[:, 0] == (
        pytest.approx([0.5, 0.0, 1.0])
    )
    line = np.full((9, 9, 3), 0.8)
    line[4, :] = 0.0  # a one-pixel ink line through the middle
    assert ts.median_sample(line, np.array([4.0]), np.array([4.0]))[0] == pytest.approx([0.8] * 3)
    assert ts.srgb_to_linear([0.0, 0.5, 1.0]) == pytest.approx([0.0, 0.21404, 1.0], abs=1e-4)
    g = ts.color_guide([[0.6, 0.2, 0.1]])[0]
    assert g == pytest.approx([0.3, 0.4, -0.3])


def test_smooth_vertex_values():
    # a path 0-1-2-3-4 with one noisy vertex in the middle
    values = np.array([[0.0], [0.0], [3.0], [0.0], [0.0]])
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4]])
    once = ts.smooth_vertex_values(values, edges, iterations=1)
    assert once[:, 0] == pytest.approx([0.0, 1.0, 1.0, 1.0, 0.0])
    assert ts.smooth_vertex_values(values, edges, 3).sum() == pytest.approx(values.sum(), rel=0.3)
    lone = ts.smooth_vertex_values(np.array([[5.0]]), np.empty((0, 2)), 2)
    assert lone[0, 0] == 5.0  # no neighbours: unchanged


def test_grid_regress_local_means():
    rng = np.random.default_rng(1)
    feat = rng.uniform(0, 1, (4000, 2))
    values = (feat[:, :1] > 0.5).astype(float) * 2.0  # 0 on the left, 2 on the right
    query = np.array([[0.1, 0.5], [0.9, 0.5], [0.5, 0.5]])
    mean, weight = ts.grid_regress(feat, values, query, (16, 16), (1.0, 1.0))
    assert mean[0, 0] == pytest.approx(0.0, abs=0.05) and mean[1, 0] == pytest.approx(2.0, abs=0.05)
    assert 0.5 < mean[2, 0] < 1.5  # on the border: a blend
    assert (weight > 10).all()


def _materials(x, z):
    """0 = hair (head, and down both sides), 1 = jacket (middle), 2 = legs."""
    hair = (z > 0.62) | ((np.abs(x) > 0.18) & (z > 0.3))
    return np.where(hair, 0, np.where(z > 0.3, 1, 2))


def test_match_colors_recolours_the_back():
    rng = np.random.default_rng(2)
    trellis = np.array([[0.17, 0.16, 0.16], [0.17, 0.05, 0.07], [0.10, 0.08, 0.10]])
    source = np.array([[0.88, 0.85, 0.86], [0.62, 0.18, 0.20], [0.10, 0.08, 0.12]])

    def points(n, material_fn):
        xz = rng.uniform([-0.3, 0.0], [0.3, 1.0], (n, 2))
        m = material_fn(xz[:, 0], xz[:, 1])
        # TRELLIS texture smudges, as left after smooth_vertex_values
        noisy = trellis[m] + rng.normal(0, 0.02, (n, 3))
        return xz, ts.color_guide(noisy), m

    src_xz, src_guide, src_m = points(20000, _materials)
    # The back: long hair covers everything above the legs, also where the front is jacket.
    back_xz, back_guide, back_m = points(5000, lambda x, z: np.where(z > 0.3, 0, 2))
    pred = ts.match_colors(src_xz, src_guide, source[src_m], back_xz, back_guide)
    for m in (0, 2):
        sel = back_m == m
        assert np.abs(pred[sel] - source[m]).mean() < 0.08, m
    hair_over_jacket = (back_m == 0) & (np.abs(back_xz[:, 0]) < 0.1) & (back_xz[:, 1] < 0.55)
    assert pred[hair_over_jacket][:, 0].mean() > 0.75  # silver, not jacket red
    assert pred[hair_over_jacket][:, 1].mean() > 0.6
    assert pred[back_m == 0].std(axis=0).max() < 0.12  # the smudges are not amplified


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
