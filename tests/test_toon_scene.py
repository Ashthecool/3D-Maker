"""Pure-Python parts of the Blender toon script (bpy is only imported inside main())."""

import pytest

from maker3d.blender import toon_scene as ts


def test_bands():
    assert ts.cel_bands("flat", 2) == [ts.Band(0.0, 1.0)]
    two = ts.cel_bands("cel", 2)
    assert [b.position for b in two] == [0.0, 0.5]
    assert two[0].light == 0.0 and two[-1].light == 1.0
    three = ts.cel_bands("cel_rim", 3)
    assert [b.position for b in three] == sorted(b.position for b in three)
    assert [b.light for b in three] == [0.0, 0.5, 1.0]
    with pytest.raises(ValueError):
        ts.cel_bands("anime", 2)
    with pytest.raises(ValueError):
        ts.cel_bands("cel", 4)


def _saturation(rgb):
    return 1 - min(rgb) / max(rgb)


def test_shadow_color_is_per_colour_and_richer():
    skin, silver, red = (1.0, 0.73, 0.6), (0.75, 0.69, 0.71), (0.34, 0.027, 0.033)
    flat = [tuple(c * t for c, t in zip(x, ts.SHADOW_TINT, strict=True)) for x in (skin, silver)]
    for base, old in zip((skin, silver), flat, strict=True):
        new = ts.shadow_color(base)
        assert max(new) < max(base)  # darker than the lit colour
        assert _saturation(new) > _saturation(old)  # not the grey of a flat multiplier
    assert ts.shadow_color(red)[0] > 5 * ts.shadow_color(red)[1]  # red stays red
    sil = ts.shadow_color(silver)
    assert sil[2] > sil[0] > sil[1]  # silver hair gets a cool lavender shadow
    sk = ts.shadow_color(skin)
    assert sk[0] > sk[1] >= sk[2]  # skin keeps its warm hue: not purple
    pale = ts.shadow_color((0.89, 0.75, 0.71))
    assert pale[0] > 1.15 * max(pale[1:]) and abs(pale[1] - pale[2]) < 0.05  # rosy, not mauve
    assert ts.shadow_color(skin, strength=0.0) == pytest.approx(skin)
    hue_shifted = ts.shadow_color(skin, hue_shift=-0.03)
    assert hue_shifted[1] < ts.shadow_color(skin)[1]  # orange skin shifts towards red
    assert all(0.0 <= c <= 1.0 for c in ts.shadow_color((1.0, 1.0, 1.0), saturation=3.0))


def test_turntable_keys():
    twos = ts.turntable_keys(96, hold=2)
    assert [f for f, _ in twos[:3]] == [1, 3, 5] and len(twos) == 48
    assert twos[24] == (49, pytest.approx(0.5))  # the back view halfway
    threes = ts.turntable_keys(96, hold=3, ease=0.3)
    assert len(threes) == 32 and threes[0] == (1, 0.0) and threes[16] == (49, pytest.approx(0.5))
    turns = [t for _, t in threes]
    steps = [b - a for a, b in zip(turns, turns[1:], strict=False)]
    assert all(s > 0 for s in steps)  # always turning the same way
    assert steps[0] < 1 / 32 < steps[8]  # slower past the front, faster past the side
    with pytest.raises(ValueError):
        ts.turntable_keys(96, hold=0)
    with pytest.raises(ValueError):
        ts.turntable_keys(96, ease=1.0)


def test_parse_args_new_options():
    args = ts.parse_args(["--glb", "m.glb", "--out", "o"])
    assert args.hold == 2 and args.ease == 0.3 and args.normal_smooth > 0
    args = ts.parse_args(["--glb", "m.glb", "--out", "o", "--hold", "3", "--normal-smooth", "0"])
    assert args.hold == 3 and args.normal_smooth == 0


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


def test_smoothing_iterations():
    assert ts.smoothing_iterations(0.0, 0.01) == 0 and ts.smoothing_iterations(0.1, 0.0) == 0
    assert ts.smoothing_iterations(0.02, 0.01) == 10  # (2 edges)^2 / 0.43, rounded up
    assert ts.smoothing_iterations(1.0, 0.001) == 300  # capped


def _grid(n=21, bump=0.0, seed=0):
    """An n x n vertex grid in the XY plane (normals +Z), split into two along x = middle
    like a glTF UV seam: the middle column of vertices exists twice with no edge between.
    """
    rng = np.random.default_rng(seed)
    ij = [(i, j) for j in range(n) for i in range(n)]
    co = np.array([(i / (n - 1), j / (n - 1), 0.0) for i, j in ij])
    normals = np.tile([0.0, 0.0, 1.0], (len(co), 1))
    normals[:, :2] += rng.normal(0, bump, (len(co), 2))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    mid = n // 2
    dup = [k for k, (i, _) in enumerate(ij) if i == mid]
    copy = {k: len(co) + d for d, k in enumerate(dup)}  # the right half's seam vertices
    co = np.concatenate([co, co[dup]])
    normals = np.concatenate([normals, normals[dup]])
    edges = []
    for k, (i, j) in enumerate(ij):
        if i + 1 < n:  # along x; leaving the seam to the right starts at the copy
            edges.append((copy[k] if i == mid else k, k + 1))
        if j + 1 < n:  # along y; the seam column has this edge on both sides
            edges.append((k, k + n))
            if i == mid:
                edges.append((copy[k], copy[k + n]))
    return co, normals, np.array(edges)


def test_weld_vertices():
    co, _, _ = _grid()
    weld = ts.weld_vertices(co, 1e-6)
    assert weld.max() + 1 == 21 * 21  # the 21 seam copies join their originals
    assert weld[21 * 21] == weld[10]  # first copy is vertex (10, 0)


def test_smooth_normals_flatten_noise_across_the_seam():
    co, normals, edges = _grid(bump=0.3)
    weld = ts.weld_vertices(co, 1e-6)
    out = ts.smooth_normals(normals, edges, 20, weld)
    assert np.linalg.norm(out, axis=1) == pytest.approx(1.0)
    assert out[:, 2].min() > 0.97 > normals[:, 2].min()  # the bumps are gone
    copies, originals = np.arange(21 * 21, len(co)), np.arange(21) * 21 + 10
    assert out[copies] == pytest.approx(out[originals])  # both sides of the seam agree
    unwelded = ts.smooth_normals(normals, edges, 20)
    assert np.abs(unwelded[copies] - unwelded[originals]).max() > 1e-3  # what welding fixes
    flat = ts.smooth_normals(normals, np.empty((0, 2), int), 5)
    assert flat == pytest.approx(normals)  # no edges: nothing to average with
    opposite = ts.smooth_normals(np.array([[0, 0, 1.0], [0, 0, -1.0]]), np.array([[0, 1]]), 1)
    assert opposite[:, 2] == pytest.approx([1.0, -1.0])  # cancels out: keep the original


def test_grid_regress_local_means():
    rng = np.random.default_rng(1)
    feat = rng.uniform(0, 1, (4000, 2))
    values = (feat[:, :1] > 0.5).astype(float) * 2.0  # 0 on the left, 2 on the right
    query = np.array([[0.1, 0.5], [0.9, 0.5], [0.5, 0.5]])
    mean, weight = ts.grid_regress(feat, values, query, (16, 16), (1.0, 1.0))
    assert mean[0, 0] == pytest.approx(0.0, abs=0.05) and mean[1, 0] == pytest.approx(2.0, abs=0.05)
    assert 0.5 < mean[2, 0] < 1.5  # on the border: a blend
    assert (weight > 10).all()


def test_kmeans():
    rng = np.random.default_rng(3)
    blobs = np.array([[0.1, 0.1, 0.1], [0.9, 0.9, 0.9], [0.7, 0.1, 0.1]])
    x = np.concatenate([b + rng.normal(0, 0.02, (300, 3)) for b in blobs])
    centres, labels = ts.kmeans(x, 3)
    assert sorted(map(tuple, centres.round(1))) == sorted(map(tuple, blobs))
    assert len(set(labels[:300])) == 1 and len(set(labels)) == 3
    one, _ = ts.kmeans(np.ones((50, 3)), 4)  # fewer distinct colours than k
    assert len(one) == 1


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
