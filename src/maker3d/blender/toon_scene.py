"""Blender script that turns a TRELLIS.2 GLB into a 2D-looking toon scene (issues #11-#13).

Run inside Blender (not regular Python):
    blender -b --factory-startup -P toon_scene.py -- --glb model.glb --out job_dir --style cel

Writes to ``--out``: ``stills/{front,three_quarter,side,back}.png``, ``turntable.mp4``
(if ffmpeg is on PATH; the PNG frames are kept in ``frames/``) and a packed ``toon.blend``.

Design (engine-agnostic, so Cycles can render headless on Colab and EEVEE works locally):
- Cel bands: dot(normal, light_dir) -> constant ColorRamp, which picks between the base
  colour and a shadow colour derived from it per pixel; output as Emission. Avoids
  EEVEE-only Shader-to-RGB. The normal can be a smoothed copy for cleaner shadow shapes.
- Outline: inverted hull (Solidify, flipped normals) with a black material masked by
  Geometry > Backfacing instead of the backface-culling flag.
- Color management: Standard view transform (not AgX) so flat colours stay flat.
- Camera: orthographic; turntable held on twos or threes (constant interpolation).

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
SHADOW_SATURATION = 1.35  # shadows are richer than the lit colour, not greyer (#21)
# Base colours at least this saturated keep their own hue in shadow (skin goes rosy, red
# goes crimson); greyer ones (silver, white, black) take the cool SHADOW_TINT instead.
OWN_HUE_SATURATION = 0.25
STILLS = {"front": 0.0, "three_quarter": 45.0, "side": 90.0, "back": 180.0}


# --------------------------------------------------------------------------------------
# Pure helpers (no bpy)
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Band:
    position: float  # 0..1 on the half-Lambert factor
    light: float  # 0 = the shadow colour, 1 = the base colour


def cel_bands(style: str, bands: int) -> list[Band]:
    """Constant ColorRamp stops for ``style``. ``flat`` has a single full-bright band."""
    if style not in STYLES:
        raise ValueError(f"unknown style {style!r}; choose from {STYLES}")
    if style == "flat":
        return [Band(0.0, 1.0)]
    if bands not in (2, 3):
        raise ValueError("bands must be 2 or 3")
    if bands == 2:
        return [Band(0.0, 0.0), Band(0.5, 1.0)]
    return [Band(0.0, 0.0), Band(0.42, 0.5), Band(0.62, 1.0)]


def tint_luminance(tint) -> float:
    """Rec.709 luminance, as Blender's RGB to BW node computes it in linear Rec.709."""
    return 0.2126 * tint[0] + 0.7152 * tint[1] + 0.0722 * tint[2]


def shadow_color(
    rgb,
    tint=SHADOW_TINT,
    saturation: float = SHADOW_SATURATION,
    hue_shift: float = 0.0,
    strength: float = 1.0,
    own_hue: float = OWN_HUE_SATURATION,
):
    """Reference for the ToonShade group's shadow colour (scene-linear RGB in and out).

    The tint fades from ``tint`` (for grey bases) to a neutral grey of the same
    luminance (for bases with saturation >= ``own_hue``), so colourful parts get darker
    in their own hue. Then saturation x ``saturation`` and hue + ``hue_shift`` in HSV,
    mixed with the base by ``strength``. Mirrors the node graph in build_toon_group().
    """
    import colorsys

    base = [float(c) for c in rgb]
    w = min(max(colorsys.rgb_to_hsv(*base)[1] / own_hue, 0.0), 1.0)
    grey = tint_luminance(tint)
    tint = [t + (grey - t) * w for t in tint]
    tinted = [c * t for c, t in zip(base, tint, strict=True)]
    h, s, v = colorsys.rgb_to_hsv(*tinted)
    shaded = colorsys.hsv_to_rgb((h + hue_shift) % 1.0, min(max(s * saturation, 0.0), 1.0), v)
    return tuple(b + (c - b) * strength for b, c in zip(base, shaded, strict=True))


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
    p.add_argument(
        "--hold", type=int, choices=(1, 2, 3, 4), default=2, help="frames per pose: 2 = on twos"
    )
    p.add_argument(
        "--ease", type=float, default=0.3, help="linger on the front/back views (0 = even spin)"
    )
    p.add_argument("--no-turntable", action="store_true")
    p.add_argument(
        "--normal-smooth",
        type=float,
        default=0.02,
        help="blur shading normals over this fraction of the model height (0 = off)",
    )
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
    p.add_argument(
        "--no-color-match",
        action="store_true",
        help="keep TRELLIS.2's own (darker) texture colours away from the front",
    )
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


# --------------------------------------------------------------------------------------
# Colour match (issue #24), numpy only. TRELLIS.2's own texture is much darker and flatter
# than the Anima image (silver hair comes out dark grey, with noisy smudges), so the sides
# and back did not match the projected front. Every vertex gets the source-image colour of
# front-visible points that are nearby in x/z *and* had a similar TRELLIS colour.
# --------------------------------------------------------------------------------------
def srgb_to_linear(c):
    import numpy as np

    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def bilinear(img, x, y):
    """Sample an (H, W, C) image at float pixel coordinates (pixel centres at integers)."""
    import numpy as np

    h, w = img.shape[:2]
    x = np.clip(np.asarray(x, np.float64), 0, w - 1)
    y = np.clip(np.asarray(y, np.float64), 0, h - 1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    top = img[y0, x0] * (1 - fx) + img[y0, x1] * fx
    bottom = img[y1, x0] * (1 - fx) + img[y1, x1] * fx
    return top * (1 - fy) + bottom * fy


def median_sample(img, x, y, radius: int = 2):
    """Per-channel median of the (2r+1)^2 pixels around each point; drops thin ink lines."""
    import numpy as np

    h, w = img.shape[:2]
    xi, yi = np.round(x).astype(int), np.round(y).astype(int)
    offs = np.arange(-radius, radius + 1)
    xs = np.clip(xi[:, None, None] + offs[None, None, :], 0, w - 1)
    ys = np.clip(yi[:, None, None] + offs[None, :, None], 0, h - 1)
    patch = img[ys, xs].reshape(len(xi), -1, img.shape[2])
    return np.median(patch, axis=1)


def smooth_vertex_values(values, edges, iterations: int = 3):
    """Average each vertex with its edge neighbours ``iterations`` times (denoises colours)."""
    import numpy as np

    values = np.asarray(values, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    n, channels = values.shape
    src = np.concatenate([edges[:, 0], edges[:, 1]])
    dst = np.concatenate([edges[:, 1], edges[:, 0]])
    degree = np.bincount(src, minlength=n).astype(np.float64)[:, None] + 1.0
    for _ in range(iterations):
        sums = [np.bincount(src, weights=values[dst, c], minlength=n) for c in range(channels)]
        values = (values + np.stack(sums, axis=1)) / degree
    return values


def color_guide(rgb):
    """Matching features for a colour: brightness and two opponent-colour axes."""
    import numpy as np

    rgb = np.asarray(rgb, dtype=np.float64)
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    return np.stack([rgb.mean(axis=1), r - g, b - 0.5 * (r + g)], axis=1)


def _blur_axis(a, axis: int, sigma: float):
    """Gaussian blur along one axis (zero outside), unnormalised: blur sums and counts alike."""
    import numpy as np

    n = a.shape[axis]
    if sigma <= 0 or n == 1:
        return a
    r = math.ceil(2.5 * sigma)
    a = np.moveaxis(a, axis, 0)
    out = np.zeros_like(a)
    for o in range(-min(r, n - 1), min(r, n - 1) + 1):
        k = math.exp(-0.5 * (o / sigma) ** 2)
        if o >= 0:
            out[: n - o] += k * a[o:]
        else:
            out[-o:] += k * a[: n + o]
    return np.moveaxis(out, 0, axis)


def grid_regress(feat, values, query, bins, sigma):
    """Bilateral-grid regression: the local mean of ``values`` around each ``query`` point.

    ``feat`` (N, D) and ``query`` (M, D) are scaled to [0, 1]; ``bins`` and ``sigma`` (in
    cells) are per dimension. Returns ``(mean (M, C), weight (M,))``, where weight is the
    blurred number of samples near the query (low = little evidence).
    """
    import itertools

    import numpy as np

    bins = np.asarray(bins, dtype=int)
    size = int(np.prod(bins))
    values = np.asarray(values, dtype=np.float64)
    cells = np.clip((np.asarray(feat) * bins).astype(int), 0, bins - 1)
    flat = np.ravel_multi_index(cells.T, bins)
    cols = [np.bincount(flat, weights=values[:, c], minlength=size) for c in range(values.shape[1])]
    cols.append(np.bincount(flat, minlength=size).astype(np.float64))
    grid = np.stack(cols, axis=-1).reshape(*bins, -1)
    for axis, s in enumerate(sigma):
        grid = _blur_axis(grid, axis, s)
    grid = grid.reshape(size, -1)
    # Multilinear lookup between cell centres, so results have no blocky steps.
    pos = np.asarray(query, dtype=np.float64) * bins - 0.5
    i0 = np.floor(pos).astype(int)
    f = pos - i0
    acc = np.zeros((len(pos), grid.shape[1]))
    for corner in itertools.product((0, 1), repeat=len(bins)):
        c = np.array(corner)
        w = np.prod(np.where(c == 1, f, 1 - f), axis=1)
        acc += w[:, None] * grid[np.ravel_multi_index(np.clip(i0 + c, 0, bins - 1).T, bins)]
    weight = acc[:, -1]
    return acc[:, :-1] / np.maximum(weight, 1e-9)[:, None], weight


def kmeans(x, k: int, iters: int = 25, seed: int = 0, fit_max: int = 20000):
    """Plain k-means (k-means++ start). Returns ``(centres (k', D), labels (N,))``."""
    import numpy as np

    x = np.asarray(x, dtype=np.float64)
    rng = np.random.default_rng(seed)
    fit = x[rng.choice(len(x), fit_max, replace=False)] if len(x) > fit_max else x
    centres = [fit[rng.integers(len(fit))]]
    for _ in range(1, k):
        d2 = ((fit[:, None, :] - np.array(centres)[None]) ** 2).sum(-1).min(axis=1)
        if d2.sum() <= 1e-12:
            break  # fewer distinct colours than k
        centres.append(fit[rng.choice(len(fit), p=d2 / d2.sum())])
    c = np.array(centres)
    for _ in range(iters):
        lab = ((fit[:, None, :] - c[None]) ** 2).sum(-1).argmin(axis=1)
        new = np.array([fit[lab == j].mean(0) if (lab == j).any() else c[j] for j in range(len(c))])
        if np.allclose(new, c):
            break
        c = new
    return c, ((x[:, None, :] - c[None]) ** 2).sum(-1).argmin(axis=1)


def match_colors(
    src_xz,
    src_guide,
    src_rgb,
    xz,
    guide,
    cell: float = 1 / 40,
    color_bins=(12, 8, 8),
    min_weight: float = 2.0,
    n_colors: int = 12,
    sharpen: float = 3.0,
):
    """Source-image colour for every point, from its x/z position and TRELLIS colour.

    ``src_*``: front-visible points with a known source colour. ``guide`` values come from
    ``color_guide``. The source colours are reduced to a palette of ``n_colors``; for every
    point the grid estimates how likely each palette colour is, going coarse to fine:
    colour only (whole model), then position + colour with a wide and a narrow spatial
    blur, each finer level taking over where it has enough samples (``min_weight``).
    Probabilities are raised to ``sharpen`` before mixing, so ambiguous spots pick the
    likelier flat colour instead of a muddy average. ``cell`` is the spatial cell size as
    a fraction of the model height.
    """
    import numpy as np

    src_xz, xz = np.asarray(src_xz, np.float64), np.asarray(xz, np.float64)
    palette, labels = kmeans(src_rgb, n_colors)
    src_rgb = np.eye(len(palette))[labels]  # one-hot: regress probabilities, not colours
    both = np.concatenate([src_xz, xz])
    lo, size = both.min(axis=0), np.ptp(both, axis=0)
    unit = max(float(size[1]), 1e-9) * cell  # cell edge in world units
    sbins = tuple(int(math.ceil(s / unit)) + 1 for s in size)

    def spatial(p):
        return (p - lo) / (np.array(sbins) * unit)

    g_all = np.concatenate([src_guide, guide])
    g_lo, g_hi = np.percentile(g_all, 0.5, axis=0), np.percentile(g_all, 99.5, axis=0)

    def colour(g):
        return np.clip((g - g_lo) / np.maximum(g_hi - g_lo, 1e-6), 0.0, 1.0 - 1e-9)

    gs, gq = colour(np.asarray(src_guide)), colour(np.asarray(guide))
    pred, w = grid_regress(gs, src_rgb, gq, color_bins, (1.0, 1.0, 1.0))
    a = (w / (w + min_weight))[:, None]
    pred = a * pred + (1 - a) * src_rgb.mean(axis=0)
    fs, fq = np.c_[spatial(src_xz), gs], np.c_[spatial(xz), gq]
    for s in (3.0, 1.0):
        p, w = grid_regress(fs, src_rgb, fq, sbins + tuple(color_bins), (s, s, 1.0, 1.0, 1.0))
        a = (w / (w + min_weight))[:, None]
        pred = a * p + (1 - a) * pred
    weights = np.clip(pred, 0.0, None) ** sharpen
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    return np.clip(weights @ palette, 0.0, 1.0)


def hex_to_rgb(value: str) -> tuple[float, float, float]:
    value = value.lstrip("#")
    if len(value) != 6:
        raise ValueError(f"expected #rrggbb, got {value!r}")
    srgb = [int(value[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    # sRGB -> linear, since Blender colour inputs are scene-linear
    return tuple(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in srgb)


# --------------------------------------------------------------------------------------
# Smoothed shading normals (issue #23), numpy only. Guilty Gear Xrd hand-edits normals
# so cel shadows fall in a few large, clean shapes. TRELLIS.2 meshes have small bumps
# everywhere, so their raw normals give blotchy bands; blurring the normals along the
# surface is the automatic stand-in. The blurred normals go into a ``toon_normal``
# attribute and the shader mixes them in, so the original normals stay available.
# --------------------------------------------------------------------------------------
def weld_vertices(co, tol: float):
    """Index of a shared vertex for each vertex: coincident vertices get the same one.

    The glTF importer splits a mesh along its UV seams, leaving two vertices with no edge
    between them at every seam point. Smoothing has to see them as one, or seams show.
    """
    import numpy as np

    q = np.round(np.asarray(co, dtype=np.float64) / tol).astype(np.int64)
    _, inverse = np.unique(q, axis=0, return_inverse=True)
    return inverse.reshape(-1)


def smoothing_iterations(radius: float, edge_length: float, max_iterations: int = 800) -> int:
    """Neighbour-averaging passes that blur over about ``radius`` (a Gaussian sigma).

    A pass moves weight about one edge (variance ~0.43 edge^2 per axis on a triangle
    mesh), so k passes spread ``edge * sqrt(0.43 k)``.
    """
    if radius <= 0 or edge_length <= 0:
        return 0
    return int(min(max_iterations, math.ceil((radius / edge_length) ** 2 / 0.43)))


def _coarse_smooth(values, edges, co, iterations: int, coarsen: float):
    """``smooth_vertex_values`` with the same reach, mostly done on a coarser graph.

    Vertices are grouped in voxel cells ``coarsen`` mean edges wide, split by the octant
    of their normal so the two sides of a thin sheet (a hair strand, a sleeve) stay
    apart. The cluster graph needs ``coarsen``^2 times fewer passes; a few passes on the
    full mesh then smooth away the cluster blocks. 10-20x faster on TRELLIS.2 meshes.
    """
    import numpy as np

    lengths = np.linalg.norm(co[edges[:, 0]] - co[edges[:, 1]], axis=1)
    cell = coarsen * float(lengths.mean())
    octant = (values > 0) @ np.array([1, 2, 4])
    key = np.c_[np.floor(co / cell).astype(np.int64), octant]
    _, cid = np.unique(key, axis=0, return_inverse=True)
    cid = cid.reshape(-1)
    k = int(cid.max()) + 1
    count = np.bincount(cid, minlength=k).astype(np.float64)[:, None]
    coarse = np.stack([np.bincount(cid, weights=values[:, c], minlength=k) for c in range(3)], 1)
    ce = np.sort(cid[edges], axis=1)
    ce = np.unique(ce[ce[:, 0] != ce[:, 1]], axis=0)
    coarse = smooth_vertex_values(coarse / count, ce, math.ceil(iterations / coarsen**2))
    fine = min(iterations, math.ceil(coarsen**2 / 0.43))  # blur about one cell
    return smooth_vertex_values(coarse[cid], edges, fine)


def smooth_normals(normals, edges, iterations: int, weld=None, co=None, coarsen: float = 4.0):
    """Blur vertex normals along the mesh ``iterations`` times; returns unit vectors.

    ``weld`` (from ``weld_vertices``) joins split seam vertices first. With ``co``
    (vertex positions), long blurs run on a coarser cluster graph (``_coarse_smooth``).
    Where the blur cancels out (the tip of a thin sheet), the original normal is kept.
    """
    import numpy as np

    normals = np.asarray(normals, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    values, e, points = normals, edges, co
    if weld is not None:
        weld = np.asarray(weld, dtype=np.int64)
        m = int(weld.max()) + 1 if len(weld) else 0
        values = np.stack(
            [np.bincount(weld, weights=normals[:, c], minlength=m) for c in range(3)], axis=1
        )
        e = np.sort(weld[edges], axis=1)
        e = np.unique(e[e[:, 0] != e[:, 1]], axis=0)  # seam edges exist on both sides
        if co is not None:
            points = np.empty((m, 3))
            points[weld] = np.asarray(co, dtype=np.float64)
    if points is not None and len(e) and iterations > 2 * coarsen**2 / 0.43:
        out = _coarse_smooth(values, e, np.asarray(points, np.float64), iterations, coarsen)
    else:
        out = smooth_vertex_values(values, e, iterations)
    if weld is not None:
        out = out[weld]
    length = np.linalg.norm(out, axis=1, keepdims=True)
    return np.where(length > 1e-6, out / np.maximum(length, 1e-12), normals)


# --------------------------------------------------------------------------------------
# Limited-animation turntable (issue #22)
# --------------------------------------------------------------------------------------
def turntable_keys(frames: int, hold: int = 2, ease: float = 0.0) -> list[tuple[int, float]]:
    """(frame, fraction of a full turn) keys: a new pose every ``hold`` frames.

    ``ease`` (0 to <1) slows the spin around the front and back views and speeds it up
    through the sides, the way hand-timed key poses linger on the good angles. The turn
    stays monotonic, starts at 0 and passes exactly 0.5 halfway.
    """
    if hold < 1:
        raise ValueError("hold must be at least 1 frame")
    if not 0.0 <= ease < 1.0:
        raise ValueError("ease must be in [0, 1)")
    keys = []
    for f in range(1, frames + 1, hold):
        t = (f - 1) / frames
        keys.append((f, t - ease * math.sin(4 * math.pi * t) / (4 * math.pi)))
    return keys


# --------------------------------------------------------------------------------------
# Blender side
# --------------------------------------------------------------------------------------
def main(argv: list[str]) -> None:
    import bpy
    from mathutils import Matrix, Vector

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
    smoothed = add_smooth_normals(meshes, args.normal_smooth)
    proj_image, color_matched = None, False
    if args.source_image:
        # Project in the pose the camera sees at yaw. glTF imports use quaternion rotation,
        # so turn the world matrix: changing rotation_euler would do nothing.
        turn = Matrix.Rotation(math.radians(args.yaw), 4, "Z")
        roots = [o for o in scene.objects if o.parent is None]
        for o in roots:
            o.matrix_world = turn @ o.matrix_world
        bpy.context.view_layer.update()
        proj_image, color_matched = add_front_projection(
            bpy,
            meshes,
            args.source_image,
            args.source_mask,
            color_match=not args.no_color_match,
        )
        for o in roots:
            o.matrix_world = turn.inverted() @ o.matrix_world
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
    toon_group = build_toon_group(bpy, args.style, args.bands, smooth_normals=smoothed)
    for mat in {s.material for o in meshes for s in o.material_slots if s.material}:
        make_toon_material(bpy, mat, toon_group)
        if proj_image is not None:
            insert_projection(bpy, mat, proj_image, vertex_color=color_matched)
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

    # --- turntable, a new pose every --hold frames ----------------------------------
    scene.frame_start, scene.frame_end = 1, args.frames
    key_turntable(pivot, base_yaw, turntable_keys(args.frames, args.hold, args.ease))
    if not args.no_turntable:
        frames_dir = out / "frames"
        frames_dir.mkdir(exist_ok=True)
        for old in frames_dir.glob("f_*.png"):  # a re-render with another --hold
            old.unlink()
        scene.frame_step = args.hold  # only the frames that change
        r.filepath = str(frames_dir / "f_")
        bpy.ops.render.render(animation=True)
        scene.frame_step = 1
        encode_mp4(frames_dir, out / "turntable.mp4", args.background, args.hold)

    # --- save -----------------------------------------------------------------------
    scene.frame_set(1)
    r.filepath = "//render/"
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(out / "toon.blend"), compress=True)
    print(f"[toon] saved {out / 'toon.blend'}", flush=True)


def _group_input(g, name: str, socket_type: str, default, lo=None, hi=None, tip: str = ""):
    sock = g.interface.new_socket(name, in_out="INPUT", socket_type=socket_type)
    sock.default_value = default
    if lo is not None:
        sock.min_value, sock.max_value = lo, hi
    sock.description = tip
    return sock


def build_toon_group(bpy, style: str, bands: int, smooth_normals: bool = False):
    """Node group: Color in -> toon-shaded Color out. Light direction lives here once.

    The shadow colour is derived from each pixel's base colour (issue #21): darker in its
    own hue, more saturated, and with the cool tint only where the base is greyish
    (``shadow_color`` is the same maths). The cel ramp picks between it and the base.
    With ``smooth_normals``, the ``toon_normal`` attribute replaces the mesh normal for
    the bands (issue #23); the "Smooth Normals" input blends back to the original.
    """
    g = bpy.data.node_groups.new("ToonShade", "ShaderNodeTree")
    g.interface.new_socket("Color", in_out="INPUT", socket_type="NodeSocketColor")
    _group_input(
        g, "Shadow Tint", "NodeSocketColor", (*SHADOW_TINT, 1.0), tip="Multiplies the shadow"
    )
    _group_input(
        g,
        "Shadow Saturation",
        "NodeSocketFloat",
        SHADOW_SATURATION,
        0.0,
        3.0,
        "Saturation of the shadow colour relative to the lit colour",
    )
    _group_input(g, "Shadow Hue Shift", "NodeSocketFloat", 0.0, -0.5, 0.5, "Rotate the shadow hue")
    _group_input(g, "Shadow Strength", "NodeSocketFloat", 1.0, 0.0, 1.0, "0 = no shadow colour")
    _group_input(
        g,
        "Smooth Normals",
        "NodeSocketFloat",
        1.0 if smooth_normals else 0.0,
        0.0,
        1.0,
        "Blend from the mesh normals (0) to the smoothed toon_normal attribute (1)",
    )
    g.interface.new_socket("Color", in_out="OUTPUT", socket_type="NodeSocketColor")
    n, links = g.nodes, g.links
    gin, gout = n.new("NodeGroupInput"), n.new("NodeGroupOutput")
    gin.location, gout.location = (-1300, 0), (700, 0)

    geo = n.new("ShaderNodeNewGeometry")
    geo.location = (-1300, -350)
    normal = geo.outputs["Normal"]
    if smooth_normals:
        attr = n.new("ShaderNodeAttribute")
        attr.attribute_type = "GEOMETRY"
        attr.attribute_name = "toon_normal"
        attr.label = "Smoothed normal (object space)"
        attr.location = (-1300, -600)
        to_world = n.new("ShaderNodeVectorTransform")
        to_world.vector_type = "NORMAL"
        to_world.convert_from = "OBJECT"
        to_world.convert_to = "WORLD"
        to_world.location = (-1100, -600)
        links.new(attr.outputs["Vector"], to_world.inputs["Vector"])
        # normal + Smooth Normals * (smoothed - normal), renormalised
        steps = []
        for op, x in (("SUBTRACT", -900), ("SCALE", -750), ("ADD", -600), ("NORMALIZE", -450)):
            node = n.new("ShaderNodeVectorMath")
            node.operation = op
            node.location = (x, -600)
            steps.append(node)
        sub, scale, add, unit = steps
        links.new(to_world.outputs["Vector"], sub.inputs[0])
        links.new(normal, sub.inputs[1])
        links.new(sub.outputs["Vector"], scale.inputs[0])
        links.new(gin.outputs["Smooth Normals"], scale.inputs["Scale"])
        links.new(normal, add.inputs[0])
        links.new(scale.outputs["Vector"], add.inputs[1])
        links.new(add.outputs["Vector"], unit.inputs[0])
        normal = unit.outputs["Vector"]
    light = n.new("ShaderNodeCombineXYZ")
    light.label = "Light direction (world)"
    light.location = (-900, -350)
    for axis, value in zip("XYZ", normalized(LIGHT_DIR), strict=True):
        light.inputs[axis].default_value = value
    dot = n.new("ShaderNodeVectorMath")
    dot.operation = "DOT_PRODUCT"
    dot.location = (-650, -300)
    links.new(normal, dot.inputs[0])
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
    elems[0].position, elems[0].color = stops[0].position, (stops[0].light,) * 3 + (1.0,)
    for band in stops[1:]:
        e = elems.new(band.position)
        e.color = (band.light,) * 3 + (1.0,)
    links.new(half.outputs["Result"], ramp.inputs["Fac"])

    # Shadow colour from the base colour (see shadow_color() for the same maths in Python)
    sep = n.new("ShaderNodeSeparateColor")
    sep.mode = "HSV"
    sep.location = (-1100, 450)
    links.new(gin.outputs["Color"], sep.inputs["Color"])
    own = n.new("ShaderNodeMapRange")  # base saturation -> 0 (cool tint) .. 1 (own hue)
    own.label = "Keep own hue from saturation"
    own.location = (-900, 450)
    own.inputs["From Max"].default_value = OWN_HUE_SATURATION
    links.new(sep.outputs[1], own.inputs["Value"])
    grey = n.new("ShaderNodeRGBToBW")
    grey.location = (-1100, 250)
    links.new(gin.outputs["Shadow Tint"], grey.inputs["Color"])
    tint = n.new("ShaderNodeMix")
    tint.data_type = "RGBA"
    tint.location = (-850, 250)
    links.new(own.outputs["Result"], tint.inputs["Factor"])
    links.new(gin.outputs["Shadow Tint"], tint.inputs["A"])
    links.new(grey.outputs["Val"], tint.inputs["B"])
    tinted = n.new("ShaderNodeMix")
    tinted.data_type = "RGBA"
    tinted.blend_type = "MULTIPLY"
    tinted.location = (-650, 250)
    tinted.inputs["Factor"].default_value = 1.0
    links.new(gin.outputs["Color"], tinted.inputs["A"])
    links.new(tint.outputs["Result"], tinted.inputs["B"])
    hue = n.new("ShaderNodeMath")
    hue.operation = "ADD"
    hue.location = (-650, 450)
    hue.inputs[1].default_value = 0.5  # 0.5 = no hue change on the HSV node
    links.new(gin.outputs["Shadow Hue Shift"], hue.inputs[0])
    hsv = n.new("ShaderNodeHueSaturation")
    hsv.label = "Shadow colour"
    hsv.location = (-450, 300)
    links.new(hue.outputs["Value"], hsv.inputs["Hue"])
    links.new(gin.outputs["Shadow Saturation"], hsv.inputs["Saturation"])
    links.new(tinted.outputs["Result"], hsv.inputs["Color"])
    strength = n.new("ShaderNodeMix")
    strength.data_type = "RGBA"
    strength.location = (-250, 250)
    links.new(gin.outputs["Shadow Strength"], strength.inputs["Factor"])
    links.new(gin.outputs["Color"], strength.inputs["A"])
    links.new(hsv.outputs["Color"], strength.inputs["B"])

    shade = n.new("ShaderNodeMix")  # band light 0 -> shadow colour, 1 -> base colour
    shade.data_type = "RGBA"
    shade.label = "Cel shade"
    shade.location = (50, 0)
    links.new(ramp.outputs["Color"], shade.inputs["Factor"])
    links.new(strength.outputs["Result"], shade.inputs["A"])
    links.new(gin.outputs["Color"], shade.inputs["B"])
    result = shade.outputs["Result"]

    if style == "cel_rim":
        rim = n.new("ShaderNodeLayerWeight")
        rim.location = (-450, 650)
        rim.inputs["Blend"].default_value = 0.35
        rim_ramp = n.new("ShaderNodeValToRGB")
        rim_ramp.label = "Rim"
        rim_ramp.location = (-250, 650)
        rim_ramp.color_ramp.interpolation = "CONSTANT"
        rim_ramp.color_ramp.elements[0].color = (0, 0, 0, 1)
        rim_ramp.color_ramp.elements[1].position = 0.6
        rim_ramp.color_ramp.elements[1].color = (0.35, 0.35, 0.35, 1)
        links.new(rim.outputs["Facing"], rim_ramp.inputs["Fac"])
        lit_only = n.new("ShaderNodeMix")  # rim only on the lit side
        lit_only.data_type = "RGBA"
        lit_only.blend_type = "MULTIPLY"
        lit_only.location = (50, 500)
        lit_only.inputs["Factor"].default_value = 1.0
        links.new(rim_ramp.outputs["Color"], lit_only.inputs["A"])
        links.new(ramp.outputs["Color"], lit_only.inputs["B"])
        add = n.new("ShaderNodeMix")
        add.data_type = "RGBA"
        add.blend_type = "ADD"
        add.location = (300, 0)
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


def add_smooth_normals(meshes, strength: float) -> bool:
    """Write blurred vertex normals to a ``toon_normal`` attribute (object space).

    ``strength``: blur radius as a fraction of the model height; 0 turns it off. Returns
    whether the attribute was written (the ToonShade group only reads it then).
    """
    import time

    import numpy as np
    from mathutils import Vector

    if strength <= 0:
        return False
    t0 = time.time()
    corners = [o.matrix_world @ Vector(c) for o in meshes for c in o.bound_box]
    height = max(c.z for c in corners) - min(c.z for c in corners)
    for o in meshes:
        me = o.data
        n = len(me.vertices)
        co = np.empty(n * 3, np.float32)
        me.vertices.foreach_get("co", co)
        co = co.reshape(n, 3)
        no = np.empty(n * 3, np.float32)
        me.vertices.foreach_get("normal", no)
        edges = np.empty(len(me.edges) * 2, np.int64)
        me.edges.foreach_get("vertices", edges)
        edges = edges.reshape(-1, 2)
        scale = abs(np.linalg.det(np.array(o.matrix_world)[:3, :3])) ** (1 / 3) or 1.0
        local_height = height / scale
        weld = weld_vertices(co, 1e-5 * local_height)
        lengths = np.linalg.norm(co[edges[:, 0]] - co[edges[:, 1]], axis=1)
        edge_length = float(lengths[lengths > 0].mean()) if lengths.size else 0.0
        iterations = smoothing_iterations(strength * local_height, edge_length)
        smooth = smooth_normals(no.reshape(n, 3), edges, iterations, weld, co)
        attr = me.attributes.get("toon_normal") or me.attributes.new(
            "toon_normal", "FLOAT_VECTOR", "POINT"
        )
        attr.data.foreach_set("vector", smooth.astype(np.float32).ravel())
        print(
            f"[toon] {o.name}: smoothed normals, {iterations} passes over {n} vertices "
            f"({n - int(weld.max()) - 1} seam duplicates welded, {time.time() - t0:.1f}s)",
            flush=True,
        )
    return True


def add_front_projection(
    bpy,
    meshes,
    image_path,
    mask_path=None,
    facing=(0.2, 0.55),
    erode_px: int = 3,
    color_match: bool = True,
):
    """Project the source image onto the front of ``meshes`` (current world transforms).

    TRELLIS.2 generates its own texture, which blurs small details such as eyes. The
    character faces the camera in the source image, so for surfaces that face the front
    and are not hidden behind other geometry, the original pixels are better. Adds a
    ``front_proj`` UV map and a ``front_weight`` point attribute to each mesh, and (with
    ``color_match``) a ``toon_color`` attribute for everywhere else. Returns a packed RGBA
    image (alpha = eroded mask) and whether ``toon_color`` was written, for
    ``insert_projection``.
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

    items = []
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
        items.append((o, wco, weight, px, py))

    matched = add_color_match(items, rgba) if color_match else False
    proj = bpy.data.images.new("FrontProjection", w, h, alpha=True)
    out = rgba.copy()
    out[..., 3] = inner.astype(np.float32)
    proj.pixels.foreach_set(out.ravel())
    proj.alpha_mode = "STRAIGHT"
    proj.pack()
    return proj, matched


def base_color_image(mat):
    """The image texture feeding a material's Principled Base Color (glTF import layout)."""
    nt = mat.node_tree if mat is not None else None
    bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None) if nt else None
    if bsdf is None or not bsdf.inputs["Base Color"].is_linked:
        return None
    node = bsdf.inputs["Base Color"].links[0].from_node
    for _ in range(3):  # e.g. a Mix node in between when baseColorFactor != 1
        if node.type == "TEX_IMAGE":
            return node.image
        linked = [i for i in node.inputs if i.is_linked]
        if not linked:
            return None
        node = linked[0].links[0].from_node
    return None


def vertex_texture_colors(me, image, down: int = 2):
    """Mean texture colour at each vertex (raw pixel values, i.e. sRGB-encoded)."""
    import numpy as np

    w, h = image.size
    px = np.empty(w * h * 4, np.float32)
    image.pixels.foreach_get(px)
    tex = px.reshape(h, w, 4)[..., :3]
    if down > 1 and min(w, h) >= 4 * down:  # average down a little: less texture noise
        tex = tex[: h - h % down, : w - w % down]
        tex = tex.reshape(h // down, down, w // down, down, 3).mean(axis=(1, 3))
    th, tw = tex.shape[:2]
    layer = next((uv for uv in me.uv_layers if uv.active_render), None) or me.uv_layers.active
    uv = np.empty(len(me.loops) * 2, np.float32)
    layer.data.foreach_get("uv", uv)
    uv = uv.reshape(-1, 2) % 1.0
    samples = bilinear(tex, uv[:, 0] * tw - 0.5, uv[:, 1] * th - 0.5)
    loop_v = np.empty(len(me.loops), np.int32)
    me.loops.foreach_get("vertex_index", loop_v)
    n = len(me.vertices)
    count = np.maximum(np.bincount(loop_v, minlength=n), 1)
    sums = [np.bincount(loop_v, weights=samples[:, c], minlength=n) for c in range(3)]
    return np.stack(sums, axis=1) / count[:, None]


def add_color_match(items, rgba, min_samples: int = 500) -> bool:
    """Write a ``toon_color`` vertex colour: the TRELLIS texture recoloured to the source.

    ``items``: (object, world coords, front weight, pixel x, pixel y) per mesh, from
    ``add_front_projection``. Returns False (and changes nothing) when it cannot run.
    """
    import numpy as np

    guides = []
    for o, *_ in items:
        mats = {s.material for s in o.material_slots if s.material}
        image = base_color_image(next(iter(mats))) if len(mats) == 1 else None
        if image is None or image.size[0] == 0:
            print(f"[toon] colour match skipped: {o.name} has no single base texture", flush=True)
            return False
        colours = vertex_texture_colors(o.data, image)
        edges = np.empty(len(o.data.edges) * 2, np.int64)
        o.data.edges.foreach_get("vertices", edges)
        guides.append(smooth_vertex_values(colours, edges))
    guide = color_guide(np.concatenate(guides))
    xz = np.concatenate([it[1][:, [0, 2]] for it in items])
    front = np.concatenate([it[2] for it in items]) > 0.9
    if front.sum() < min_samples:
        print(f"[toon] colour match skipped: only {int(front.sum())} front samples", flush=True)
        return False
    px = np.concatenate([it[3] for it in items])[front]
    py = np.concatenate([it[4] for it in items])[front]
    src = median_sample(rgba[..., :3], px, py)
    pred = match_colors(xz[front], guide[front], src, xz, guide)
    linear = np.c_[srgb_to_linear(pred), np.ones(len(pred))].astype(np.float32)
    start = 0
    for o, wco, *_ in items:
        me, n = o.data, len(wco)
        attr = me.attributes.get("toon_color") or me.attributes.new(
            "toon_color", "FLOAT_COLOR", "POINT"
        )
        attr.data.foreach_set("color", linear[start : start + n].ravel())
        start += n
    print(f"[toon] colour match: {int(front.sum())} front samples -> {start} vertices", flush=True)
    return True


def insert_projection(bpy, mat, proj_image, vertex_color: bool = False) -> bool:
    """Blend ``proj_image`` (via ``front_proj`` UVs) into the colour feeding ToonShade.

    ``vertex_color``: use the colour-matched ``toon_color`` attribute instead of the
    TRELLIS texture underneath the projection.
    """
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
    if vertex_color:
        matched = nt.nodes.new("ShaderNodeAttribute")
        matched.attribute_type = "GEOMETRY"
        matched.attribute_name = "toon_color"
        matched.label = "Colour-matched TRELLIS texture"
        matched.location = (x - 700, y + 50)
        nt.links.new(matched.outputs["Color"], mix.inputs["A"])
    elif base is not None:
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


def action_fcurves(action) -> list:
    """F-curves of an action, for the legacy API and Blender 4.4+ layered actions."""
    if hasattr(action, "fcurves"):
        return list(action.fcurves)
    curves = []
    for layer in getattr(action, "layers", []):
        for strip in layer.strips:
            for bag in getattr(strip, "channelbags", []):
                curves.extend(bag.fcurves)
    return curves


def key_turntable(pivot, base_yaw: float, keys: list[tuple[int, float]]) -> None:
    """Key the pivot's spin at ``keys`` (from ``turntable_keys``), held until the next key."""
    for frame, turn in keys:
        pivot.rotation_euler.z = base_yaw + 2 * math.pi * turn
        pivot.keyframe_insert("rotation_euler", index=2, frame=frame)
    for fc in action_fcurves(pivot.animation_data.action):
        for kp in fc.keyframe_points:
            kp.interpolation = "CONSTANT"


def encode_mp4(frames_dir: Path, dest: Path, background: str, hold: int = 2) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print("[toon] ffmpeg not found; turntable frames kept in", frames_dir, flush=True)
        return
    # Only every ``hold``-th frame is rendered: play them at 24/hold fps, output 24 fps,
    # so each pose lasts ``hold`` video frames.
    cmd = [
        ffmpeg, "-y", "-loglevel", "error", "-framerate", f"24/{hold}", "-pattern_type", "glob",
        "-i", str(frames_dir / "f_*.png"), "-r", "24",
        "-vf", f"pad=ceil(iw/2)*2:ceil(ih/2)*2:color={background}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(dest),
    ]  # fmt: skip
    subprocess.run(cmd, check=True)
    print(f"[toon] turntable -> {dest}", flush=True)


if __name__ == "__main__":
    main(sys.argv)
