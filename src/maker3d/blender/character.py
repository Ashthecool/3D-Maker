"""A toon character built from scratch, one clean object per body part.

Instead of cutting up the single fused mesh TRELLIS.2 makes, this models the character
in code: every part is its own smooth, closed mesh (head, face, hair, torso, jacket,
upper arms, forearms, hands, thighs, shins, feet), coloured by role (skin, hair,
jacket, ...) and moved by the same skeleton and pose language as rig.py. Limbs are
capsules whose round ends meet in a ball at each joint, so bending never opens a gap.

Run inside Blender:
    blender -b --factory-startup -P character.py -- --out job_dir [--spec spec.json]
            [--pose cheer] [--animations wave,walk]

Writes ``stills/``, ``turntable.mp4`` (with ffmpeg), ``animations/`` and a packed
``character.blend``. The geometry helpers above the Blender section are numpy only.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path

if __package__ in (None, ""):  # run by Blender as a script: import the siblings directly
    sys.path.insert(0, str(Path(__file__).parent))
    import rig as R  # noqa: E402
    import toon_scene as ts  # noqa: E402
else:
    from maker3d.blender import rig as R
    from maker3d.blender import toon_scene as ts


# --------------------------------------------------------------------------------------
# Look of the character
# --------------------------------------------------------------------------------------
@dataclass
class CharacterSpec:
    """Colours (#rrggbb) and a few shape choices. Sizes are fractions of the height."""

    height: float = 1.6  # metres
    skin: str = "#ffe2d2"
    hair: str = "#dcd7e6"
    eyes: str = "#3f7fd6"
    jacket: str = "#c51f2c"
    trim: str = "#f1eef4"
    shorts: str = "#2b2733"
    stockings: str = "#29242f"
    boots: str = "#b4212c"
    soles: str = "#2c2329"
    mouth: str = "#a8434b"
    hair_length: float = 0.56  # height of the back hair's tips (0.8 = shoulder length)
    bangs: int = 7

    @classmethod
    def from_dict(cls, data: dict | None) -> CharacterSpec:
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (data or {}).items() if k in names})


ROLES = ("skin", "hair", "jacket", "trim", "shorts", "stockings", "boots", "soles")

# Joint positions (fractions of the height) of the template skeleton, A-pose, facing -Y.
_ARM = (math.sin(math.radians(46)), 0.0, -math.cos(math.radians(46)))


def template_joints() -> dict:
    """Joints for rig.build_rig, in model heights (x = left, -y = forward, z = up)."""
    j = {
        "pelvis": (0.0, 0.0, 0.525),
        "waist": (0.0, 0.002, 0.6),
        "chest": (0.0, 0.003, 0.69),
        "neck": (0.0, 0.006, 0.79),
        "head": (0.0, 0.008, 0.858),
        "head_top": (0.0, 0.008, 1.0),
    }
    for s, sign in (("L", 1), ("R", -1)):
        sh = (sign * 0.084, 0.004, 0.772)

        def along(p, d, k, sign=sign):
            return (p[0] + sign * d[0] * k, p[1] + d[1] * k, p[2] + d[2] * k)

        elbow = along(sh, _ARM, 0.142)
        wrist = along(elbow, _ARM, 0.122)
        j.update(
            {
                f"clavicle.{s}": (sign * 0.018, -0.008, 0.786),
                f"shoulder.{s}": sh,
                f"elbow.{s}": (elbow[0], elbow[1] + 0.006, elbow[2]),
                f"wrist.{s}": wrist,
                f"hand_tip.{s}": along(wrist, _ARM, 0.052),
                f"hip.{s}": (sign * 0.046, 0.0, 0.5),
                f"knee.{s}": (sign * 0.05, -0.006, 0.282),
                f"ankle.{s}": (sign * 0.052, 0.006, 0.056),
                f"ball.{s}": (sign * 0.052, -0.045, 0.02),
                f"toe_tip.{s}": (sign * 0.052, -0.078, 0.02),
            }
        )
    return j


# --------------------------------------------------------------------------------------
# Geometry (numpy only). A part is (verts, faces, face roles, weights {bone: (N,)}).
# --------------------------------------------------------------------------------------
@dataclass
class Part:
    verts: object  # (N, 3)
    faces: list  # tuples of vertex indices, outward winding
    roles: list  # role name per face
    weights: dict  # bone -> (N,) weight
    # Optional shading normals (N, 3); rows of zeros keep the mesh's own normal. Round
    # normals on the head give clean anime face shadows (Guilty Gear edits them by hand).
    normals: object = None

    def __add__(self, other: Part) -> Part:
        import numpy as np

        n, m = len(self.verts), len(other.verts)
        weights = {}
        for b in set(self.weights) | set(other.weights):
            a = self.weights.get(b, np.zeros(n))
            c = other.weights.get(b, np.zeros(m))
            weights[b] = np.concatenate([a, c])
        normals = None
        if self.normals is not None or other.normals is not None:
            normals = np.concatenate(
                [
                    np.zeros((n, 3)) if self.normals is None else self.normals,
                    np.zeros((m, 3)) if other.normals is None else other.normals,
                ]
            )
        return Part(
            np.concatenate([self.verts, other.verts]),
            self.faces + [tuple(i + n for i in f) for f in other.faces],
            self.roles + other.roles,
            weights,
            normals,
        )


HEAD_CENTRE = (0.0, 0.004, 0.928)  # skull centre, model heights


def round_normals(verts, centre, flatten=(1.0, 1.0, 1.0)):
    """Normals pointing away from ``centre``: a sphere's shading on any shape."""
    import numpy as np

    d = (np.asarray(verts, dtype=np.float64) - np.asarray(centre)) * np.asarray(flatten)
    return d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)


def _unit(v):
    import numpy as np

    v = np.asarray(v, dtype=np.float64)
    return v / max(float(np.linalg.norm(v)), 1e-12)


def frame(direction, side_hint=(1.0, 0.0, 0.0)):
    """(ax, ay) perpendicular to ``direction`` with ax x ay = direction."""
    import numpy as np

    d = _unit(direction)
    hint = np.asarray(side_hint, dtype=np.float64)
    if abs(float(np.dot(_unit(hint), d))) > 0.95:
        hint = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    ax = _unit(hint - np.dot(hint, d) * d)
    return ax, np.cross(d, ax)


def loft(rings, n: int = 16, cap_start=None, cap_end=None):
    """Mesh through elliptical rings ``(centre, ax, ay, rx, ry)``.

    ``cap_start``/``cap_end``: a pole point closing that end with a triangle fan, or None
    for an open end. With ax x ay pointing along the loft, faces wind outwards.
    """
    import numpy as np

    t = 2 * np.pi * np.arange(n) / n
    cos, sin = np.cos(t)[:, None], np.sin(t)[:, None]
    verts = [
        np.asarray(c) + cos * rx * np.asarray(ax) + sin * ry * np.asarray(ay)
        for c, ax, ay, rx, ry in rings
    ]
    verts = list(np.concatenate(verts))
    faces = []
    for r in range(len(rings) - 1):
        for k in range(n):
            a, b = r * n + k, r * n + (k + 1) % n
            faces.append((a, b, b + n, a + n))
    if cap_start is not None:
        p = len(verts)
        verts.append(np.asarray(cap_start, dtype=np.float64))
        faces += [(p, (k + 1) % n, k) for k in range(n)]
    if cap_end is not None:
        p = len(verts)
        verts.append(np.asarray(cap_end, dtype=np.float64))
        last = (len(rings) - 1) * n
        faces += [(p, last + k, last + (k + 1) % n) for k in range(n)]
    return np.array(verts), faces


def capsule(
    a,
    b,
    r0: float,
    r1: float,
    n: int = 16,
    body: int = 8,
    cap: int = 4,
    squash: float = 1.0,
    side=(1.0, 0.0, 0.0),
    profile=None,
):
    """Closed capsule from ``a`` to ``b`` with hemispherical ends of radius r0 and r1.

    The round ends are centred on ``a`` and ``b``, so two capsules that share an end
    point and radius stay joined at any angle. ``profile(t)`` scales the body radius
    (t = 0 at a, 1 at b); ``squash`` flattens the cross-section along ``ay``.
    Returns (verts, faces, t per vertex).
    """
    import numpy as np

    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    length = float(np.linalg.norm(b - a))
    d = (b - a) / length
    ax, ay = frame(d, side)
    rings, ts = [], []
    for j in range(1, cap + 1):  # start cap, from the pole to the equator
        alpha = 0.5 * math.pi * j / cap
        rings.append(
            (
                a - d * r0 * math.cos(alpha),
                ax,
                ay,
                r0 * math.sin(alpha),
                r0 * math.sin(alpha) * squash,
            )
        )
        ts.append(-r0 * math.cos(alpha) / length)
    for i in range(1, body):
        t = i / body
        r = (r0 + (r1 - r0) * t) * (profile(t) if profile else 1.0)
        rings.append((a + d * length * t, ax, ay, r, r * squash))
        ts.append(t)
    for j in range(cap, 0, -1):  # end cap, from the equator to the pole
        alpha = 0.5 * math.pi * j / cap
        rings.append(
            (
                b + d * r1 * math.cos(alpha),
                ax,
                ay,
                r1 * math.sin(alpha),
                r1 * math.sin(alpha) * squash,
            )
        )
        ts.append(1 + r1 * math.cos(alpha) / length)
    verts, faces = loft(rings, n, cap_start=a - d * r0, cap_end=b + d * r1)
    t_vert = np.concatenate([np.repeat(ts, n), [-r0 / length, 1 + r1 / length]])
    return verts, faces, t_vert


def face_values(values, faces):
    """Mean of a per-vertex value over each face."""
    import numpy as np

    values = np.asarray(values, dtype=np.float64)
    return np.array([values[list(f)].mean() for f in faces])


def bone_weights(points, bones: dict, power: float = 4.0):
    """Smooth weights {bone: (N,)} over a few bones by inverse distance to their segments."""
    import numpy as np

    points = np.asarray(points, dtype=np.float64)
    names = list(bones)
    d = np.stack(
        [
            R.point_segment_distance(points, np.asarray(h, float), np.asarray(t, float))
            for h, t in bones.values()
        ],
        axis=1,
    )
    w = (d + 1e-4) ** -power
    w /= w.sum(axis=1, keepdims=True)
    return {name: w[:, i] for i, name in enumerate(names)}


def limb(joints, bone, r0, r1, H, roles_by_t, side=(1.0, 0.0, 0.0), **kw) -> Part:
    """A capsule along ``bone`` (rig.DEFORM_BONES), rigidly weighted to it.

    ``roles_by_t``: [(t_from, role), ...] sorted by t; a face takes the role of the last
    entry whose t_from is below the face's position along the bone.
    """
    import numpy as np

    _, h, t = R.DEFORM_BONES[bone]
    verts, faces, tv = capsule(
        np.array(joints[h]) * H, np.array(joints[t]) * H, r0 * H, r1 * H, side=side, **kw
    )
    ft = face_values(tv, faces)
    roles = []
    for x in ft:
        role = roles_by_t[0][1]
        for start, r in roles_by_t:
            if x >= start:
                role = r
        roles.append(role)
    return Part(verts, faces, roles, {bone: np.ones(len(verts))})


def head_shape(H: float, n: int = 28, rings: int = 18):
    """Anime head: a round skull with a narrow, slightly forward chin. Returns (verts, faces)."""
    import numpy as np

    c = np.array([0.0, 0.004, 0.928]) * H
    rx, ry, rz = 0.069 * H, 0.074 * H, 0.072 * H
    ring_list = []
    for i in range(1, rings):  # polar angle from bottom (pi) to top (0)
        th = math.pi * (1 - i / rings)
        ring_list.append(
            (
                c + np.array([0, 0, rz * math.cos(th)]),
                (1.0, 0, 0),
                (0, 1.0, 0),
                rx * math.sin(th),
                ry * math.sin(th),
            )
        )
    verts, faces = loft(
        ring_list, n, cap_start=c - np.array([0, 0, rz]), cap_end=c + np.array([0, 0, rz])
    )
    # jaw: below the centre, narrow the sides, pull the back in, push the chin forward and down
    h = np.clip((c[2] - verts[:, 2]) / rz, 0.0, 1.0)
    verts[:, 0] *= 1 - 0.48 * h**1.4
    front = verts[:, 1] < c[1]
    verts[front, 1] -= 0.2 * ry * h[front] ** 1.6
    verts[~front, 1] = c[1] + (verts[~front, 1] - c[1]) * (1 - 0.45 * h[~front])
    verts[:, 2] = c[2] - (c[2] - verts[:, 2]) * (1 + 0.24 * h)
    return verts, faces


def hair_cap(H: float, spec: CharacterSpec, n: int = 32, rings: int = 12):
    """Shell over the skull, down to a hairline that is high at the front, low at the back."""
    import numpy as np

    c = np.array([0.0, 0.004, 0.928]) * H
    rx, ry, rz = 0.077 * H, 0.083 * H, 0.08 * H
    phi = 2 * np.pi * np.arange(n) / n  # 0 = character's left (+x), pi/2 = back (+y)
    back = 0.5 * (1 + np.sin(phi))  # 0 at the front, 1 at the back
    th_max = np.radians(62 + 70 * back)  # how far down from the crown the hair goes
    verts = [c + np.array([0, 0, rz])]
    for i in range(1, rings + 1):
        th = th_max * i / rings
        verts += list(
            np.stack(
                [
                    c[0] + rx * np.sin(th) * np.cos(phi),
                    c[1] + ry * np.sin(th) * np.sin(phi),
                    c[2] + rz * np.cos(th),
                ],
                1,
            )
        )
    verts = np.array(verts)
    faces = [(0, 1 + (k + 1) % n, 1 + k) for k in range(n)]
    for r in range(rings - 1):
        for k in range(n):
            a, b = 1 + r * n + k, 1 + r * n + (k + 1) % n
            faces.append((a, a + n, b + n, b))
    return verts, faces


def _catmull(points, samples: int):
    import numpy as np

    p = np.asarray(points, dtype=np.float64)
    p = np.concatenate([2 * p[:1] - p[1:2], p, 2 * p[-1:] - p[-2:-1]])
    out = []
    for i in range(1, len(p) - 2):
        for s in np.linspace(0, 1, samples, endpoint=False):
            p0, p1, p2, p3 = p[i - 1], p[i], p[i + 1], p[i + 2]
            out.append(
                0.5
                * (
                    2 * p1
                    + (-p0 + p2) * s
                    + (2 * p0 - 5 * p1 + 4 * p2 - p3) * s * s
                    + (-p0 + 3 * p1 - 3 * p2 + p3) * s**3
                )
            )
    out.append(p[-2])
    return np.array(out)


def strand(
    points, width: float, thickness: float, centre, n: int = 8, samples: int = 6, taper=None
):
    """A flat, tapering lock of hair along a smooth curve through ``points``.

    The flat side faces away from ``centre`` (the head). ``taper(t)`` scales the width
    (default: full at the root, a point at the tip). Returns (verts, faces).
    """
    import numpy as np

    path = _catmull(points, samples)
    centre = np.asarray(centre, dtype=np.float64)
    m = len(path)
    rings = []
    for i, p in enumerate(path):
        t = i / (m - 1)
        tangent = _unit(path[min(i + 1, m - 1)] - path[max(i - 1, 0)])
        out = p - centre
        out = _unit(out - np.dot(out, tangent) * tangent)
        side = np.cross(tangent, out)
        k = taper(t) if taper else (1 - t) ** 0.8 * min(1.0, 0.6 + 4 * t)
        k = max(k, 0.02)
        # ax x ay must point along the tangent: side x out = tangent
        rings.append((p, side, out, width * k, thickness * k))
    tip = path[-1] + _unit(path[-1] - path[-2]) * 0.002 * width
    return loft(rings[:-1], n, cap_start=path[0], cap_end=tip)


def hair_strands(H: float, spec: CharacterSpec):
    """Bangs, side locks and long back hair. Returns a list of (verts, faces, kind)."""
    import numpy as np

    c = np.array([0.0, 0.004, 0.928]) * H
    out = []

    def on_cap(phi_deg, th_deg, scale=1.0):
        ph, th = math.radians(phi_deg), math.radians(th_deg)
        r = np.array(
            [
                0.077 * math.sin(th) * math.cos(ph),
                0.083 * math.sin(th) * math.sin(ph),
                0.08 * math.cos(th),
            ]
        )
        return c + r * H * scale

    # bangs: from near the crown, over the forehead, ending just above the eyes
    nb = max(spec.bangs, 1)
    for i in range(nb):
        u = (i + 0.5) / nb * 2 - 1  # -1 .. 1 across the forehead
        phi = -90 + 62 * u  # -90 = front
        root = on_cap(phi, 18, 0.98)
        mid = on_cap(phi, 52, 1.07)
        tip = on_cap(phi * 0.97 - 90 * 0.03, 78 + 8 * abs(u), 1.06)
        tip[2] = (0.932 + 0.012 * abs(u) ** 2 - 0.006 * (i % 2)) * H
        v, f = strand([root, mid, tip], (0.03 - 0.004 * abs(u)) * H, 0.009 * H, c, n=8)
        out.append((v, f, "bangs"))
    # side locks: in front of the ears, down to the chest
    for sign in (1, -1):
        for k, (phi, dx, length) in enumerate(((-28, 0.0, 0.74), (-8, 0.012, 0.79))):
            ph = phi if sign > 0 else 180 - phi
            root = on_cap(ph, 40, 1.0)
            ear = on_cap(ph, 92, 1.12)
            low = np.array([sign * (0.082 + dx), -0.012 + 0.01 * k, 0.83]) * H
            tip = np.array([sign * (0.088 + dx), -0.02 + 0.012 * k, length]) * H
            v, f = strand([root, ear, low, tip], 0.026 * H, 0.009 * H, c, n=8)
            out.append((v, f, "side"))
    # back hair: wide, overlapping locks over the back of the head, falling straight down
    tip_z = spec.hair_length
    count = 13
    for i in range(count):
        u = (i + 0.5) / count * 2 - 1
        phi = 90 + 96 * u  # 90 = straight back, about +-6 = just behind the ears
        cos, sin = math.cos(math.radians(phi)), math.sin(math.radians(phi))
        root = on_cap(phi, 22, 0.98)
        crown = on_cap(phi, 62, 1.04)
        nape = on_cap(phi, 104, 1.07)
        shoulder = np.array([0.092 * cos, 0.035 + 0.05 * sin, 0.79])
        mid_z = 0.5 * (0.79 + tip_z)
        low = np.array([0.1 * cos, 0.072 + 0.012 * sin, mid_z])
        jitter = 0.012 * ((i * 7) % 3 - 1)
        tip = np.array([0.104 * cos, 0.074 + 0.01 * sin, tip_z + 0.025 * abs(u) + jitter])
        width = (0.05 - 0.008 * abs(u)) * H

        def taper(t):
            grow = min(1.0, 0.55 + 4 * t)
            return grow * (1.0 if t < 0.55 else max(0.0, 1 - ((t - 0.55) / 0.45) ** 1.6))

        pts = [root, crown, nape, shoulder * H, low * H, tip * H]
        v, f = strand(pts, width, 0.013 * H, c, n=8, samples=6, taper=taper)
        out.append((v, f, "back"))
    return out


def torso_rings(H: float, table, n: int, ax=(1.0, 0.0, 0.0), ay=(0.0, 1.0, 0.0)):
    import numpy as np

    return [(np.array([0.0, y, z]) * H, ax, ay, rx * H, ry * H) for z, rx, ry, y in table]


# z, half-width, half-depth, centre y (fractions of the height), bottom to top
TORSO = [
    (0.468, 0.03, 0.03, 0.0),
    (0.478, 0.058, 0.046, 0.0),
    (0.5, 0.077, 0.055, 0.0),
    (0.535, 0.082, 0.057, 0.0),
    (0.57, 0.072, 0.05, -0.001),
    (0.61, 0.06, 0.045, -0.003),
    (0.65, 0.062, 0.046, -0.005),
    (0.69, 0.07, 0.051, -0.008),
    (0.73, 0.075, 0.053, -0.008),
    (0.765, 0.08, 0.05, -0.002),
    (0.785, 0.072, 0.044, 0.002),
    (0.8, 0.05, 0.034, 0.004),
    (0.808, 0.024, 0.024, 0.006),
]
JACKET = [  # the jacket body, open at the hem (bottom) and at the collar (top)
    (0.592, 0.077, 0.06, -0.003),
    (0.61, 0.074, 0.058, -0.004),
    (0.65, 0.075, 0.058, -0.006),
    (0.69, 0.083, 0.062, -0.008),
    (0.73, 0.088, 0.064, -0.008),
    (0.762, 0.092, 0.06, -0.002),
    (0.785, 0.084, 0.054, 0.002),
    (0.801, 0.062, 0.044, 0.004),
    (0.81, 0.036, 0.034, 0.006),
    (0.835, 0.037, 0.036, 0.006),
]


def build_parts(spec: CharacterSpec) -> dict:
    """All parts of the character: {object name: Part}, in metres, rest pose."""
    import numpy as np

    H = spec.height
    J = template_joints()
    jw = {k: np.array(v) * H for k, v in J.items()}

    def segs(*names):
        return {b: (jw[R.DEFORM_BONES[b][1]], jw[R.DEFORM_BONES[b][2]]) for b in names}

    parts = {}

    # --- head: skull and neck --------------------------------------------------------
    hv, hf = head_shape(H)
    centre = np.array(HEAD_CENTRE) * H
    head = Part(hv, hf, ["skin"] * len(hf), {"head": np.ones(len(hv))}, round_normals(hv, centre))
    nv, nf, _ = capsule(jw["neck"] + np.array([0, 0, -0.012 * H]), jw["head"], 0.02 * H, 0.019 * H)
    neck = Part(nv, nf, ["skin"] * len(nf), {"neck": np.ones(len(nv))})
    parts["Head"] = head + neck

    # --- hair: cap and locks; long hair follows the chest further down ----------------
    cv, cf = hair_cap(H, spec)
    hair = Part(cv, cf, ["hair"] * len(cf), {"head": np.ones(len(cv))}, round_normals(cv, centre))
    for v, f, _kind in hair_strands(H, spec):
        hair = hair + Part(v, f, ["hair"] * len(f), {"head": np.ones(len(v))})
    z = hair.verts[:, 2] / H
    to_head = np.clip((z - 0.7) / 0.12, 0.0, 1.0) ** 1.5
    hair.weights = {"head": to_head, "chest": 1 - to_head}
    parts["Hair"] = hair

    # --- torso: body with shorts, skin above them (mostly under the jacket) ----------
    tv, tf = loft(
        torso_rings(H, TORSO, 24),
        24,
        cap_start=np.array([0, 0, 0.462]) * H,
        cap_end=np.array([0, 0.006, 0.812]) * H,
    )
    fz = face_values(tv[:, 2], tf) / H
    roles = ["shorts" if z < 0.556 else "trim" if z < 0.568 else "skin" for z in fz]
    parts["Torso"] = Part(tv, tf, roles, bone_weights(tv, segs("hips", "spine", "chest")))

    # --- jacket: cropped, with a white hem and collar ---------------------------------
    jv, jf = loft(torso_rings(H, JACKET, 28), 28)
    fz = face_values(jv[:, 2], jf) / H
    roles = ["trim" if z < 0.604 or z > 0.818 else "jacket" for z in fz]
    parts["Jacket"] = Part(jv, jf, roles, bone_weights(jv, segs("spine", "chest")))

    for s in R.SIDES:
        sx = 1.0 if s == "L" else -1.0
        side = (0.0, 0.0, 1.0)  # ring seams on top of the limbs
        parts[f"UpperArm.{s}"] = limb(
            J, f"upper_arm.{s}", 0.031, 0.027, H, [(-1, "jacket")], side=side,
            profile=lambda t: 1 + 0.12 * math.sin(math.pi * t), n=20,
        )  # fmt: skip
        parts[f"Forearm.{s}"] = limb(
            J, f"forearm.{s}", 0.027, 0.026, H, [(-1, "jacket"), (0.8, "trim")], side=side,
            profile=lambda t: 1 + 0.14 * t, n=20,
        )  # fmt: skip
        parts[f"Hand.{s}"] = limb(
            J, f"hand.{s}", 0.019, 0.021, H, [(-1, "skin")], side=(sx, 0.0, 0.0), squash=0.82,
            profile=lambda t: 1 + 0.08 * math.sin(math.pi * t), n=16, body=6,
        )  # fmt: skip
        parts[f"Thigh.{s}"] = limb(
            J, f"thigh.{s}", 0.047, 0.031, H,
            [(-1, "shorts"), (0.12, "skin"), (0.42, "trim"), (0.47, "stockings")],
            side=(0.0, -1.0, 0.0), profile=lambda t: 1 + 0.06 * math.sin(math.pi * t), n=20,
        )  # fmt: skip

        def shin_profile(t):
            calf = 1 + 0.16 * math.exp(-(((t - 0.3) / 0.2) ** 2))
            return calf if t < 0.5 else 1.45 - 0.2 * (t - 0.5)  # the boot shaft is wider

        parts[f"Shin.{s}"] = limb(
            J, f"shin.{s}", 0.031, 0.021, H, [(-1, "stockings"), (0.5, "boots")],
            side=(0.0, -1.0, 0.0), profile=shin_profile, n=20, body=14,
        )  # fmt: skip
        parts[f"Foot.{s}"] = boot_foot(jw, s, H)
    return parts


def boot_foot(jw, s: str, H: float) -> Part:
    """A boot from heel to toe: rings across the foot, flat sole, rounded toe."""
    import numpy as np

    ankle, toe = jw[f"ankle.{s}"], jw[f"toe_tip.{s}"]
    x = ankle[0]
    y_heel, y_toe = ankle[1] + 0.032 * H, toe[1] + 0.004 * H
    rings = []
    m = 12
    for i in range(m + 1):
        u = i / m  # 0 heel, 1 toe
        y = y_heel + (y_toe - y_heel) * u
        width = (0.024 + 0.012 * math.sin(math.pi * min(u * 1.25, 1.0)) - 0.006 * u**3) * H
        top = (0.075 - 0.05 * u**0.8) * H  # high at the ankle, low at the toe
        half_h = 0.5 * top
        rings.append((np.array([x, y, half_h]), (1.0, 0, 0), (0, 0, 1.0), width, half_h))
    rings[0] = (
        rings[0][0] + np.array([0, -0.004 * H, 0]),
        rings[0][1],
        rings[0][2],
        rings[0][3] * 0.7,
        rings[0][4] * 0.85,
    )
    # ax x ay = (1,0,0) x (0,0,1) = -y: the loft runs heel (+y) to toe (-y)
    verts, faces = loft(
        rings,
        20,
        cap_start=rings[0][0] + np.array([0, 0.012 * H, 0]),
        cap_end=rings[-1][0] + np.array([0, -0.012 * H, 0]),
    )
    verts[:, 2] = np.maximum(verts[:, 2], 0.0)  # flat sole on the floor
    fz = face_values(verts[:, 2], faces) / H
    roles = ["soles" if z < 0.006 else "boots" for z in fz]
    return Part(verts, faces, roles, {f"foot.{s}": np.ones(len(verts))})


# --------------------------------------------------------------------------------------
# Face texture (numpy only): eyes, brows, nose, mouth and blush on a transparent image
# --------------------------------------------------------------------------------------
FACE_BOX = (-0.07, 0.07, 0.83, 0.97)  # x0, x1, z0, z1 in model heights


def hex_rgb(value: str):
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) / 255 for i in (0, 2, 4))


def paint_face(spec: CharacterSpec, width: int = 512):
    """RGBA face texture (sRGB, row 0 = top) covering FACE_BOX."""
    import numpy as np

    x0, x1, z0, z1 = FACE_BOX
    height = int(round(width * (z1 - z0) / (x1 - x0)))
    px = (x1 - x0) / width  # one pixel in model heights
    xs = x0 + (np.arange(width) + 0.5) * px
    zs = z1 - (np.arange(height) + 0.5) * px
    X, Z = np.meshgrid(xs, zs)
    rgb = np.zeros((height, width, 3))
    alpha = np.zeros((height, width))

    def put(mask, color, a=1.0):
        nonlocal rgb, alpha
        m = np.clip(mask, 0, 1)[..., None] * a
        rgb = rgb * (1 - m) + np.asarray(color)[None, None, :] * m
        alpha = alpha + (1 - alpha) * m[..., 0]

    def fill(sdf):  # antialiased inside of a signed distance (negative inside)
        return np.clip(0.5 - sdf / px, 0, 1)

    def ellipse(cx, cz, rx, rz):
        return (np.sqrt(((X - cx) / rx) ** 2 + ((Z - cz) / rz) ** 2) - 1) * min(rx, rz)

    iris = np.array(hex_rgb(spec.eyes))
    dark = np.array([0.13, 0.09, 0.12])
    hair = np.array(hex_rgb(spec.hair))
    brow = hair * 0.55 + dark * 0.45
    for sign in (1, -1):
        ex, ez = sign * 0.031, 0.902
        sclera = ellipse(ex, ez, 0.0165, 0.0155)
        put(fill(sclera), (1.0, 1.0, 1.0))
        # iris: dark at the top, bright at the bottom, inside the white
        ir = ellipse(ex + sign * 0.0012, ez - 0.0015, 0.0118, 0.0158)
        grad = np.clip((Z - (ez - 0.016)) / 0.032, 0, 1)[..., None]
        col = (iris * 1.15 + 0.15) * (1 - grad) + iris * 0.42 * grad
        inside = np.minimum(fill(ir), fill(sclera))
        m = inside[..., None]
        rgb = rgb * (1 - m) + np.clip(col, 0, 1) * m
        put(
            np.minimum(fill(ellipse(ex + sign * 0.0012, ez - 0.001, 0.0052, 0.0085)), fill(sclera)),
            iris * 0.22,
        )
        # upper lash: a thick line along the top of the eye, with a flick at the outside
        upper = ellipse(ex, ez - 0.003, 0.0185, 0.0185)
        band = np.abs(upper + 0.0005) - 0.0019
        top = Z > ez + 0.002 - 0.25 * np.abs(X - ex)
        put(fill(band) * top, dark)
        flick = ellipse(ex + sign * 0.0175, ez + 0.0065, 0.0055, 0.0022)
        put(fill(flick), dark)
        lower = np.abs(ellipse(ex, ez + 0.002, 0.0165, 0.0175)) - 0.0006
        put(fill(lower) * (Z < ez - 0.01) * (sign * (X - ex) > 0.002), dark, 0.9)
        # highlights
        put(fill(ellipse(ex - sign * 0.004, ez + 0.0055, 0.0034, 0.0042)), (1, 1, 1))
        put(fill(ellipse(ex + sign * 0.0045, ez - 0.0075, 0.0017, 0.0017)), (1, 1, 1), 0.9)
        # brow: a thin arc
        b = np.abs(ellipse(ex + sign * 0.002, ez + 0.012, 0.02, 0.02)) - 0.0009
        put(fill(b) * (Z > ez + 0.026) * (np.abs(X - ex) < 0.016), brow)
        # blush: soft pink oval with three short strokes
        blush = np.exp(-(((X - sign * 0.036) / 0.012) ** 2) - ((Z - 0.878) / 0.0045) ** 2)
        put(blush, (1.0, 0.55, 0.6), 0.45)
    # nose: a tiny short stroke; mouth: a small smile
    put(fill(ellipse(0.0015, 0.874, 0.0012, 0.0022)), (0.85, 0.55, 0.5), 0.8)
    smile = np.abs(ellipse(0.0, 0.862, 0.009, 0.006)) - 0.0007
    put(fill(smile) * (Z < 0.861), hex_rgb(spec.mouth))
    # rgb is premultiplied by alpha while painting; images store straight colour
    straight = rgb / np.maximum(alpha, 1e-6)[..., None]
    return np.dstack([np.clip(straight, 0, 1), np.clip(alpha, 0, 1)])


# --------------------------------------------------------------------------------------
# Blender side
# --------------------------------------------------------------------------------------
def parse_args(argv: list[str]) -> argparse.Namespace:
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]

    p = argparse.ArgumentParser(prog="character.py", description=__doc__.split("\n")[0])
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--spec", type=Path, help="JSON with CharacterSpec fields (colours, ...)")
    ts.add_render_args(p)
    return p.parse_args(argv)


def _mesh_object(bpy, name: str, part: Part, materials: dict):
    import bmesh
    import numpy as np

    me = bpy.data.meshes.new(name)
    me.from_pydata(part.verts.tolist(), [], [list(f) for f in part.faces])
    order = []
    for role in part.roles:
        if role not in order:
            order.append(role)
    for role in order:
        me.materials.append(materials[role])
    me.polygons.foreach_set(
        "material_index", np.array([order.index(r) for r in part.roles], np.int32)
    )
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(me)
    bm.free()
    me.shade_smooth()
    if part.normals is not None:
        own = np.empty(len(me.vertices) * 3, np.float32)
        me.vertices.foreach_get("normal", own)
        own = own.reshape(-1, 3)
        custom = np.asarray(part.normals, dtype=np.float64)
        keep = np.linalg.norm(custom, axis=1) < 0.5
        custom[keep] = own[keep]
        me.normals_split_custom_set_from_vertices(custom.tolist())
    obj = bpy.data.objects.new(name, me)
    for bone, w in part.weights.items():
        vg = obj.vertex_groups.new(name=bone)
        q = np.round(np.asarray(w) * 255).astype(int)
        for level in np.unique(q[q > 0]):
            vg.add(np.flatnonzero(q == level).tolist(), float(level) / 255, "REPLACE")
    return obj


def _flat_material(bpy, name: str, color: str):

    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = (*ts.hex_to_rgb(color), 1.0)
    return mat


def face_object(bpy, head_obj, spec: CharacterSpec, materials: dict):
    """A thin decal over the front of the head carrying the painted face."""
    import numpy as np
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree

    H = spec.height
    depsgraph = bpy.context.evaluated_depsgraph_get()
    bvh = BVHTree.FromObject(head_obj, depsgraph)
    x0, x1, z0, z1 = FACE_BOX
    nx, nz = 36, 36
    verts, uvs, ok = [], [], []
    for j in range(nz + 1):
        for i in range(nx + 1):
            u, v = i / nx, j / nz
            x, z = (x0 + (x1 - x0) * u) * H, (z0 + (z1 - z0) * v) * H
            hit, normal, _, _ = bvh.ray_cast(Vector((x, -1.0 * H, z)), Vector((0, 1, 0)))
            good = hit is not None and normal.y < -0.15
            p = hit + normal * 0.0012 * H if good else Vector((x, 0, z))
            verts.append(tuple(p))
            uvs.append((u, v))
            ok.append(good)
    faces = []
    for j in range(nz):
        for i in range(nx):
            a = j * (nx + 1) + i
            quad = (a, a + 1, a + nx + 2, a + nx + 1)
            if all(ok[k] for k in quad):
                faces.append(quad)
    me = bpy.data.meshes.new("Face")
    me.from_pydata(verts, [], faces)
    uv = me.uv_layers.new(name="UVMap")
    loop_v = np.empty(len(me.loops), np.int32)
    me.loops.foreach_get("vertex_index", loop_v)
    uv.data.foreach_set("uv", np.array(uvs, np.float32)[loop_v].ravel())
    me.shade_smooth()
    me.materials.append(materials["face"])
    obj = bpy.data.objects.new("Face", me)
    obj.vertex_groups.new(name="head").add(list(range(len(verts))), 1.0, "REPLACE")
    return obj


def face_material(bpy, spec: CharacterSpec):
    """Unshaded painted face with transparency (anime eyes stay bright)."""
    import numpy as np

    rgba = paint_face(spec)[::-1]  # Blender images start at the bottom row
    h, w = rgba.shape[:2]
    img = bpy.data.images.new("FaceTexture", w, h, alpha=True)
    img.pixels.foreach_set(rgba.astype(np.float32).ravel())
    img.pack()
    img.colorspace_settings.name = "sRGB"
    mat = bpy.data.materials.new("Face")
    mat.use_nodes = True
    nt = mat.node_tree
    for node in list(nt.nodes):
        nt.nodes.remove(node)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = img
    tex.interpolation = "Cubic"
    emit = nt.nodes.new("ShaderNodeEmission")
    clear = nt.nodes.new("ShaderNodeBsdfTransparent")
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(tex.outputs["Color"], emit.inputs["Color"])
    nt.links.new(tex.outputs["Alpha"], mix.inputs["Fac"])
    nt.links.new(clear.outputs["BSDF"], mix.inputs[1])
    nt.links.new(emit.outputs["Emission"], mix.inputs[2])
    nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])
    if hasattr(mat, "surface_render_method"):
        mat.surface_render_method = "BLENDED"
    ts.no_light_sampling(mat)
    return mat


# Thinner lines on small or delicate parts (the guide: no heavy lines on faces and fingers).
LINE_WEIGHT = {"Hand": 0.6, "Head": 0.8, "Hair": 0.85, "Foot": 0.9}


def add_role_outline(obj, lines: dict, thickness: float) -> None:
    """Inverted-hull outline whose colour follows each face's role.

    The hull of a face with material slot i uses slot i + n (n = the number of role
    slots), so the line slots are appended in the same order as the role slots.
    """
    mats = list(obj.data.materials)
    n = len(mats)
    for mat in mats:
        role = next((r for r in ROLES if mat.name.lower().startswith(r)), "skin")
        obj.data.materials.append(lines[role])
    mod = obj.modifiers.new("ToonOutline", "SOLIDIFY")
    mod.thickness = thickness
    mod.offset = 1.0
    mod.use_flip_normals = True
    mod.use_rim = False
    mod.use_even_offset = False
    mod.use_quality_normals = True
    mod.material_offset = n


def build_character(
    bpy, spec: CharacterSpec, style: str = "cel", bands: int = 2, outline: float = 0.0035
):
    """Rig + one object per part, toon materials and outlines. Returns (rig, objects)."""

    H = spec.height
    joints = {k: tuple(c * H for c in v) for k, v in template_joints().items()}
    rig = R.build_rig(bpy, joints, H, name=R.RIG_NAME)
    toon_group = ts.build_toon_group(bpy, style, bands, smooth_normals=False)
    materials = {}
    for role in ROLES:
        mat = _flat_material(bpy, role.capitalize(), getattr(spec, role))
        ts.make_toon_material(bpy, mat, toon_group)
        materials[role] = mat
    materials["face"] = face_material(bpy, spec)
    # Coloured line art: every colour role gets its own, darker line colour.
    lines = {}
    if outline > 0:
        for role in ROLES:
            color = ts.line_color(ts.hex_to_rgb(getattr(spec, role)))
            lines[role] = ts.build_outline_material(bpy, color, f"Line{role.capitalize()}")
    coll = bpy.data.collections.new("Character")
    bpy.context.scene.collection.children.link(coll)
    objects = {}
    for name, part in build_parts(spec).items():
        obj = _mesh_object(bpy, name, part, materials)
        coll.objects.link(obj)
        objects[name] = obj
    objects["Face"] = face_object(bpy, objects["Head"], spec, materials)
    coll.objects.link(objects["Face"])
    objects["Head"]["toon_shadow_bias"] = 0.22  # keep the face lit (see ToonShade)
    for name, obj in objects.items():
        mod = obj.modifiers.new("Armature", "ARMATURE")
        mod.object = rig
        obj.parent = rig
        if name != "Face":
            sub = obj.modifiers.new("Smooth", "SUBSURF")
            sub.levels = sub.render_levels = 1
            sub.use_custom_normals = obj.data.has_custom_normals
        if lines and name != "Face":
            weight = LINE_WEIGHT.get(name.split(".")[0], 1.0)
            add_role_outline(obj, lines, ts.outline_thickness(H, outline * weight))
    return rig, objects


def main(argv: list[str]) -> None:
    import bpy

    args = parse_args(argv)
    spec = CharacterSpec.from_dict(json.loads(args.spec.read_text("utf-8")) if args.spec else None)
    out = args.out.resolve()
    (out / "stills").mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    rig, objects = build_character(
        bpy, spec, args.style, args.bands, 0.0 if args.no_outline else args.outline
    )
    pivot = bpy.data.objects.new("ToonPivot", None)
    scene.collection.objects.link(pivot)
    rig.parent = pivot
    if args.pose:
        R.apply_pose(rig, R.resolve_pose(args.pose))
    R.embed(bpy)
    (out / "spec.json").write_text(json.dumps(asdict(spec), indent=2), "utf-8")
    anims = [a.strip() for a in args.animations.split(",") if a.strip()]
    ts.render_outputs(
        bpy,
        scene,
        pivot,
        list(objects.values()),
        args,
        out,
        rig=rig,
        animations=anims,
        blend_name="character.blend",
    )


if __name__ == "__main__":
    main(sys.argv)
