"""Automatic humanoid rig for TRELLIS.2 characters, and a pose language for it (issue #19).

The characters come out of Anima in an A-pose (prompt preset), standing on the origin and
facing -Y after toon_scene.py centres them. This module:

1. Voxelizes the mesh into a solid grid (``voxelize``).
2. Finds the joints from the front silhouette and the grid (``find_joints``): legs from
   the two runs at the bottom, arms from the hand tips inwards, spine and head from
   proportions and the side profile.
3. Skins the mesh with geodesic voxel weights (``skin_weights``): the distance from each
   bone *through the inside of the model*, so the left thigh does not pull the right
   one and a hair strand hanging by the arm stays with the head.
4. Builds the armature in Blender (``build_rig``): deform bones, IK legs (on by default)
   and arms (off, i.e. FK), pole targets, root and torso controls, colours and shapes.

Poses are written in plain words instead of quaternions (``POSE_CHANNELS``), for example
``{"upper_arm.L": {"raise": 150}, "forearm.L": {"bend": 40}, "head": {"turn": 20}}``.
Angles are degrees from the rest pose, except ``upper_arm.raise`` (0 = arm down at the
side, 90 = straight out, 180 = straight up). Moves are in model heights, in the
character's frame: ``{"torso": {"move": [0, 0, -0.05]}}`` crouches (x = the character's
left, y = forward, z = up). See docs/rig.md.

Everything above the Blender section imports without ``bpy`` (numpy only), for tests.
Inside Blender the same file is also stored in the .blend as the text "toon_rig.py",
so a pose can be applied there with ``exec`` (see docs/rig.md).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

# __file__ is missing when this runs as the text block embedded in a .blend
HERE = Path(globals().get("__file__", "toon_rig.py")).parent
POSES_FILE = HERE / "poses.json"
ANIMATIONS_DIR = HERE / "animations"
SIDES = ("L", "R")

# --------------------------------------------------------------------------------------
# Skeleton layout
# --------------------------------------------------------------------------------------
# name -> (parent, head joint, tail joint). Joints come from find_joints().
DEFORM_BONES = {
    "hips": (None, "pelvis", "waist"),
    "spine": ("hips", "waist", "chest"),
    "chest": ("spine", "chest", "neck"),
    "neck": ("chest", "neck", "head"),
    "head": ("neck", "head", "head_top"),
}
for _s in SIDES:
    DEFORM_BONES.update(
        {
            f"shoulder.{_s}": ("chest", f"clavicle.{_s}", f"shoulder.{_s}"),
            f"upper_arm.{_s}": (f"shoulder.{_s}", f"shoulder.{_s}", f"elbow.{_s}"),
            f"forearm.{_s}": (f"upper_arm.{_s}", f"elbow.{_s}", f"wrist.{_s}"),
            f"hand.{_s}": (f"forearm.{_s}", f"wrist.{_s}", f"hand_tip.{_s}"),
            f"thigh.{_s}": ("hips", f"hip.{_s}", f"knee.{_s}"),
            f"shin.{_s}": (f"thigh.{_s}", f"knee.{_s}", f"ankle.{_s}"),
            f"foot.{_s}": (f"shin.{_s}", f"ankle.{_s}", f"ball.{_s}"),
            f"toe.{_s}": (f"foot.{_s}", f"ball.{_s}", f"toe_tip.{_s}"),
        }
    )
# Separate objects the character is cut into (split_parts): name -> deform bones.
PARTS = {"Head": ("head", "neck"), "Torso": ("hips", "spine", "chest", "shoulder.L", "shoulder.R")}
for _s in SIDES:
    PARTS.update(
        {
            f"UpperArm.{_s}": (f"upper_arm.{_s}",),
            f"Forearm.{_s}": (f"forearm.{_s}",),
            f"Hand.{_s}": (f"hand.{_s}",),
            f"Thigh.{_s}": (f"thigh.{_s}",),
            f"Shin.{_s}": (f"shin.{_s}",),
            f"Foot.{_s}": (f"foot.{_s}", f"toe.{_s}"),
        }
    )
# Bones that lie along the ground (foot, toe) keep their local Z up; all others face
# forward, so every bone's local X is a "swing forward" axis (see POSE_CHANNELS).
FLAT_BONES = {f"{b}.{s}" for b in ("foot", "toe") for s in SIDES}

# Pose channels: word -> (local Euler axis, sign on .L / centre bones, sign on .R).
# Rotation mode YXZ: Z (raise/lean) first, then X (forward/bend), then Y (twist).
_UPRIGHT = {"bend": ("x", 1, 1), "lean": ("z", -1, -1), "turn": ("y", 1, 1)}
_ARM = {"raise": ("z", 1, -1), "forward": ("x", 1, 1), "twist": ("y", 1, -1)}
_FOOT = {"point": ("x", -1, -1), "tilt": ("y", 1, -1), "out": ("z", 1, -1)}
POSE_CHANNELS = {
    "root": _UPRIGHT,
    "torso": _UPRIGHT,
    "hips": _UPRIGHT,
    "spine": _UPRIGHT,
    "chest": _UPRIGHT,
    "neck": _UPRIGHT,
    "head": _UPRIGHT,
    "shoulder": _ARM,
    "upper_arm": _ARM,
    "forearm": {"bend": ("x", 1, 1), "twist": ("y", 1, -1)},
    "hand": {"bend": ("x", 1, 1), "side": ("z", 1, -1), "twist": ("y", 1, -1)},
    "hand_ik": {"bend": ("x", 1, 1), "side": ("z", 1, -1), "twist": ("y", 1, -1)},
    "thigh": {"forward": ("x", 1, 1), "out": ("z", 1, -1), "twist": ("y", 1, -1)},
    "shin": {"bend": ("x", -1, -1), "twist": ("y", 1, -1)},
    "foot": _FOOT,
    "foot_ik": _FOOT,
    "toe": {"curl": ("x", -1, -1)},
    "knee_pole": {},
    "elbow_pole": {},
}
MOVABLE = {"root", "torso", "foot_ik", "hand_ik", "knee_pole", "elbow_pole"}
# Rig switches (custom properties on the armature object): 1 = IK, 0 = FK.
SWITCHES = {f"ik_leg.{s}": 1.0 for s in SIDES} | {f"ik_arm.{s}": 0.0 for s in SIDES}


def bone_kind(name: str) -> tuple[str, str]:
    """('upper_arm', 'L') for 'upper_arm.L'; ('head', '') for 'head'."""
    base, _, side = name.partition(".")
    return base, side


def semantic_euler(bone: str, values: dict, rest_raise: float = 0.0) -> tuple[float, ...]:
    """Local Euler angles (x, y, z) in radians, rotation mode YXZ, for pose words.

    ``rest_raise``: for upper arms, the rest pose's arm elevation in degrees (0 = down),
    because their "raise" is absolute.
    """
    base, side = bone_kind(bone)
    channels = POSE_CHANNELS.get(base)
    if channels is None:
        raise ValueError(f"unknown bone {bone!r}")
    angles = {"x": 0.0, "y": 0.0, "z": 0.0}
    for word, value in values.items():
        if word in ("move", "ik"):
            continue
        if word not in channels:
            raise ValueError(f"{bone}: unknown word {word!r}; use {sorted(channels)}")
        axis, sign_l, sign_r = channels[word]
        degrees = float(value)
        if base == "upper_arm" and word == "raise":
            degrees -= rest_raise
        angles[axis] += degrees * (sign_r if side == "R" else sign_l)
    return tuple(math.radians(angles[a]) for a in "xyz")


def merge_poses(*layers: dict) -> dict:
    """Merge pose dicts left to right, word by word (later layers win)."""
    out: dict = {}
    for layer in layers:
        for bone, values in layer.items():
            if bone.startswith("ik_"):
                out[bone] = values
                continue
            out[bone] = {**out.get(bone, {}), **values}
    return out


def mirror_pose(pose: dict) -> dict:
    """The same pose with left and right swapped (moves and turns flip sides)."""
    out = {}
    for bone, values in pose.items():
        name = bone
        for a, b in ((".L", ".R"), (".R", ".L")):
            if bone.endswith(a):
                name = bone[: -len(a)] + b
        if bone.startswith("ik_"):
            out[name] = values
            continue
        new = dict(values)
        if "move" in new:
            x, y, z = new["move"]
            new["move"] = [-x, y, z]
        if bone_kind(bone)[1] == "":
            for word in ("lean", "turn"):
                if word in new:
                    new[word] = -new[word]
        out[name] = new
    return out


def load_library(path: Path = POSES_FILE) -> dict:
    """The pose library: poses.json, or the copy embedded in an open .blend."""
    path = Path(path)
    if not path.is_file():
        bundle = _embedded_bundle()
        if bundle is not None:
            return bundle["poses"]
    return json.loads(path.read_text("utf-8"))


def resolve_pose(spec, library: dict | None = None) -> dict:
    """A pose from a dict, a library name, "mirror:<name>", or a list of those (merged)."""
    if isinstance(spec, dict):
        return spec
    if isinstance(spec, str):
        library = load_library() if library is None else library
        if spec.startswith("mirror:"):
            return mirror_pose(resolve_pose(spec[len("mirror:") :], library))
        if spec not in library:
            raise ValueError(f"unknown pose {spec!r}; library has {sorted(library)}")
        return resolve_pose(library[spec], library)
    if isinstance(spec, list):
        return merge_poses(*(resolve_pose(s, library) for s in spec))
    raise ValueError(f"a pose is a dict, a name or a list, not {type(spec).__name__}")


def validate_pose(pose: dict) -> None:
    """Raise ValueError for unknown bones, words or malformed moves."""
    for bone, values in pose.items():
        if bone in SWITCHES:
            if not 0.0 <= float(values) <= 1.0:
                raise ValueError(f"{bone} must be between 0 (FK) and 1 (IK)")
            continue
        semantic_euler(bone, values)
        if "move" in values:
            if bone_kind(bone)[0] not in MOVABLE:
                raise ValueError(f"{bone} cannot move; only {sorted(MOVABLE)} can")
            if len(values["move"]) != 3:
                raise ValueError(f"{bone}: move is [left, forward, up]")


@dataclass
class Animation:
    name: str
    frames: int
    keys: list[tuple[int, dict]]
    hold: int = 2
    loop: bool = False
    interpolation: str = "smooth"  # smooth | linear | constant
    description: str = ""
    camera: dict = field(default_factory=dict)


def parse_animation(data: dict, library: dict | None = None) -> Animation:
    """Check an animation dict (see docs/rig.md) and resolve its poses.

    Every key is a full pose: bones and words that a key leaves out go back to the rest
    pose, so a key reads the same wherever it is. With ``loop``, the first key is
    repeated one frame after the end, so the cycle closes smoothly.
    """
    frames = int(data["frames"])
    if frames < 2:
        raise ValueError("an animation needs at least 2 frames")
    interpolation = data.get("interpolation", "smooth")
    if interpolation not in ("smooth", "linear", "constant"):
        raise ValueError("interpolation is smooth, linear or constant")
    hold = int(data.get("hold", 2))
    if hold not in (1, 2, 3, 4):
        raise ValueError("hold is 1 to 4 frames")
    keys = []
    for key in data["keys"]:
        frame = int(key["frame"])
        if not 1 <= frame <= frames:
            raise ValueError(f"key frame {frame} is outside 1..{frames}")
        pose = resolve_pose(key.get("pose", {}), library)
        validate_pose(pose)
        keys.append((frame, pose))
    if not keys:
        raise ValueError("an animation needs at least one key")
    keys.sort(key=lambda k: k[0])
    if len({f for f, _ in keys}) != len(keys):
        raise ValueError("two keys on the same frame")
    loop = bool(data.get("loop", False))
    if loop:
        keys.append((frames + 1, keys[0][1]))
    return Animation(
        name=str(data.get("name", "animation")),
        frames=frames,
        keys=keys,
        hold=hold,
        loop=loop,
        interpolation=interpolation,
        description=str(data.get("description", "")),
        camera=dict(data.get("camera", {})),
    )


def animation_names(directory: Path = ANIMATIONS_DIR) -> list[str]:
    return sorted(p.stem for p in Path(directory).glob("*.json"))


def load_animation(name_or_path: str, library: dict | None = None) -> Animation:
    """An animation from the built-in library (by name) or a JSON file path."""
    path = Path(name_or_path)
    if not path.suffix:
        path = ANIMATIONS_DIR / f"{name_or_path}.json"
        bundle = None if path.is_file() else _embedded_bundle()
        if bundle is not None and name_or_path in bundle["animations"]:
            return parse_animation(
                {"name": name_or_path, **bundle["animations"][name_or_path]}, library
            )
    if not path.is_file():
        raise ValueError(f"no animation {name_or_path!r}; built in: {animation_names()}")
    data = json.loads(path.read_text("utf-8"))
    data.setdefault("name", path.stem)
    return parse_animation(data, library)


def pole_angle(base_head, base_tail, base_x_axis, ik_tail, pole) -> float:
    """IK pole angle (radians) that keeps the chain in its rest orientation.

    The angle between the chain root's X axis and the pole direction, both projected
    onto the plane normal to the chain root bone, signed around that bone.
    """
    import numpy as np

    head, tail = np.asarray(base_head, float), np.asarray(base_tail, float)
    axis = tail - head
    pole_normal = np.cross(np.asarray(ik_tail, float) - head, np.asarray(pole, float) - head)
    projected = np.cross(pole_normal, axis)
    x = np.asarray(base_x_axis, float)
    cos = np.dot(x, projected) / (np.linalg.norm(x) * np.linalg.norm(projected) + 1e-12)
    angle = math.acos(max(-1.0, min(1.0, float(cos))))
    if np.dot(np.cross(x, projected), axis) < 0:
        angle = -angle
    return angle


# --------------------------------------------------------------------------------------
# Voxel grid (numpy only)
# --------------------------------------------------------------------------------------
@dataclass
class Grid:
    lo: object  # (3,) world position of voxel (0, 0, 0)'s corner
    size: float  # voxel edge
    solid: object  # (X, Y, Z) bool: inside or on the surface
    shell: object  # (X, Y, Z) bool: on the surface

    def index(self, points):
        import numpy as np

        idx = np.floor((np.asarray(points, float) - self.lo) / self.size).astype(int)
        return np.clip(idx, 0, np.array(self.solid.shape) - 1)

    def center(self, idx):
        import numpy as np

        return self.lo + (np.asarray(idx, float) + 0.5) * self.size


def surface_points(co, tris, spacing: float):
    """Points on every triangle, no further apart than ``spacing`` (vertices included)."""
    import numpy as np

    co, tris = np.asarray(co, float), np.asarray(tris, int).reshape(-1, 3)
    out = [co]
    if len(tris) == 0:
        return co
    a, b, c = co[tris[:, 0]], co[tris[:, 1]], co[tris[:, 2]]
    sides = [np.linalg.norm(p - q, axis=1) for p, q in ((b, a), (c, b), (a, c))]
    longest = np.max(sides, axis=0)
    k_all = np.ceil(longest / spacing).astype(int)
    for k in np.unique(k_all[k_all > 1]):
        sel = k_all == k
        i, j = np.meshgrid(np.arange(k + 1), np.arange(k + 1), indexing="ij")
        keep = i + j <= k
        u, v = i[keep] / k, j[keep] / k
        w = 1.0 - u - v
        pa, pb, pc = a[sel], b[sel], c[sel]
        pts = pa[:, None] * w[None, :, None] + pb[:, None] * u[None, :, None]
        pts += pc[:, None] * v[None, :, None]
        out.append(pts.reshape(-1, 3))
    return np.concatenate(out)


def _shift(a, axis: int, step: int, fill=False):
    import numpy as np

    out = np.full_like(a, fill)
    src = [slice(None)] * a.ndim
    dst = [slice(None)] * a.ndim
    if step > 0:
        src[axis], dst[axis] = slice(0, -step), slice(step, None)
    else:
        src[axis], dst[axis] = slice(-step, None), slice(0, step)
    out[tuple(dst)] = a[tuple(src)]
    return out


def dilate6(a):
    out = a.copy()
    for axis in range(a.ndim):
        out |= _shift(a, axis, 1) | _shift(a, axis, -1)
    return out


def dilate26(a):
    out = a
    for axis in range(a.ndim):
        out = out | _shift(out, axis, 1) | _shift(out, axis, -1)
    return out


def voxelize(co, tris, voxel: float, pad: int = 2) -> Grid:
    """Solid voxel grid: the surface, plus everything a flood from outside cannot reach."""
    import numpy as np

    co = np.asarray(co, float)
    lo = co.min(axis=0) - pad * voxel
    dims = np.ceil((co.max(axis=0) - lo) / voxel).astype(int) + pad + 1
    shell = np.zeros(tuple(dims), bool)
    idx = np.floor((surface_points(co, tris, voxel * 0.45) - lo) / voxel).astype(int)
    idx = np.clip(idx, 0, dims - 1)
    shell[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    free = ~shell
    outside = np.zeros_like(shell)
    for axis in range(3):
        for end in (0, -1):
            sl = [slice(None)] * 3
            sl[axis] = end
            outside[tuple(sl)] = free[tuple(sl)]
    while True:
        grown = dilate6(outside) & free
        if grown.sum() == outside.sum():
            break
        outside = grown
    return Grid(lo=lo, size=float(voxel), solid=~outside, shell=shell)


def first_run(column, start: int = 0):
    """(first, last) index of the first True run in a 1-D bool array, or None."""
    import numpy as np

    hits = np.flatnonzero(column[start:])
    if hits.size == 0:
        return None
    a = int(hits[0]) + start
    gaps = np.flatnonzero(~column[a:])
    b = a + int(gaps[0]) - 1 if gaps.size else len(column) - 1
    return a, b


def runs(row):
    """List of (first, last) for every True run in a 1-D bool array."""
    import numpy as np

    r = np.asarray(row, bool).astype(np.int8)
    d = np.diff(np.concatenate([[0], r, [0]]))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


# --------------------------------------------------------------------------------------
# Joint finding (numpy only)
# --------------------------------------------------------------------------------------
class RigError(ValueError):
    """The model does not look like an A-pose humanoid; no rig is built."""


def _depth_center(
    grid: Grid, x: float, z: float, cap: float, radius: int = 1, min_run: float = 0.0
):
    """World y of the middle of the first solid run along +y (from the front) at x, z.

    Looks at a small (2r+1)^2 patch of columns and keeps the median, so one stray
    column does not decide. ``cap``: the run is cut to this depth (hair behind a back).
    ``min_run``: thinner runs in front (an open jacket, a strand of hair) are skipped.
    """
    import numpy as np

    ix, _, iz = grid.index([[x, 0.0, z]])[0]
    mids = []
    need = max(int(min_run / grid.size), 1)
    for dx in range(-radius, radius + 1):
        for dz in range(-radius, radius + 1):
            cx, cz = ix + dx, iz + dz
            if not (0 <= cx < grid.solid.shape[0] and 0 <= cz < grid.solid.shape[2]):
                continue
            column = grid.solid[cx, :, cz]
            found = [r for r in runs(column) if r[1] - r[0] + 1 >= need] or runs(column)[:1]
            if not found:
                continue
            a, b = found[0]
            b = min(b, a + max(int(cap / grid.size), 1))
            mids.append(grid.lo[1] + (a + b + 1) / 2 * grid.size)
    if not mids:
        return None
    return float(np.median(mids))


def find_joints(grid: Grid, height: float) -> dict:
    """World positions of the joints of an A-pose character standing on z = 0, facing -Y.

    Returns ``{name: (x, y, z)}`` for every joint DEFORM_BONES uses, plus "rest_raise.L"
    and "rest_raise.R" (arm elevation in degrees, 0 = down) and "info" (diagnostics).
    Raises RigError when the silhouette does not show two legs and two arms.

    - Legs: the two runs of the silhouette near the ground, followed up until they
      join (the crotch, or a skirt hem); knees and ankles by proportion along them.
    - Neck: the most recessed point of the front profile between chest and face.
      Head, neck base and shoulder height hang off it.
    - Arms: each hand tip is the outermost point; the arm is followed inwards until it
      joins the torso. The wrist is where the thin hand meets the thicker sleeve.
    - Depth: the middle of the first solid run seen from the front, skipping thin layers.
    """
    import numpy as np

    solid, v, lo = grid.solid, grid.size, grid.lo
    front = solid.any(axis=1)  # (X, Z): the silhouette seen from the front
    nx, nz = front.shape
    xs = lo[0] + (np.arange(nx) + 0.5) * v
    zs = lo[2] + (np.arange(nz) + 0.5) * v
    iz0 = int(np.searchsorted(zs, 0.0))
    ztop = float(zs[np.flatnonzero(front.any(axis=0))[-1]])
    H = height
    info: dict = {}

    def col(x):
        return int(np.clip(round((x - lo[0]) / v - 0.5), 0, nx - 1))

    def row(z):
        return int(np.clip(round((z - lo[2]) / v - 0.5), 0, nz - 1))

    # --- legs: two runs at the bottom, tracked upwards until they merge -------------
    start = row(0.08 * H)
    legs = [r for r in runs(front[:, start]) if r[1] - r[0] >= 2]
    if len(legs) < 2:
        raise RigError("no two separate legs near the ground")
    legs = sorted(legs, key=lambda r: (r[0] + r[1]) / 2)
    left, right = legs[-1], legs[0]  # +X is the character's left
    centres = {"L": [], "R": []}
    merge_z = None
    for iz in range(start, nz):
        rs = runs(front[:, iz])
        lr = [r for r in rs if r[0] <= left[1] + 1 and r[1] >= left[0] - 1]
        rr = [r for r in rs if r[0] <= right[1] + 1 and r[1] >= right[0] - 1]
        if not lr or not rr or lr[0] == rr[0]:
            merge_z = zs[iz]
            break
        left, right = lr[0], rr[0]
        for s, r in (("L", left), ("R", right)):
            centres[s].append((zs[iz], xs[r[0]] / 2 + xs[r[1]] / 2, (r[1] - r[0] + 1) * v))
    if merge_z is None:
        raise RigError("the legs never join")
    info["leg_merge"] = float(merge_z / H)
    # Thick thighs touch below the crotch and a skirt hides it, so the legs' merge point is
    # only a hint: hip joints of these figures sit at 47-56 % of the height.
    hip_z = float(np.clip(merge_z + 0.035 * H, 0.47 * H, 0.56 * H))
    ankle_z = 0.055 * H
    knee_z = ankle_z + 0.47 * (hip_z - ankle_z)

    def leg_x(side, z):
        pts = np.array(centres[side])
        widths = pts[:, 2]
        # a run that suddenly widens a lot is a skirt hem, not the leg
        ok = widths < 2.5 * np.median(widths[: max(3, len(widths) // 2)])
        pts = pts[ok] if ok.sum() >= 3 else pts
        return float(np.interp(z, pts[:, 0], pts[:, 1]))

    joints: dict = {}
    for s in SIDES:
        top = np.array(centres[s])[-1, 0]
        hx = leg_x(s, min(hip_z, top))
        kx, ax = leg_x(s, knee_z), leg_x(s, ankle_z)
        # toes: the forward-most surface of this foot
        ix0, ix1 = sorted((col(ax - 0.06 * H), col(ax + 0.06 * H)))
        foot = grid.shell[ix0 : ix1 + 1, :, iz0 : row(0.04 * H) + 1]
        ys = np.flatnonzero(foot.any(axis=(0, 2)))
        if ys.size == 0:
            raise RigError("no foot found")
        toe_y = lo[1] + (ys[0] + 0.5) * v
        heel_y = lo[1] + (ys[-1] + 0.5) * v
        ay = _depth_center(grid, ax, ankle_z, 0.2 * H)
        ay = 0.5 * (heel_y + toe_y) if ay is None else min(ay, heel_y - 0.02 * H)
        joints[f"hip.{s}"] = (hx, None, hip_z)
        joints[f"knee.{s}"] = (kx, None, knee_z)
        joints[f"ankle.{s}"] = (ax, ay, ankle_z)
        joints[f"ball.{s}"] = (ax, toe_y + 0.35 * (ay - toe_y), 0.02 * H)
        joints[f"toe_tip.{s}"] = (ax, toe_y, 0.02 * H)

    # --- arms, part 1: follow each arm from the hand tip inwards ---------------------
    occupied_cols = np.flatnonzero(front[:, row(0.3 * H) :].any(axis=1))
    arms = {}
    for s, sign in (("L", 1), ("R", -1)):
        tip_col = occupied_cols[-1] if sign > 0 else occupied_cols[0]
        tip_x = xs[tip_col]
        if abs(tip_x) < 0.15 * H:
            raise RigError("the arms are not away from the body")
        # One column at a time, following the span of the runs that touch the previous
        # column's span (spread fingers merge into one span).
        span_rows = []
        prev = None
        c = tip_col
        while 0 <= c < nx and abs(xs[c]) > 0.04 * H:
            rs = [r for r in runs(front[c, :]) if zs[r[1]] > 0.25 * H]
            if prev is not None:
                rs = [r for r in rs if r[0] <= prev[1] + 2 and r[1] >= prev[0] - 2]
            if not rs:
                break
            if prev is None:
                rs = [max(rs, key=lambda r: r[1])]
            prev = (min(r[0] for r in rs), max(r[1] for r in rs))
            span_rows.append((xs[c], prev))
            c -= sign
        # The arm ends where its span suddenly gets much taller: it joined the torso.
        dist = np.abs(np.array([x for x, _ in span_rows]) - tip_x)
        thick = np.array([(b - a + 1) * v for _, (a, b) in span_rows])
        ref_band = (dist > 0.07 * H) & (dist < 0.17 * H)
        if ref_band.sum() < 3:
            raise RigError("could not follow an arm")
        ref = float(np.median(thick[ref_band]))
        joined = np.flatnonzero((dist > 0.1 * H) & (thick > 1.9 * ref))
        end = int(joined[0]) if joined.size else len(span_rows)
        pts = np.array([(x, (zs[a] + zs[b]) / 2, (b - a + 1) * v) for x, (a, b) in span_rows[:end]])
        if len(pts) < 8:
            raise RigError("could not follow an arm")
        arms[s] = (pts, tip_x, sign)
        info[f"armpit_x.{s}"] = float(pts[-1, 0] / H)

    # --- neck: the most recessed point of the front profile between chest and face ---
    centre = np.abs(xs) < 0.025 * H
    side_view = solid[centre].any(axis=0)  # (Y, Z) through the middle of the body
    front_y = np.full(nz, np.inf)
    for iz in range(nz):
        hit = np.flatnonzero(side_view[:, iz])
        if hit.size:
            front_y[iz] = lo[1] + (hit[0] + 0.5) * v
    face_rows = np.flatnonzero((zs > 0.78 * H) & (zs < 0.97 * H) & np.isfinite(front_y))
    if face_rows.size == 0:
        raise RigError("no head found")
    face_row = int(face_rows[np.argmin(front_y[face_rows])])
    chest_rows = np.flatnonzero((zs > 0.55 * H) & (zs < zs[face_row] - 0.08 * H))
    chest_row = (
        int(chest_rows[np.argmin(front_y[chest_rows])]) if chest_rows.size else row(0.65 * H)
    )
    between = np.arange(chest_row, face_row + 1)
    neck_mid = float(zs[between[np.argmax(front_y[between])]])
    info["neck"] = neck_mid / H
    if not hip_z + 0.15 * H < neck_mid < 0.93 * H:
        raise RigError("neck and hips are not where a standing figure has them")
    neck_z = neck_mid - 0.03 * H
    head_z = neck_mid + 0.025 * H
    sh_z = neck_mid - 0.045 * H
    joints["pelvis"] = (0.0, None, hip_z + 0.01 * H)
    joints["waist"] = (0.0, None, hip_z + 0.35 * (neck_z - hip_z))
    joints["chest"] = (0.0, None, hip_z + 0.7 * (neck_z - hip_z))
    joints["neck"] = (0.0, None, neck_z)
    joints["head"] = (0.0, None, head_z)
    # Depth: collars, open jackets and hair make the middle of the body hard to read
    # higher up, so only the pelvis is measured; the head pivots behind the front of
    # the neck, and the spine runs straight between them.
    pelvis_y = _depth_center(grid, 0.0, joints["pelvis"][2], 0.13 * H, radius=2, min_run=0.035 * H)
    pelvis_y = 0.0 if pelvis_y is None else pelvis_y
    neck_front = front_y[row(neck_mid)]
    head_y = float(neck_front + 0.028 * H) if np.isfinite(neck_front) else pelvis_y
    z0, z1 = joints["pelvis"][2], head_z
    for name in ("pelvis", "waist", "chest", "neck", "head"):
        x, _, z = joints[name]
        joints[name] = (x, pelvis_y + (head_y - pelvis_y) * (z - z0) / (z1 - z0), z)
    joints["head_top"] = (0.0, head_y, max(ztop, head_z + 0.08 * H))
    spine_y = joints["neck"][1]

    # --- arms, part 2: shoulder above the armpit, wrist where hand meets sleeve -----
    rest_raise = {}
    for s, (pts, tip_x, sign) in arms.items():
        armpit_x = abs(pts[-1, 0])
        inset = armpit_x - 0.25 * np.median(pts[:, 2])
        sh_x = sign * float(np.clip(inset, 0.065 * H, 0.095 * H))
        slope = np.polyfit(pts[:, 0], pts[:, 1], 1)[0]
        cos = 1.0 / math.sqrt(1.0 + slope * slope)
        along = np.abs(pts[:, 0] - tip_x) / cos  # distance from the tip along the arm
        perp = pts[:, 2] * cos  # thickness across the arm
        hand = perp[(along > 0.015 * H) & (along < 0.04 * H)]
        ref = float(np.median(hand)) if hand.size else float(np.median(perp[:4]))
        jump = np.flatnonzero((along > 0.035 * H) & (perp > 1.5 * ref))
        hand_len = float(along[jump[0]]) - 0.005 * H if jump.size else 0.07 * H
        hand_len = float(np.clip(hand_len, 0.04 * H, 0.1 * H))
        info[f"hand_length.{s}"] = hand_len / H
        order = np.argsort(along)

        def on_arm(d, order=order, along=along, pts=pts):
            """(x, z) on the traced centre line, ``d`` from the tip."""
            return (
                float(np.interp(d, along[order], pts[order, 0])),
                float(np.interp(d, along[order], pts[order, 1])),
            )

        wx, wz = on_arm(hand_len)
        tx, tz = on_arm(0.01 * H)
        ex, ez = sh_x + 0.55 * (wx - sh_x), sh_z + 0.55 * (wz - sh_z)
        if abs(ex - tip_x) / cos <= along.max():  # still on the traced arm: its centre line
            ex, ez = on_arm(abs(ex - tip_x) / cos)
        for name, (x, z) in (("elbow", (ex, ez)), ("wrist", (wx, wz)), ("hand_tip", (tx, tz))):
            y = _depth_center(grid, x, z, 0.2 * H)
            joints[f"{name}.{s}"] = (x, spine_y if y is None else y, z)
        sh_y = 0.5 * (spine_y + joints[f"elbow.{s}"][1])
        joints[f"shoulder.{s}"] = (sh_x, sh_y, sh_z)
        joints[f"clavicle.{s}"] = (sign * 0.02 * H, spine_y - 0.015 * H, sh_z + 0.012 * H)
        rest_raise[s] = math.degrees(math.atan2(abs(wx - sh_x), sh_z - wz))

    # --- leg depth -------------------------------------------------------------------
    for s in SIDES:
        x, _, z = joints[f"hip.{s}"]
        joints[f"hip.{s}"] = (x, joints["pelvis"][1], z)  # a skirt in front is not the hip
        x, _, z = joints[f"knee.{s}"]
        y = _depth_center(grid, x, z, 0.2 * H)
        joints[f"knee.{s}"] = (x, joints["pelvis"][1] if y is None else y, z)
        # a hint of bend, so IK knows which way knees and elbows go
        x, y, z = joints[f"knee.{s}"]
        joints[f"knee.{s}"] = (x, y - 0.006 * H, z)
        x, y, z = joints[f"elbow.{s}"]
        joints[f"elbow.{s}"] = (x, y + 0.006 * H, z)
    out = {k: tuple(float(c) for c in p) for k, p in joints.items()}
    out["rest_raise.L"], out["rest_raise.R"] = rest_raise["L"], rest_raise["R"]
    out["info"] = info
    return out


# --------------------------------------------------------------------------------------
# Skin weights (numpy only): geodesic voxel binding
# --------------------------------------------------------------------------------------
def segment_voxels(grid: Grid, head, tail):
    """Indices of the voxels a bone segment passes through."""
    import numpy as np

    head, tail = np.asarray(head, float), np.asarray(tail, float)
    n = max(2, int(math.ceil(np.linalg.norm(tail - head) / (0.5 * grid.size))) + 1)
    t = np.linspace(0.0, 1.0, n)[:, None]
    return np.unique(grid.index(head + t * (tail - head)), axis=0)


def geodesic_distance(grid: Grid, seeds, max_steps: int):
    """Voxel steps from ``seeds`` through solid voxels (approximately Euclidean).

    Alternates 6- and 26-neighbour growth (an octagonal metric) so diagonals are not
    over- or under-counted much. Unreached voxels get ``max_steps + 1``.
    """
    import numpy as np

    solid = grid.solid
    dist = np.full(solid.shape, max_steps + 1, np.int32)
    reached = np.zeros_like(solid)
    seeds = np.asarray(seeds, int)
    reached[seeds[:, 0], seeds[:, 1], seeds[:, 2]] = True
    reached &= solid
    if not reached.any():  # a bone outside the model: start from the nearest solid voxels
        near = np.zeros_like(solid)
        near[seeds[:, 0], seeds[:, 1], seeds[:, 2]] = True
        for _ in range(max_steps):
            near = dilate26(near)
            if (near & solid).any():
                break
        reached = near & solid
    dist[reached] = 0
    for step in range(1, max_steps + 1):
        grown = (dilate26(reached) if step % 2 else dilate6(reached)) & solid & ~reached
        if not grown.any():
            break
        dist[grown] = step
        reached |= grown
    return dist


def skin_weights(
    grid: Grid,
    points,
    bones: dict,
    power: float = 4.0,
    max_influences: int = 4,
    reach: float = 0.6,
    height: float = 1.0,
):
    """Weights (N, B) for ``points`` from bone segments ``{name: (head, tail)}``.

    Each point takes the geodesic distance to every bone (through the solid grid), and
    w = 1 / d^power, keeping the ``max_influences`` largest, normalised. Points no bone
    reaches within ``reach`` model heights fall back to straight-line distance.
    """
    import numpy as np

    points = np.asarray(points, float)
    names = list(bones)
    steps = int(reach * height / grid.size)
    idx = grid.index(points)
    d = np.empty((len(points), len(names)))
    for j, name in enumerate(names):
        head, tail = bones[name]
        field_ = geodesic_distance(grid, segment_voxels(grid, head, tail), steps)
        d[:, j] = field_[idx[:, 0], idx[:, 1], idx[:, 2]]
    far = d.min(axis=1) > steps
    if far.any():
        for j, name in enumerate(names):
            head, tail = (np.asarray(p, float) for p in bones[name])
            d[far, j] = point_segment_distance(points[far], head, tail) / grid.size
    d = (d + 0.5) * grid.size  # voxel steps -> distance; +0.5 keeps 1/d finite
    w = d ** (-power)
    if max_influences < len(names):
        cut = np.sort(w, axis=1)[:, -max_influences][:, None]
        w = np.where(w >= cut, w, 0.0)
    return w / w.sum(axis=1, keepdims=True), names


def point_segment_distance(points, a, b):
    import numpy as np

    ab = b - a
    t = np.clip(((points - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
    return np.linalg.norm(points - (a + t[:, None] * ab), axis=1)


def weld(co, tol: float):
    """Index of a shared vertex for every vertex: glTF splits vertices along UV seams."""
    import numpy as np

    q = np.round(np.asarray(co, dtype=np.float64) / tol).astype(np.int64)
    _, inverse = np.unique(q, axis=0, return_inverse=True)
    return inverse.reshape(-1)


def smooth_weights(weights, edges, iterations: int, welded=None):
    """Average every vertex's weights with its neighbours ``iterations`` times.

    ``welded`` (from ``weld``) makes the split copies of a seam vertex one vertex, so
    both sides of a seam keep exactly the same weights (or the seam would open up).
    """
    import numpy as np

    w = np.asarray(weights, dtype=np.float64)
    e = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if welded is not None:
        m = int(welded.max()) + 1
        w = np.stack(
            [np.bincount(welded, weights=w[:, j], minlength=m) for j in range(w.shape[1])], 1
        )
        w /= np.bincount(welded, minlength=m)[:, None]
        e = np.sort(welded[e], axis=1)
        e = np.unique(e[e[:, 0] != e[:, 1]], axis=0)
    n = len(w)
    src = np.concatenate([e[:, 0], e[:, 1]])
    dst = np.concatenate([e[:, 1], e[:, 0]])
    degree = np.bincount(src, minlength=n).astype(np.float64)[:, None] + 1.0
    for _ in range(iterations):
        sums = [np.bincount(src, weights=w[dst, j], minlength=n) for j in range(w.shape[1])]
        w = (w + np.stack(sums, axis=1)) / degree
    if welded is not None:
        w = w[welded]
    return w / np.maximum(w.sum(axis=1, keepdims=True), 1e-12)


def local_move(rest_rotation, move, height: float):
    """Pose-bone location for a move [left, forward, up] in model heights.

    ``rest_rotation``: the bone's rest orientation (3x3, armature space). The character
    faces -Y, so forward is -Y and the character's left is +X.
    """
    import numpy as np

    world = np.array([move[0], -move[1], move[2]], dtype=np.float64) * height
    return np.asarray(rest_rotation, dtype=np.float64).T @ world


# --------------------------------------------------------------------------------------
# Blender side
# --------------------------------------------------------------------------------------
RIG_NAME = "ToonRig"
# Bone colour palettes: red = left, blue = right, yellow = middle, purple = IK controls.
COLORS = {"L": "THEME01", "R": "THEME04", "": "THEME09", "ik": "THEME06", "root": "THEME10"}


def _ring(radius: float, y: float = 0.0, segments: int = 24):
    tau = 2 * math.pi
    pts = [
        (radius * math.cos(tau * i / segments), y, radius * math.sin(tau * i / segments))
        for i in range(segments)
    ]
    return pts, [(i, (i + 1) % segments) for i in range(segments)]


def _box(x0, x1, y0, y1, z0, z1):
    pts = [(x, y, z) for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)]
    edges = [(a, b) for a in range(8) for b in range(a + 1, 8) if bin(a ^ b).count("1") == 1]
    return pts, edges


def _diamond(r: float):
    pts = [(r, 0, 0), (-r, 0, 0), (0, r, 0), (0, -r, 0), (0, 0, r), (0, 0, -r)]
    return pts, [(a, b) for a in range(6) for b in range(a + 1, 6) if a // 2 != b // 2]


RINGS = {  # radius (model heights) and position along the bone (0 = head, 1 = tail)
    "torso": (0.17, 0.0),
    "hips": (0.12, 0.3),
    "spine": (0.11, 0.5),
    "chest": (0.12, 0.5),
    "neck": (0.04, 0.5),
    "head": (0.1, 0.45),
    "shoulder": (0.03, 0.85),
    "upper_arm": (0.04, 0.5),
    "forearm": (0.035, 0.5),
    "hand": (0.035, 0.5),
    "thigh": (0.06, 0.35),
    "shin": (0.045, 0.5),
    "foot": (0.04, 0.5),
    "toe": (0.025, 0.5),
}


def bone_shape(name: str, length: float, H: float):
    """Custom shape (points, edges) for a bone, in bone space and world units."""
    base, _ = bone_kind(name)
    if base == "root":
        pts, edges = _ring(0.28 * H)
        n = len(pts)
        arrow = [(0, 0.28), (0, 0.36), (-0.04, 0.31), (0.04, 0.31)]  # points forward
        pts += [(x * H, 0, z * H) for x, z in arrow]
        return pts, edges + [(n, n + 1), (n + 1, n + 2), (n + 1, n + 3)]
    if base == "foot_ik":
        return _box(-0.04 * H, 0.04 * H, -0.04 * H, length + 0.02 * H, -0.06 * H, 0.015 * H)
    if base == "hand_ik":
        return _box(-0.04 * H, 0.04 * H, 0.0, length, -0.04 * H, 0.04 * H)
    if base in ("knee_pole", "elbow_pole"):
        return _diamond(0.02 * H)
    r, f = RINGS[base]
    return _ring(r * H, f * length)


def build_rig(bpy, joints: dict, height: float, name: str = RIG_NAME):
    """Armature object for ``joints`` (from find_joints), in the scene, in rest pose.

    Deform bones follow DEFORM_BONES. Controls: ``root`` (whole character, on the
    floor), ``torso`` (hips and everything above; IK feet stay planted), ``foot_ik.*``
    and ``knee_pole.*`` (legs, IK on), ``hand_ik.*`` and ``elbow_pole.*`` (arms, IK
    off). The object properties ``ik_leg.L`` ... ``ik_arm.R`` blend IK (1) and FK (0).
    """
    from mathutils import Vector

    H = float(height)
    J = {k: Vector(v) for k, v in joints.items() if isinstance(v, tuple) and len(v) == 3}
    arm = bpy.data.armatures.new(name)
    arm.display_type = "OCTAHEDRAL"
    rig = bpy.data.objects.new(name, arm)
    bpy.context.scene.collection.objects.link(rig)
    rig.show_in_front = True
    view_layer = bpy.context.view_layer
    for o in view_layer.objects:
        o.select_set(False)
    view_layer.objects.active = rig
    rig.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    eb = arm.edit_bones
    forward, up = Vector((0.0, -1.0, 0.0)), Vector((0.0, 0.0, 1.0))

    def bone(bname, head, tail, parent=None, roll_to=forward, deform=False, connect=False):
        b = eb.new(bname)
        b.head, b.tail = head, tail
        b.align_roll(roll_to)
        if parent:
            b.parent = eb[parent]
            b.use_connect = connect
        b.use_deform = deform
        return b

    bone("root", Vector((0, 0, 0)), Vector((0, 0, 0.12 * H)))
    bone("torso", J["pelvis"], J["pelvis"] + Vector((0, 0, 0.08 * H)), "root")
    for bname, (parent, h, t) in DEFORM_BONES.items():
        par = parent or "torso"
        connect = parent is not None and (J[h] - eb[parent].tail).length < 1e-6
        roll_to = up if bname in FLAT_BONES else forward
        bone(bname, J[h], J[t], par, roll_to, deform=True, connect=connect)
    poles = {}
    for s in SIDES:
        for ik, src in ((f"foot_ik.{s}", f"foot.{s}"), (f"hand_ik.{s}", f"hand.{s}")):
            b = bone(ik, eb[src].head.copy(), eb[src].tail.copy(), "root")
            b.roll = eb[src].roll  # same rest orientation: Copy Rotation is identity at rest
        lift = Vector((0, 0, 0.04 * H))
        knee = J[f"knee.{s}"] + Vector((0, -0.3 * H, 0))
        elbow = J[f"elbow.{s}"] + Vector((0, 0.3 * H, 0))
        bone(f"knee_pole.{s}", knee, knee + lift, "root")
        bone(f"elbow_pole.{s}", elbow, elbow + lift, "root")
        for chain, upper, lower, pole in (
            ("leg", f"thigh.{s}", f"shin.{s}", f"knee_pole.{s}"),
            ("arm", f"upper_arm.{s}", f"forearm.{s}", f"elbow_pole.{s}"),
        ):
            a, b = eb[upper], eb[lower]
            poles[(chain, s)] = pole_angle(a.head, a.tail, a.x_axis, b.tail, eb[pole].head)
    rest_raise = {}
    for s in SIDES:
        d = eb[f"upper_arm.{s}"].tail - eb[f"upper_arm.{s}"].head
        rest_raise[s] = math.degrees(math.atan2(abs(d.x), -d.z))
    lengths = {b.name: b.length for b in eb}
    bpy.ops.object.mode_set(mode="POSE")

    # --- switches and constraints ------------------------------------------------------
    for key, value in SWITCHES.items():
        rig[key] = value
        what = "legs" if key.startswith("ik_leg") else "arms"
        rig.id_properties_ui(key).update(
            min=0.0,
            max=1.0,
            soft_min=0.0,
            soft_max=1.0,
            description=f"1 = IK (the {what} follow the *_ik controls), 0 = FK (rotate the bones)",
        )
    for s in SIDES:
        rig[f"rest_raise.{s}"] = rest_raise[s]
    rig["toon_height"] = H
    rig["toon_rig_version"] = 1

    def drive(con, prop):
        drv = con.driver_add("influence").driver
        drv.type = "AVERAGE"  # no Python expression: works without auto-run scripts
        var = drv.variables.new()
        var.name = "ik"
        var.type = "SINGLE_PROP"
        var.targets[0].id_type = "OBJECT"
        var.targets[0].id = rig
        var.targets[0].data_path = f'["{prop}"]'

    pb = rig.pose.bones
    for s in SIDES:
        for chain, lower, end, target, pole in (
            ("leg", f"shin.{s}", f"foot.{s}", f"foot_ik.{s}", f"knee_pole.{s}"),
            ("arm", f"forearm.{s}", f"hand.{s}", f"hand_ik.{s}", f"elbow_pole.{s}"),
        ):
            ik = pb[lower].constraints.new("IK")
            ik.name = "IK"
            ik.target, ik.subtarget = rig, target
            ik.pole_target, ik.pole_subtarget = rig, pole
            ik.pole_angle = poles[(chain, s)]
            ik.chain_count = 2
            ik.use_stretch = False
            drive(ik, f"ik_{chain}.{s}")
            cr = pb[end].constraints.new("COPY_ROTATION")
            cr.name = "IK rotation"
            cr.target, cr.subtarget = rig, target
            drive(cr, f"ik_{chain}.{s}")

    # --- rotation mode, locks, shapes, colours, collections -----------------------------
    shapes = bpy.data.collections.new(f"{name}Shapes")  # not linked to the scene: hidden
    colls = {c: arm.collections.new(c) for c in ("Body", "IK", "Root")}
    for p in pb:
        base, side = bone_kind(p.name)
        p.rotation_mode = "YXZ"
        p.lock_scale = (True, True, True)
        if base not in MOVABLE:
            p.lock_location = (True, True, True)
        kind = "root" if base in ("root", "torso") else "ik" if base in MOVABLE else side
        p.color.palette = COLORS[kind]
        p.bone.color.palette = COLORS[kind]
        coll = "Root" if kind == "root" else "IK" if kind == "ik" else "Body"
        colls[coll].assign(p.bone)
        pts, edges = bone_shape(p.name, lengths[p.name], H)
        me = bpy.data.meshes.new(f"WGT-{p.name}")
        me.from_pydata(pts, edges, [])
        obj = bpy.data.objects.new(f"WGT-{p.name}", me)
        shapes.objects.link(obj)
        p.custom_shape = obj
        p.use_custom_shape_bone_size = False
    bpy.ops.object.mode_set(mode="OBJECT")
    return rig


def skin(bpy, rig, meshes, height: float, voxel: float | None = None, smooth: int = 6) -> dict:
    """Skin ``meshes`` to ``rig`` with geodesic voxel weights.

    Adds a vertex group per deform bone and an Armature modifier (call this before other
    modifiers are added, so the armature runs first), and parents the meshes to the rig
    without moving them. Returns ``{object name: (weights (N, B), bone names)}``.
    """
    import time

    import numpy as np

    t0 = time.time()
    H = float(height)
    parts, tris, offset = [], [], 0
    for o in meshes:
        me = o.data
        n = len(me.vertices)
        co = np.empty(n * 3, np.float32)
        me.vertices.foreach_get("co", co)
        m = np.array(o.matrix_world, dtype=np.float64)
        wco = co.reshape(n, 3) @ m[:3, :3].T + m[:3, 3]
        me.calc_loop_triangles()
        t = np.empty(len(me.loop_triangles) * 3, np.int64)
        me.loop_triangles.foreach_get("vertices", t)
        e = np.empty(len(me.edges) * 2, np.int64)
        me.edges.foreach_get("vertices", e)
        parts.append((o, wco, e.reshape(-1, 2)))
        tris.append(t.reshape(-1, 3) + offset)
        offset += n
    all_co = np.concatenate([p[1] for p in parts])
    grid = voxelize(all_co, np.concatenate(tris), voxel or H / 128)
    rm = np.array(rig.matrix_world, dtype=np.float64)
    bones = {}
    for b in rig.data.bones:
        if b.use_deform:
            head = rm[:3, :3] @ np.array(b.head_local) + rm[:3, 3]
            tail = rm[:3, :3] @ np.array(b.tail_local) + rm[:3, 3]
            bones[b.name] = (head, tail)
    result = {}
    for o, wco, edges in parts:
        w, names = skin_weights(grid, wco, bones, height=H)
        if smooth:
            w = smooth_weights(w, edges, smooth, weld(wco, 1e-5 * H))
        result[o.name] = (w, names)
        q = np.round(w * 255).astype(np.int16)
        for j, bname in enumerate(names):
            vg = o.vertex_groups.get(bname) or o.vertex_groups.new(name=bname)
            column = q[:, j]
            for level in np.unique(column[column > 0]):
                vg.add(np.flatnonzero(column == level).tolist(), float(level) / 255.0, "REPLACE")
        mod = o.modifiers.new("Armature", "ARMATURE")
        mod.object = rig
        mw = o.matrix_world.copy()
        o.parent = rig
        o.matrix_world = mw
    print(f"[rig] skinned {len(parts)} mesh(es) in {time.time() - t0:.1f}s", flush=True)
    return result


def part_scores(weights, bone_names):
    """(N, P) score of every vertex for every part in PARTS: its summed bone weights."""
    import numpy as np

    w = np.asarray(weights, dtype=np.float64)
    index = {b: i for i, b in enumerate(bone_names)}
    cols = []
    for bones in PARTS.values():
        idx = [index[b] for b in bones if b in index]
        cols.append(w[:, idx].sum(axis=1) if idx else np.zeros(len(w)))
    return np.stack(cols, axis=1)


def face_parts(scores, loop_vertices, loop_starts):
    """Part index (into PARTS) of every face: the part its corners score highest on.

    ``loop_vertices``: vertex index of every face corner; ``loop_starts``: index of each
    face's first corner (faces may have any number of corners).
    """
    import numpy as np

    scores = np.asarray(scores, dtype=np.float64)
    loop_vertices = np.asarray(loop_vertices, dtype=np.int64)
    starts = np.asarray(loop_starts, dtype=np.int64)
    face_of_loop = np.repeat(np.arange(len(starts)), np.diff(np.append(starts, len(loop_vertices))))
    totals = np.stack(
        [
            np.bincount(face_of_loop, weights=scores[loop_vertices, p], minlength=len(starts))
            for p in range(scores.shape[1])
        ],
        axis=1,
    )
    return totals.argmax(axis=1)


def split_parts(bpy, obj, weights, bone_names) -> list:
    """Separate ``obj`` into one object per body part (PARTS); returns the new objects.

    Faces go to the part their vertices are weighted to most. Vertices on a cut are
    kept in both parts with the same weights, so the cut never opens when posed.
    """
    import numpy as np

    me = obj.data
    loop_v = np.empty(len(me.loops), np.int64)
    me.loops.foreach_get("vertex_index", loop_v)
    starts = np.empty(len(me.polygons), np.int64)
    me.polygons.foreach_get("loop_start", starts)
    labels = face_parts(part_scores(weights, bone_names), loop_v, starts)
    attr = me.attributes.get("toon_part") or me.attributes.new("toon_part", "INT", "FACE")
    attr.data.foreach_set("value", labels.astype(np.int32))
    names = list(PARTS)
    sizes = np.bincount(labels, minlength=len(names))
    order = [int(p) for p in np.argsort(-sizes) if sizes[p] > 0]
    view_layer = bpy.context.view_layer
    for o in view_layer.objects:
        o.select_set(False)
    view_layer.objects.active = obj
    obj.select_set(True)
    parts = []
    base_name = obj.name
    for p in order[1:]:  # the biggest part stays in obj
        me = obj.data
        loop_v = np.empty(len(me.loops), np.int64)
        me.loops.foreach_get("vertex_index", loop_v)
        loop_e = np.empty(len(me.loops), np.int64)
        me.loops.foreach_get("edge_index", loop_e)
        starts = np.empty(len(me.polygons), np.int64)
        me.polygons.foreach_get("loop_start", starts)
        counts = np.diff(np.append(starts, len(loop_v)))
        part = np.empty(len(me.polygons), np.int32)
        me.attributes["toon_part"].data.foreach_get("value", part)
        faces = part == p
        corners = np.repeat(faces, counts)
        verts = np.zeros(len(me.vertices), bool)
        verts[loop_v[corners]] = True
        edges = np.zeros(len(me.edges), bool)
        edges[loop_e[corners]] = True
        me.vertices.foreach_set("select", verts)
        me.edges.foreach_set("select", edges)
        me.polygons.foreach_set("select", faces)
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.separate(type="SELECTED")
        bpy.ops.object.mode_set(mode="OBJECT")
        new = next(o for o in view_layer.objects if o.select_get() and o is not obj)
        new.name = new.data.name = names[p]
        new.select_set(False)
        parts.append(new)
    obj.name = obj.data.name = names[order[0]]
    parts.insert(0, obj)
    print(f"[rig] split {base_name} into {len(parts)} parts", flush=True)
    return parts


def rest_rotation(pose_bone):
    import numpy as np

    return np.array(pose_bone.bone.matrix_local.to_3x3(), dtype=np.float64)


def apply_pose(rig, pose: dict, frame: int | None = None, reset: bool = True) -> None:
    """Pose ``rig`` from pose words (see POSE_CHANNELS); key every channel at ``frame``.

    ``reset``: start from the rest pose and the default IK/FK switches, so the result
    does not depend on the pose before (a key in an animation is always a full pose).
    """
    H = float(rig.get("toon_height", 1.0))
    validate_pose(pose)
    if reset:
        for p in rig.pose.bones:
            p.rotation_euler = (0.0, 0.0, 0.0)
            p.location = (0.0, 0.0, 0.0)
        for key, value in SWITCHES.items():
            rig[key] = value
    for bname, values in pose.items():
        if bname in SWITCHES:
            rig[bname] = float(values)
            continue
        p = rig.pose.bones.get(bname)
        if p is None:
            raise ValueError(f"the rig has no bone {bname!r}")
        side = bone_kind(bname)[1]
        p.rotation_euler = semantic_euler(bname, values, float(rig.get(f"rest_raise.{side}", 0)))
        if "move" in values:
            p.location = tuple(local_move(rest_rotation(p), values["move"], H))
    if frame is not None:
        for p in rig.pose.bones:
            p.keyframe_insert("rotation_euler", frame=frame, group=p.name)
            if bone_kind(p.name)[0] in MOVABLE:
                p.keyframe_insert("location", frame=frame, group=p.name)
        for key in SWITCHES:
            rig.keyframe_insert(f'["{key}"]', frame=frame, group="IK switches")


def _fcurves(action) -> list:
    """F-curves of an action, for the legacy API and Blender 4.4+ layered actions."""
    layers = list(getattr(action, "layers", []))
    if not layers:
        return list(getattr(action, "fcurves", []))
    out = []
    for layer in layers:
        for strip in layer.strips:
            for bag in getattr(strip, "channelbags", []):
                out.extend(bag.fcurves)
    return out


def key_animation(bpy, rig, anim: Animation):
    """Key ``anim`` on ``rig`` as a new action (kept with a fake user); returns it."""
    action = bpy.data.actions.new(anim.name)
    action.use_fake_user = True
    rig.animation_data_create()
    rig.animation_data.action = action
    for frame, pose_ in anim.keys:
        apply_pose(rig, pose_, frame=frame)
    interp = {"smooth": "BEZIER", "linear": "LINEAR", "constant": "CONSTANT"}[anim.interpolation]
    for fc in _fcurves(action):
        for kp in fc.keyframe_points:
            kp.interpolation = interp
            kp.handle_left_type = kp.handle_right_type = "AUTO_CLAMPED"
        fc.update()
    action["toon_frames"] = anim.frames
    action["toon_hold"] = anim.hold
    action["toon_loop"] = anim.loop
    if anim.description:
        action["toon_description"] = anim.description
    return action


def library_bundle() -> dict:
    """Poses and animations together, as embedded in the .blend."""
    anims = {}
    if ANIMATIONS_DIR.is_dir():
        for p in sorted(ANIMATIONS_DIR.glob("*.json")):
            anims[p.stem] = json.loads(p.read_text("utf-8"))
    return {"poses": load_library(), "animations": anims}


def embed(bpy) -> None:
    """Store this module and the pose/animation library in the .blend as text blocks.

    Blender imports text blocks by name, so in the saved file (and through blender-mcp)
    ``import toon_rig; toon_rig.pose("cheer")`` works without this repository.
    """
    for text_name, content in (
        ("toon_rig.py", Path(__file__).read_text("utf-8")),
        ("toon_library.json", json.dumps(library_bundle(), indent=1)),
    ):
        txt = bpy.data.texts.get(text_name) or bpy.data.texts.new(text_name)
        txt.from_string(content)
    # "Register": run on load when the file is trusted (Preferences > Auto Run Scripts)
    bpy.data.texts["toon_rig.py"].use_module = True


def _embedded_bundle() -> dict | None:
    try:
        import bpy
    except ImportError:
        return None
    txt = bpy.data.texts.get("toon_library.json")
    return json.loads(txt.as_string()) if txt else None


# --- for interactive use inside Blender (UI Python console, or blender-mcp) ------------
def get_rig(name: str = RIG_NAME):
    import bpy

    rig = bpy.data.objects.get(name)
    if rig is None:
        raise ValueError(f"no object {name!r}; is this a 3D-Maker toon.blend with a rig?")
    return rig


def pose(spec, frame: int | None = None, reset: bool = True, rig_name: str = RIG_NAME) -> dict:
    """Pose the rig in the open file and return the resolved pose. Examples:
    ``pose({"head": {"turn": 20}})``, ``pose("cheer")``,
    ``pose(["idle", {"head": {"turn": 20}}], frame=12)`` (also keys it on frame 12)."""
    rig = get_rig(rig_name)
    resolved = resolve_pose(spec)
    apply_pose(rig, resolved, frame=frame, reset=reset)
    return resolved


def animate(spec, rig_name: str = RIG_NAME):
    """Key an animation (a library name, a JSON path or a dict) on the rig in the open
    file, make it the active action and set the frame range. Returns the action."""
    import bpy

    rig = get_rig(rig_name)
    if isinstance(spec, dict):
        anim = parse_animation(spec)
    else:
        bundle = _embedded_bundle()
        if bundle and spec in bundle["animations"]:
            anim = parse_animation({"name": spec, **bundle["animations"][spec]})
        else:
            anim = load_animation(spec)
    action = key_animation(bpy, rig, anim)
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 1, anim.frames
    scene.frame_set(1)
    return action


if globals().get("__name__") not in ("rig", "maker3d.blender.rig"):
    # Run inside Blender from the text block (Text Editor > Run Script, on load when the
    # file is trusted, or exec(bpy.data.texts["toon_rig.py"].as_string()) from Python or
    # blender-mcp): register this code as the module "toon_rig", so that afterwards
    # ``import toon_rig; toon_rig.pose("cheer")`` works anywhere in this Blender.
    import sys as _sys
    import types as _types

    _module = _types.ModuleType("toon_rig")
    _module.__dict__.update({k: v for k, v in globals().items() if not k.startswith("_")})
    _sys.modules["toon_rig"] = _module
