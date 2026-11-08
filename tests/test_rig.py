"""Pure-Python parts of blender/rig.py: pose language, animations, joints, skin weights."""

import math

import pytest

from maker3d.blender import character as C
from maker3d.blender import rig as R

np = pytest.importorskip("numpy")


# --- pose language --------------------------------------------------------------------
def test_semantic_euler_mirrors_the_sides():
    words = {"raise": 90, "forward": 30, "twist": 10}
    left = R.semantic_euler("upper_arm.L", words, rest_raise=45)
    right = R.semantic_euler("upper_arm.R", words, rest_raise=45)
    assert left[0] == right[0] == pytest.approx(math.radians(30))  # forward: same on both
    # raise is absolute (0 = down): 90 from a 45 degree rest is +45, mirrored on the right
    assert left[2] == pytest.approx(math.radians(45)) and right[2] == pytest.approx(-left[2])
    assert left[1] == pytest.approx(-right[1])
    assert R.semantic_euler("shin.L", {"bend": 30})[0] < 0  # knees bend backwards
    assert R.semantic_euler("forearm.R", {"bend": 30})[0] > 0  # elbows bend forwards
    assert R.semantic_euler("head", {"lean": 10})[2] < 0  # lean = to the character's left
    assert R.semantic_euler("foot_ik.L", {"point": 20})[0] < 0  # toes down
    with pytest.raises(ValueError):
        R.semantic_euler("head", {"raise": 10})
    with pytest.raises(ValueError):
        R.semantic_euler("tail", {})


def test_mirror_pose_round_trip():
    pose = {
        "upper_arm.L": {"raise": 120},
        "foot_ik.R": {"move": [0.1, 0.2, 0.0]},
        "head": {"turn": 20, "bend": 5},
        "ik_arm.L": 1,
    }
    mirrored = R.mirror_pose(pose)
    assert mirrored["upper_arm.R"] == {"raise": 120}
    assert mirrored["foot_ik.L"]["move"] == [-0.1, 0.2, 0.0]
    assert mirrored["head"] == {"turn": -20, "bend": 5}
    assert mirrored["ik_arm.R"] == 1
    assert R.mirror_pose(mirrored) == pose


def test_resolve_pose_names_lists_and_mirrors():
    lib = {"a": {"head": {"turn": 10}}, "b": ["a", {"head": {"bend": 5}}]}
    assert R.resolve_pose("b", lib) == {"head": {"turn": 10, "bend": 5}}
    assert R.resolve_pose(["a", {"head": {"turn": 30}}], lib) == {"head": {"turn": 30}}
    assert R.resolve_pose("mirror:a", lib) == {"head": {"turn": -10}}
    with pytest.raises(ValueError):
        R.resolve_pose("nope", lib)
    with pytest.raises(ValueError):
        R.resolve_pose(3, lib)


def test_validate_pose():
    R.validate_pose({"torso": {"move": [0, 0, -0.1]}, "ik_leg.L": 0})
    with pytest.raises(ValueError):
        R.validate_pose({"head": {"move": [0, 0, 1]}})  # heads do not move on their own
    with pytest.raises(ValueError):
        R.validate_pose({"ik_arm.L": 2})
    with pytest.raises(ValueError):
        R.validate_pose({"torso": {"move": [0, 1]}})


def test_builtin_poses_and_animations_are_valid():
    lib = R.load_library()
    assert {"rest", "t_pose", "idle", "cheer", "wave_out", "jump"} <= set(lib)
    for name in lib:
        R.validate_pose(R.resolve_pose(name, lib))
    names = R.animation_names()
    assert {"idle", "wave", "walk", "jump", "cheer", "dance"} <= set(names)
    for name in names:
        anim = R.load_animation(name, lib)
        assert anim.name == name and anim.keys[0][0] == 1
        assert all(1 <= f <= anim.frames + 1 for f, _ in anim.keys)
    bundle = R.library_bundle()
    assert set(bundle["animations"]) == set(names) and bundle["poses"] == lib


def test_parse_animation():
    data = {
        "frames": 24,
        "loop": True,
        "keys": [{"frame": 13, "pose": {"head": {"turn": 10}}}, {"frame": 1, "pose": {}}],
    }
    anim = R.parse_animation(data, {})
    assert [f for f, _ in anim.keys] == [1, 13, 25]  # sorted, and the loop closes on 25
    assert anim.keys[-1][1] == {} and anim.hold == 2 and anim.interpolation == "smooth"
    bad = [
        {"frames": 24, "keys": []},
        {"frames": 24, "keys": [{"frame": 30}]},
        {"frames": 24, "interpolation": "cubic", "keys": [{"frame": 1}]},
        {"frames": 24, "keys": [{"frame": 1}, {"frame": 1}]},
        {"frames": 24, "hold": 5, "keys": [{"frame": 1}]},
        {"frames": 1, "keys": [{"frame": 1}]},
    ]
    for data in bad:
        with pytest.raises(ValueError):
            R.parse_animation(data, {})
    with pytest.raises(ValueError):
        R.load_animation("no_such_animation")


def test_pole_angle():
    head, tail, x_axis = (0, 0, 1.0), (0, 0, 0.5), (1.0, 0, 0)
    ik_tail = (0, -0.01, 0.0)
    assert R.pole_angle(head, tail, x_axis, ik_tail, (1.0, 0, 0.5)) == pytest.approx(0, abs=0.02)
    side = R.pole_angle(head, tail, x_axis, ik_tail, (0, -1.0, 0.5))
    other = R.pole_angle(head, tail, x_axis, ik_tail, (0, 1.0, 0.5))
    assert abs(side) == pytest.approx(math.pi / 2, abs=0.03)
    assert abs(side - other) == pytest.approx(math.pi, abs=0.06)


def test_local_move():
    assert R.local_move(np.eye(3), [0.1, 0.2, 0.3], 2.0) == pytest.approx([0.2, -0.4, 0.6])
    turned = np.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])  # rest rotated 90 deg about Z
    # the bone's local -Y points along world +X (the character's left)
    assert R.local_move(turned, [1, 0, 0], 1.0) == pytest.approx([0, -1, 0])


# --- voxels, joints and weights -------------------------------------------------------
def _box(lo, hi):
    co = np.array(
        [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
    )
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    tris = [(a, b, c) for a, b, c, d in quads] + [(a, c, d) for a, b, c, d in quads]
    return co.astype(float), np.array(tris)


def test_voxelize_fills_the_inside():
    co, tris = _box((0, 0, 0), (1, 0.4, 0.4))
    grid = R.voxelize(co, tris, 0.05)
    mid = grid.index([[0.5, 0.2, 0.2]])[0]
    assert grid.solid[tuple(mid)] and not grid.shell[tuple(mid)]
    assert not grid.solid[0, 0, 0]  # padding outside
    assert len(R.surface_points(co, tris, 0.05)) > 100  # big triangles get sampled
    assert R.runs(np.array([0, 1, 1, 0, 1], bool)) == [(1, 2), (4, 4)]
    assert R.first_run(np.array([0, 0, 1, 1, 0], bool)) == (2, 3)


def test_skin_weights_follow_the_nearest_bone_through_the_inside():
    co, tris = _box((0, 0, 0), (0.2, 0.2, 1.0))  # an upright bar
    grid = R.voxelize(co, tris, 0.025)
    bones = {
        "lower": ((0.1, 0.1, 0.0), (0.1, 0.1, 0.5)),
        "upper": ((0.1, 0.1, 0.5), (0.1, 0.1, 1.0)),
    }
    pts = np.array([[0.0, 0.1, 0.1], [0.0, 0.1, 0.9], [0.0, 0.1, 0.5]])
    w, names = R.skin_weights(grid, pts, bones, height=1.0)
    assert names == ["lower", "upper"]
    assert w.sum(axis=1) == pytest.approx(1.0)
    assert w[0, 0] > 0.95 and w[1, 1] > 0.95 and 0.3 < w[2, 0] < 0.7


def test_smooth_weights_keep_seam_copies_equal():
    # path 0-1-2 plus vertex 3, a seam copy of 1 (same place, no edge to 1) linked to 4
    co = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [1, 0, 0], [1, 1, 0.0]])
    edges = np.array([[0, 1], [1, 2], [3, 4]])
    w = np.array([[1.0, 0], [1.0, 0], [1.0, 0], [0.0, 1], [0.0, 1]])
    out = R.smooth_weights(w, edges, 3, R.weld(co, 1e-6))
    assert out[1] == pytest.approx(out[3]) and out.sum(axis=1) == pytest.approx(1.0)


def test_parts_from_weights():
    names = ["head", "neck", "upper_arm.L"]
    w = np.array([[0.9, 0.1, 0.0], [0.0, 0.2, 0.8], [0.3, 0.3, 0.4]])
    scores = R.part_scores(w, names)
    parts = list(R.PARTS)
    assert scores[0, parts.index("Head")] == pytest.approx(1.0)
    assert scores[2, parts.index("Head")] == pytest.approx(0.6)  # head + neck beat the arm
    labels = R.face_parts(scores, [0, 1, 2, 1, 2, 2], [0, 3])  # faces (0,1,2) and (1,2,2)
    assert [parts[i] for i in labels] == ["Head", "UpperArm.L"]


def test_find_joints_on_a_figure_with_known_joints():
    """The from-scratch character (character.py) is an A-pose figure whose joints we know."""
    parts = C.build_parts(C.CharacterSpec(height=1.0))
    verts, tris, offset = [], [], 0
    for part in parts.values():
        verts.append(part.verts)
        tris += [
            (f[0] + offset, f[k] + offset, f[k + 1] + offset)
            for f in part.faces
            for k in range(1, len(f) - 1)
        ]
        offset += len(part.verts)
    co = np.concatenate(verts)
    height = float(co[:, 2].max())
    joints = R.find_joints(R.voxelize(co, np.array(tris), height / 160), height)
    truth = C.template_joints()
    for name, tol in (
        ("shoulder", 0.03), ("elbow", 0.04), ("wrist", 0.04), ("hip", 0.04), ("knee", 0.05),
    ):  # fmt: skip
        for s in R.SIDES:
            err = np.linalg.norm(np.array(joints[f"{name}.{s}"]) - truth[f"{name}.{s}"])
            assert err < tol * height, (name, s, err)
    for name in ("neck", "head"):
        assert abs(joints[name][2] - truth[name][2]) < 0.04 * height, name
    assert 30 < joints["rest_raise.L"] < 60 and 30 < joints["rest_raise.R"] < 60
    with pytest.raises(R.RigError):  # a box has no legs
        co, tris = _box((0, 0, 0), (0.4, 0.2, 1.0))
        R.find_joints(R.voxelize(co, tris, 0.01), 1.0)
