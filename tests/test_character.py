"""Geometry and face painting of the from-scratch character (blender/character.py)."""

import pytest

from maker3d.blender import character as C
from maker3d.blender import rig as R

np = pytest.importorskip("numpy")


def _euler_characteristic(verts, faces):
    edges = {tuple(sorted((f[i], f[(i + 1) % len(f)]))) for f in faces for i in range(len(f))}
    return len(verts) - len(edges) + len(faces)


def _signed_volume(verts, faces):
    v = np.asarray(verts)
    total = 0.0
    for f in faces:
        for k in range(1, len(f) - 1):
            total += np.dot(v[f[0]], np.cross(v[f[k]], v[f[k + 1]])) / 6
    return total


def test_capsule_is_closed_and_faces_outwards():
    verts, faces, t = C.capsule((0, 0, 0), (0, 0, 1), 0.2, 0.1, n=12)
    assert _euler_characteristic(verts, faces) == 2  # a closed surface, like a sphere
    assert _signed_volume(verts, faces) > 0  # outward winding
    assert t.min() == pytest.approx(-0.2) and t.max() == pytest.approx(1.1)
    assert verts[:, 2].min() == pytest.approx(-0.2) and verts[:, 2].max() == pytest.approx(1.1)
    sideways, sideways_faces, _ = C.capsule((0, 0, 0), (1, 0, 0), 0.1, 0.1, n=8)
    assert _signed_volume(sideways, sideways_faces) > 0


def test_frame_is_right_handed():
    for d in ((0, 0, 1), (1, 0, 0), (0.3, -0.5, 0.8)):
        ax, ay = C.frame(d)
        assert np.cross(ax, ay) == pytest.approx(C._unit(d))


def test_strand_tapers_to_a_point():
    verts, faces = C.strand([(0, 0, 1), (0, 0.1, 0.9), (0, 0.1, 0.5)], 0.04, 0.01, (0, 0, 1.0))
    assert _euler_characteristic(verts, faces) == 2
    assert np.isfinite(verts).all() and verts[:, 2].min() < 0.51


def test_parts_are_complete_and_weighted():
    spec = C.CharacterSpec(height=1.6)
    parts = C.build_parts(spec)
    assert set(R.PARTS) <= set(parts) and {"Hair", "Jacket"} <= set(parts)
    for name, part in parts.items():
        assert len(part.roles) == len(part.faces), name
        assert set(part.roles) <= set(C.ROLES), name
        assert set(part.weights) <= set(R.DEFORM_BONES), name
        total = sum(part.weights.values())
        assert total == pytest.approx(np.ones(len(part.verts))), name
        assert part.verts[:, 2].min() >= -1e-9 and part.verts[:, 2].max() < 1.02 * spec.height
        assert max(max(f) for f in part.faces) < len(part.verts)
    for name in ("Thigh.L", "Forearm.R", "Hand.L", "Torso"):
        assert _signed_volume(parts[name].verts, parts[name].faces) > 0, name
    head = parts["Head"]
    assert head.normals is not None and head.normals.shape == head.verts.shape
    # the hair hangs from the head, and its long ends follow the chest
    hair = parts["Hair"]
    low = hair.verts[:, 2] < 0.65 * spec.height
    assert low.any() and (hair.weights["chest"][low] > 0.99).all()


def test_template_joints_are_symmetric():
    j = C.template_joints()
    for name in ("shoulder", "elbow", "wrist", "hip", "knee", "ankle"):
        left, right = np.array(j[f"{name}.L"]), np.array(j[f"{name}.R"])
        assert left == pytest.approx(right * [-1, 1, 1])
    assert set(j) >= {h for _, h, _ in R.DEFORM_BONES.values()}


def test_paint_face():
    spec = C.CharacterSpec(eyes="#2050e0")
    img = C.paint_face(spec, width=256)
    h, w, channels = img.shape
    assert channels == 4 and w == 256 and 0 <= img.min() and img.max() <= 1
    x0, x1, z0, z1 = C.FACE_BOX

    def pixel(x, z):
        return img[int((z1 - z) / (z1 - z0) * h), int((x - x0) / (x1 - x0) * w)]

    assert pixel(x0 + 0.001, z0 + 0.001)[3] < 1e-6  # transparent away from the features
    iris = pixel(0.031, 0.894)  # the lower iris of the left eye
    assert iris[3] == pytest.approx(1) and iris[2] > iris[0]  # opaque and blue
    blush = pixel(0.036, 0.878)
    assert 0 < blush[3] < 1 and blush[0] > 0.8  # soft pink, stored un-premultiplied


def test_spec_from_dict_ignores_unknown_keys():
    spec = C.CharacterSpec.from_dict({"hair": "#000000", "wings": True})
    assert spec.hair == "#000000" and spec.skin == C.CharacterSpec().skin
    assert C.hex_rgb("#ff8000") == pytest.approx((1.0, 0.50196, 0.0), abs=1e-4)
