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
