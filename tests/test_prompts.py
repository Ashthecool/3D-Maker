from maker3d.prompts import QUALITY_PREFIX, build_prompt


def test_prompt_starts_with_quality_prefix():
    assert build_prompt("1girl, silver hair").startswith(QUALITY_PREFIX)


def test_3d_friendly_tags_toggle():
    assert "a-pose" in build_prompt("1boy")
    assert "a-pose" not in build_prompt("1boy", three_d_friendly=False)


def test_empty_subject_has_no_trailing_comma():
    assert not build_prompt("   ").endswith(",")
