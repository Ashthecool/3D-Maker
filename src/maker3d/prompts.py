"""Prompt presets that make Anima output easy to turn into 3D (issue #7).

Anima prompt format (model card): lowercase Danbooru tags with spaces, natural
language, or both. Recommended quality prefix and negative prompt below.
"""

from __future__ import annotations

QUALITY_PREFIX = "masterpiece, best quality, score_7, safe"

# Single-image-to-3D works best with a full, unobstructed, front-facing body.
THREE_D_FRIENDLY_TAGS = (
    "solo, full body, standing, a-pose, arms slightly away from body, "
    "front view, looking at viewer, simple background, white background, no props"
)

# Hands are the weak spot: Anima tends to draw spread, claw-like fingers that are only
# ~50 px wide in a full-body image, and TRELLIS.2 then fuses or drops them. Simple hand
# shapes survive the 3D step much better (tested 2026-11-07, seed 2).
HAND_TAGS = {
    "fist": "clenched hands",
    "relaxed": "open hands, relaxed hands, fingers together",
    "free": "",
}
DEFAULT_HANDS = "fist"
HAND_NEGATIVE = "bad hands, extra digits, fewer digits, fused fingers, spread fingers, claws"

NEGATIVE = (
    "worst quality, low quality, score_1, score_2, score_3, artist name, blurry, "
    "jpeg artifacts, chromatic aberration, cropped, out of frame, multiple views, "
    "dynamic pose, foreshortening, " + HAND_NEGATIVE
)


def build_prompt(subject: str, three_d_friendly: bool = True, hands: str = DEFAULT_HANDS) -> str:
    """Return the positive prompt for a character description.

    ``hands``: one of ``HAND_TAGS`` ("fist" is the most reliable for 3D).
    """
    if hands not in HAND_TAGS:
        raise ValueError(f"hands must be one of {sorted(HAND_TAGS)}")
    parts = [QUALITY_PREFIX]
    if three_d_friendly:
        parts.append(THREE_D_FRIENDLY_TAGS)
        if HAND_TAGS[hands]:
            parts.append(HAND_TAGS[hands])
    subject = subject.strip()
    if subject:
        parts.append(subject)
    return ", ".join(parts)
