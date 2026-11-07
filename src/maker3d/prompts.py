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

NEGATIVE = (
    "worst quality, low quality, score_1, score_2, score_3, artist name, blurry, "
    "jpeg artifacts, chromatic aberration, cropped, out of frame, multiple views, "
    "dynamic pose, foreshortening"
)


def build_prompt(subject: str, three_d_friendly: bool = True) -> str:
    """Return the positive prompt for a character description."""
    parts = [QUALITY_PREFIX]
    if three_d_friendly:
        parts.append(THREE_D_FRIENDLY_TAGS)
    subject = subject.strip()
    if subject:
        parts.append(subject)
    return ", ".join(parts)
