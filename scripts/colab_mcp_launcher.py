"""Start colab-mcp so that it opens a chosen Colab notebook instead of a blank one.

colab-mcp always opens ``/notebooks/empty.ipynb#mcpProxyToken=...&mcpProxyPort=...`` and its
MCP session stays bound to the first tab that connects: moving the token to another tab later
leaves every tool call hanging. This wrapper points that URL at the notebook named in
``$COLAB_NOTEBOOK`` or in the git-ignored file ``.colab-notebook`` at the repo root. Either may
hold a Colab URL, a ``drive/<id>`` path or a bare Drive file id. Without one, colab-mcp opens
its usual scratch notebook.

Run by ``.mcp.json`` (colab-mcp needs Python 3.13+)::

    uv run --no-project --python 3.13 --with git+https://github.com/googlecolab/colab-mcp \
        python scripts/colab_mcp_launcher.py
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

NOTEBOOK_FILE = Path(__file__).resolve().parent.parent / ".colab-notebook"
_DRIVE_ID = re.compile(r"[A-Za-z0-9_-]{20,}")


def notebook_path(value: str) -> str | None:
    """Turn a Colab URL, ``drive/<id>`` or a bare Drive id into a path like ``/drive/<id>``."""
    value = value.strip().split("#", 1)[0].split("?", 1)[0]
    if not value:
        return None
    value = re.sub(r"^https?://colab\.(research\.)?google\.com", "", value)
    if _DRIVE_ID.fullmatch(value):
        return f"/drive/{value}"
    if not value.startswith("/"):
        value = "/" + value
    if not re.fullmatch(r"/(drive|notebooks|github|gist)/[\w./~-]+", value):
        raise ValueError(f"not a Colab notebook: {value!r}")
    return value


def configured_path(env: dict | None = None, file: Path = NOTEBOOK_FILE) -> str | None:
    env = os.environ if env is None else env
    if env.get("COLAB_NOTEBOOK"):
        return notebook_path(env["COLAB_NOTEBOOK"])
    if file.is_file():
        return notebook_path(file.read_text(encoding="utf-8"))
    return None


def main() -> None:
    import colab_mcp
    import colab_mcp.session

    path = configured_path()
    if path:
        # Read at call time by open_colab_browser_connection.
        colab_mcp.session.SCRATCH_PATH = path
    colab_mcp.main()


if __name__ == "__main__":
    sys.exit(main())
