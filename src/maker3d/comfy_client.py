"""Headless ComfyUI HTTP API client (issue #5).

Planned API:
    queue(workflow: dict) -> prompt_id        POST /prompt
    wait(prompt_id, timeout) -> history dict  GET  /history/<prompt_id>
    fetch(filename, subfolder, type) -> bytes GET  /view
"""

from __future__ import annotations


class ComfyClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8188") -> None:
        self.base_url = base_url.rstrip("/")

    def queue(self, workflow: dict) -> str:  # pragma: no cover - issue #5
        raise NotImplementedError("Implemented in issue #5")
