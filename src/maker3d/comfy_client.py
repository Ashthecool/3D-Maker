"""Headless ComfyUI HTTP API client (issue #5).

    client = ComfyClient()
    files = client.run(workflows.anima_t2i("1girl, ...", seed=1), timeout=300)
    client.download(files[0], "/content/out")

Endpoints used: POST ``/prompt``, GET ``/history/<id>``, GET ``/view``,
POST ``/upload/image``, POST ``/interrupt``, POST ``/free``, GET ``/system_stats``.
"""

from __future__ import annotations

import copy
import time
import uuid
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import requests

from maker3d.config import COMFYUI_URL

Workflow = dict[str, dict]


# --------------------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------------------
class ComfyError(RuntimeError):
    """Something ComfyUI reported as a failure."""


class ComfyValidationError(ComfyError):
    """POST /prompt rejected the workflow before running it."""

    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.node_errors: dict = payload.get("node_errors") or {}
        super().__init__(format_validation_error(payload))


class ComfyExecutionError(ComfyError):
    """A node raised an exception while the workflow was running."""

    def __init__(self, details: dict) -> None:
        self.details = details
        self.node_id: str | None = details.get("node_id")
        self.node_type: str | None = details.get("node_type")
        exc = details.get("exception_type", "Exception")
        msg = (details.get("exception_message") or "").strip()
        super().__init__(f"node {self.node_id} ({self.node_type}) failed: {exc}: {msg}")


def format_validation_error(payload: dict) -> str:
    """One readable message from a /prompt error response."""
    error = payload.get("error") or {}
    lines = [error.get("message") or "workflow rejected"]
    if error.get("details"):
        lines[0] += f": {error['details']}"
    for node_id, info in (payload.get("node_errors") or {}).items():
        for err in info.get("errors", []):
            detail = f" ({err['details']})" if err.get("details") else ""
            lines.append(
                f"  node {node_id} ({info.get('class_type')}): {err.get('message')}{detail}"
            )
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Workflow helpers
# --------------------------------------------------------------------------------------
def find_node(workflow: Workflow, title: str) -> str:
    """Return the id of the one node whose ``_meta.title`` is ``title``."""
    ids = [nid for nid, node in workflow.items() if node.get("_meta", {}).get("title") == title]
    if not ids:
        raise KeyError(f"no node titled {title!r}")
    if len(ids) > 1:
        raise ValueError(f"several nodes titled {title!r}: {ids}")
    return ids[0]


def with_inputs(workflow: Workflow, title: str, **inputs) -> Workflow:
    """Copy of ``workflow`` with ``inputs`` set on the node titled ``title``."""
    wf = copy.deepcopy(workflow)
    wf[find_node(wf, title)]["inputs"].update(inputs)
    return wf


@dataclass(frozen=True)
class OutputFile:
    """A file a node wrote, as listed in a /history entry."""

    node_id: str
    kind: str  # key in the node's outputs, e.g. "images" or "3d"
    filename: str
    subfolder: str = ""
    type: str = "output"  # "output", "temp" or "input"


def output_files(entry: dict) -> list[OutputFile]:
    """Every file listed in a /history entry's outputs, in node order."""
    files = []
    for node_id, outputs in (entry.get("outputs") or {}).items():
        for kind, items in outputs.items():
            if not isinstance(items, list):
                continue
            for item in items:
                if isinstance(item, dict) and item.get("filename"):
                    files.append(
                        OutputFile(
                            node_id=str(node_id),
                            kind=kind,
                            filename=item["filename"],
                            subfolder=item.get("subfolder", ""),
                            type=item.get("type", "output"),
                        )
                    )
    return files


# --------------------------------------------------------------------------------------
# Client
# --------------------------------------------------------------------------------------
class ComfyClient:
    def __init__(
        self,
        base_url: str = COMFYUI_URL,
        client_id: str | None = None,
        session: requests.Session | None = None,
        request_timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.client_id = client_id or uuid.uuid4().hex
        self.session = session or requests.Session()
        self.request_timeout = request_timeout

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _get(self, path: str, **params) -> requests.Response:
        r = self.session.get(self._url(path), params=params or None, timeout=self.request_timeout)
        r.raise_for_status()
        return r

    def _post(self, path: str, **kwargs) -> requests.Response:
        r = self.session.post(self._url(path), timeout=self.request_timeout, **kwargs)
        r.raise_for_status()
        return r

    # --- queue and wait -------------------------------------------------------------
    def queue(self, workflow: Workflow) -> str:
        """Queue ``workflow``; return its prompt id. Raises ComfyValidationError if rejected."""
        r = self.session.post(
            self._url("/prompt"),
            json={"prompt": workflow, "client_id": self.client_id},
            timeout=self.request_timeout,
        )
        if r.status_code == 400:
            try:
                payload = r.json()
            except ValueError:
                payload = {"error": {"message": r.text}}
            raise ComfyValidationError(payload)
        r.raise_for_status()
        data = r.json()
        if data.get("node_errors"):
            # Queued anyway: ComfyUI runs the outputs that did validate.
            warnings.warn(format_validation_error(data), stacklevel=2)
        return data["prompt_id"]

    def history(self, prompt_id: str) -> dict | None:
        """The /history entry for ``prompt_id``, or None while it is queued or running."""
        return self._get(f"/history/{prompt_id}").json().get(prompt_id)

    def wait(
        self,
        prompt_id: str,
        timeout: float = 600.0,
        interval: float = 1.0,
        on_poll: Callable[[float], None] | None = None,
    ) -> dict:
        """Poll until ``prompt_id`` finishes; return its /history entry.

        Raises ComfyExecutionError if a node failed, ComfyError if the run was
        interrupted and TimeoutError if it is still unfinished after ``timeout`` seconds.
        """
        start = time.monotonic()
        while True:
            entry = self.history(prompt_id)
            if entry is not None:
                check_status(entry)
                return entry
            elapsed = time.monotonic() - start
            if elapsed >= timeout:
                raise TimeoutError(f"prompt {prompt_id} not finished after {timeout:.0f}s")
            if on_poll is not None:
                on_poll(elapsed)
            time.sleep(interval)

    def run(self, workflow: Workflow, timeout: float = 600.0, **wait_kwargs) -> list[OutputFile]:
        """Queue ``workflow``, wait for it and return the files it wrote."""
        return output_files(self.wait(self.queue(workflow), timeout=timeout, **wait_kwargs))

    # --- files ----------------------------------------------------------------------
    def fetch(self, filename: str, subfolder: str = "", type: str = "output") -> bytes:
        return self._get("/view", filename=filename, subfolder=subfolder, type=type).content

    def download(self, file: OutputFile, dest_dir: str | Path) -> Path:
        dest = Path(dest_dir) / file.filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(self.fetch(file.filename, file.subfolder, file.type))
        return dest

    def upload_image(self, path: str | Path, subfolder: str = "", overwrite: bool = True) -> str:
        """Upload to ComfyUI's input folder; return the name to give LoadImage."""
        path = Path(path)
        with path.open("rb") as fh:
            r = self._post(
                "/upload/image",
                files={"image": (path.name, fh, "application/octet-stream")},
                data={"type": "input", "subfolder": subfolder, "overwrite": str(overwrite).lower()},
            )
        info = r.json()
        return f"{info['subfolder']}/{info['name']}" if info.get("subfolder") else info["name"]

    # --- server control -------------------------------------------------------------
    def system_stats(self) -> dict:
        return self._get("/system_stats").json()

    def interrupt(self) -> None:
        self._post("/interrupt", json={})

    def free(self, unload_models: bool = True, free_memory: bool = True) -> None:
        self._post("/free", json={"unload_models": unload_models, "free_memory": free_memory})


def check_status(entry: dict) -> None:
    """Raise if a finished /history entry did not succeed."""
    status = entry.get("status") or {}
    if status.get("status_str", "success") == "success":
        return
    for event, data in status.get("messages", []):
        if event == "execution_error":
            raise ComfyExecutionError(data)
    for event, data in status.get("messages", []):
        if event == "execution_interrupted":
            raise ComfyError(f"interrupted at node {data.get('node_id')} ({data.get('node_type')})")
    raise ComfyError(f"workflow finished with status {status.get('status_str')!r}")
