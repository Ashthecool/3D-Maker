"""Start and health-check the headless ComfyUI server on Colab (issue #3)."""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from maker3d.config import COMFYUI_DIR, COMFYUI_URL


def system_stats(url: str = COMFYUI_URL, timeout: float = 5.0) -> dict | None:
    """Return ComfyUI's /system_stats, or None if the server is not reachable."""
    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/system_stats", timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError):
        return None


def wait_ready(
    url: str = COMFYUI_URL,
    timeout: float = 300.0,
    interval: float = 2.0,
    proc: subprocess.Popen | None = None,
) -> dict:
    """Block until ComfyUI answers /system_stats. Raises if it dies or times out."""
    deadline = time.monotonic() + timeout
    while True:
        stats = system_stats(url)
        if stats is not None:
            return stats
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"ComfyUI exited with code {proc.returncode}; see the log file")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"ComfyUI did not answer at {url} within {timeout:.0f}s")
        time.sleep(interval)


def start_comfyui(
    comfy_dir: str | Path = COMFYUI_DIR,
    port: int = 8188,
    log_file: str | Path = "/content/comfyui.log",
    extra_args: list[str] | None = None,
    timeout: float = 300.0,
) -> dict:
    """Start ComfyUI in the background (if not already running) and wait until ready."""
    url = f"http://127.0.0.1:{port}"
    stats = system_stats(url)
    if stats is not None:
        return stats
    cmd = [sys.executable, "main.py", "--listen", "127.0.0.1", "--port", str(port)]
    cmd += extra_args or []
    log = open(log_file, "ab")  # noqa: SIM115 - handle must outlive this function
    proc = subprocess.Popen(cmd, cwd=str(comfy_dir), stdout=log, stderr=subprocess.STDOUT)
    return wait_ready(url, timeout=timeout, proc=proc)


def free_vram(url: str = COMFYUI_URL) -> None:
    """Ask ComfyUI to unload models and free VRAM (before running Blender)."""
    body = json.dumps({"unload_models": True, "free_memory": True}).encode("utf-8")
    req = urllib.request.Request(
        f"{url.rstrip('/')}/free",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30):
        pass


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Start ComfyUI headless and wait for it")
    parser.add_argument("--comfy-dir", default=COMFYUI_DIR)
    parser.add_argument("--port", type=int, default=8188)
    parser.add_argument("--cpu", action="store_true", help="CPU-only (for smoke tests)")
    args = parser.parse_args()
    stats = start_comfyui(args.comfy_dir, args.port, extra_args=["--cpu"] if args.cpu else None)
    system = stats.get("system", {})
    print("ComfyUI ready:", system.get("comfyui_version"), "| python", system.get("python_version"))
    for dev in stats.get("devices", []):
        print(f"  {dev.get('name')}: {dev.get('vram_total', 0) / 2**30:.1f} GiB VRAM")


if __name__ == "__main__":
    main()
