"""Download model files from the Hugging Face Hub into ComfyUI's models folders (issue #3).

Files are fetched into the normal HF cache and symlinked to
``<comfy_dir>/models/<subdir>/<filename>``, so re-running is cheap and nothing is
downloaded twice.

    python -m maker3d.download --comfy-dir /content/ComfyUI
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable, Iterable
from pathlib import Path

from maker3d.config import ALL_MODELS, COMFYUI_DIR, ModelFile

# (repo_id, path_in_repo) -> local file path
Fetcher = Callable[[str, str], str]


def target_path(comfy_dir: str | Path, model: ModelFile) -> Path:
    return Path(comfy_dir) / "models" / model.comfy_subdir / model.filename


def is_present(path: Path) -> bool:
    """True if the file (or the file a symlink points to) exists and is not empty."""
    try:
        return path.exists() and path.stat().st_size > 0
    except OSError:
        return False


def hf_fetch(repo_id: str, path_in_repo: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo_id, filename=path_in_repo)


def ensure_models(
    comfy_dir: str | Path = COMFYUI_DIR,
    models: Iterable[ModelFile] = ALL_MODELS,
    fetch: Fetcher = hf_fetch,
    log: Callable[[str], None] = print,
) -> list[Path]:
    """Make every model available in ComfyUI. Returns the target paths."""
    targets = []
    for model in models:
        dest = target_path(comfy_dir, model)
        targets.append(dest)
        if is_present(dest):
            log(f"ok    {dest}")
            continue
        log(f"fetch {model.repo_id}/{model.path_in_repo}")
        src = Path(fetch(model.repo_id, model.path_in_repo))
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.is_symlink() or dest.exists():
            dest.unlink()
        try:
            os.symlink(src, dest)
        except OSError:
            # Filesystems without symlink support: fall back to a hard link, then a move.
            try:
                os.link(src, dest)
            except OSError:
                src.replace(dest)
        log(f"ok    {dest}")
    return targets


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy-dir", default=COMFYUI_DIR)
    args = parser.parse_args(argv)
    ensure_models(args.comfy_dir)


if __name__ == "__main__":
    main()
