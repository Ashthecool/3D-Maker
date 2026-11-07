from pathlib import Path

from maker3d.config import ModelFile
from maker3d.download import ensure_models, target_path

MODEL = ModelFile("org/repo", "split_files/vae/some_vae.safetensors", "vae")


def test_target_path_uses_comfy_subdir_and_filename(tmp_path):
    assert target_path(tmp_path, MODEL) == tmp_path / "models" / "vae" / "some_vae.safetensors"


def _fake_fetch(cache: Path, calls: list):
    def fetch(repo_id: str, path_in_repo: str) -> str:
        calls.append((repo_id, path_in_repo))
        f = cache / path_in_repo.replace("/", "_")
        f.write_bytes(b"weights")
        return str(f)

    return fetch


def test_downloads_missing_then_skips(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    calls: list = []
    fetch = _fake_fetch(cache, calls)
    comfy = tmp_path / "ComfyUI"

    (dest,) = ensure_models(comfy, [MODEL], fetch=fetch, log=lambda _: None)
    assert dest.read_bytes() == b"weights"
    assert calls == [("org/repo", "split_files/vae/some_vae.safetensors")]

    ensure_models(comfy, [MODEL], fetch=fetch, log=lambda _: None)
    assert len(calls) == 1  # second run is a no-op


def test_replaces_empty_file(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    calls: list = []
    comfy = tmp_path / "ComfyUI"
    dest = target_path(comfy, MODEL)
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"")  # interrupted download

    ensure_models(comfy, [MODEL], fetch=_fake_fetch(cache, calls), log=lambda _: None)
    assert calls and dest.read_bytes() == b"weights"
