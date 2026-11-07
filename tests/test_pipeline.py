"""Job folders, stage order and resume (issue #15), with a fake ComfyUI and a fake Blender."""

import json
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import pytest

from maker3d import pipeline as pl
from maker3d.blender import toon_scene
from maker3d.comfy_client import ComfyExecutionError, OutputFile, find_node

NOW = datetime(2026, 11, 7, 12, 0, 0)

FAKE_BLENDER = """
import sys
from maker3d.blender.toon_scene import parse_args

args = parse_args(sys.argv)
(args.out / "stills").mkdir(parents=True, exist_ok=True)
for name in ("front", "three_quarter", "side", "back"):
    (args.out / "stills" / f"{name}.png").write_bytes(b"png")
if not args.no_turntable:
    (args.out / "turntable.mp4").write_bytes(b"mp4")
info = [args.style, str(args.source_image is not None), str(args.source_mask is not None)]
(args.out / "toon.blend").write_text(" ".join(info))
print("[toon] saved", args.out / "toon.blend")
print("noise that is not forwarded")
"""


class FakeClient:
    """Stands in for ComfyClient: records calls, writes small fake outputs."""

    def __init__(self, fail: dict | None = None):
        self.calls: list[str] = []
        self.fail = dict(fail or {})
        self.uploaded: list[tuple[bytes, str]] = []
        self.freed = 0
        self.workflows: dict[str, dict] = {}

    def run(self, workflow, timeout=600, on_poll=None):
        types = {n["class_type"] for n in workflow.values()}
        stage = "model" if "SaveGLB" in types else "images"
        self.calls.append(stage)
        self.workflows[stage] = workflow
        if stage in self.fail:
            raise self.fail.pop(stage)
        if on_poll:
            on_poll(20.0)
        save = find_node(workflow, "save")
        if stage == "images":
            latent = workflow[find_node(workflow, "latent")]["inputs"]
            return [
                OutputFile(save, "images", f"anima_{i:05d}_.png", "maker3d/x")
                for i in range(1, latent["batch_size"] + 1)
            ]
        return [
            OutputFile(find_node(workflow, "mask"), "images", "model_mask_00001_.png"),
            OutputFile(save, "3d", "model_00001_.glb", "maker3d/x"),
        ]

    def download(self, file, dest_dir):
        dest = Path(dest_dir) / file.filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(file.filename.encode())
        return dest

    def upload_image(self, path, subfolder=""):
        self.uploaded.append((Path(path).read_bytes(), subfolder))
        return f"{subfolder}/{Path(path).name}"

    def free(self, unload_models=True, free_memory=True):
        self.freed += 1


@pytest.fixture
def make_pipe(tmp_path):
    script = tmp_path / "fake_blender.py"
    script.write_text(FAKE_BLENDER)

    def make(client=None, blender_cmd=None):
        messages: list[str] = []
        pipe = pl.Pipeline(
            client=client or FakeClient(),
            blender_cmd=blender_cmd or [sys.executable, str(script), "--"],
            work_dir=tmp_path / "work",
            warm_log=None,
            log=messages.append,
        )
        pipe.messages = messages
        return pipe

    return make


@pytest.fixture
def job(tmp_path):
    return pl.Job.create("1girl, Red jacket!", seed=2, jobs_dir=tmp_path / "jobs", now=NOW)


def test_slugify():
    assert pl.slugify("1girl, Red jacket!") == "1girl-red-jacket"
    assert pl.slugify("   ") == "job"
    assert len(pl.slugify("a" * 100)) == 40


def test_create_and_load(tmp_path, job):
    assert job.id == "20261107-120000-1girl-red-jacket"
    assert "clenched hands" in job.data["prompt"] and job.data["seed"] == 2
    again = pl.Job.create("1girl, Red jacket!", jobs_dir=tmp_path / "jobs", now=NOW)
    assert again.id.endswith("-2") and 0 <= again.data["seed"] < 2**32
    loaded = pl.Job.load(job.dir / "job.json")
    assert loaded.data == job.data and loaded.next_stage() == "images"
    assert loaded.model_settings() == pl.Trellis2Settings()
    assert loaded.toon_settings() == pl.ToonSettings()
    with pytest.raises(ValueError):
        pl.Job.create("x", hands="claws", jobs_dir=tmp_path / "other")
    assert not (tmp_path / "other").exists()
    assert [j.id for j in pl.list_jobs(tmp_path / "jobs")] == [again.id, job.id]


def test_settings_ignore_unknown_keys():
    assert pl.from_dict(pl.ToonSettings, {"style": "flat", "removed_option": 1}).style == "flat"


def test_full_run_then_nothing_to_do(make_pipe, job):
    pipe = make_pipe()
    pipe.run_all(job, pick=2)
    client = pipe.client
    assert client.calls == ["images", "model"]
    assert [p.name for p in job.candidates()] == [f"candidate_{i}.png" for i in range(1, 5)]
    assert job.image.read_bytes() == b"anima_00002_.png"
    assert client.uploaded == [(b"anima_00002_.png", f"maker3d/{job.id}")]
    assert job.mask.read_bytes() == b"model_mask_00001_.png"
    assert job.model.read_bytes() == b"model_00001_.glb"
    assert job.blend.read_text() == "cel True True"  # projection got image + mask
    assert [p.stem for p in job.stills()] == ["front", "three_quarter", "side", "back"]
    assert job.turntable.is_file() and (job.dir / "blender.log").is_file()
    assert client.freed == 1 and job.next_stage() is None
    assert "[toon] saved" in " ".join(pipe.messages)
    assert "noise" not in " ".join(pipe.messages)
    assert {s: job.status(s) for s in pl.STAGES} == dict.fromkeys(pl.STAGES, "done")
    # The prompt and seed reached the workflows.
    anima = client.workflows["images"]
    assert anima[find_node(anima, "prompt")]["inputs"]["text"] == job.data["prompt"]
    assert anima[find_node(anima, "sampler")]["inputs"]["seed"] == 2

    pipe.run_all(pl.Job.load(job.dir))
    assert client.calls == ["images", "model"]  # everything was done already


def test_crash_in_model_stage_resumes_there(make_pipe, job):
    err = ComfyExecutionError({"node_id": "9", "node_type": "KSampler", "exception_message": "x"})
    pipe = make_pipe(FakeClient(fail={"model": err}))
    with pytest.raises(ComfyExecutionError):
        pipe.run_all(job, pick=1)
    reloaded = pl.Job.load(job.dir)
    assert reloaded.status("model") == "failed" and reloaded.next_stage() == "model"
    assert "KSampler" in reloaded.data["stages"]["model"]["error"]

    # A runtime that was killed mid-stage leaves "running" behind: also resumes there.
    reloaded.mark("model", "running")
    assert pl.Job.load(job.dir).next_stage() == "model"

    fresh = make_pipe()
    fresh.run_all(pl.Job.load(job.dir))
    assert fresh.client.calls == ["model"]  # the images were not generated again
    assert pl.Job.load(job.dir).next_stage() is None


def test_out_of_memory_gives_a_hint(make_pipe, job):
    oom = ComfyExecutionError(
        {"exception_type": "torch.OutOfMemoryError", "exception_message": "CUDA out of memory"}
    )
    pipe = make_pipe(FakeClient(fail={"model": oom}))
    with pytest.raises(pl.PipelineError, match="upsample_resolution"):
        pipe.run_all(job)
    assert "ran out of memory" in pl.Job.load(job.dir).data["stages"]["model"]["error"]


def test_new_toon_settings_render_again(make_pipe, job):
    pipe = make_pipe()
    pipe.run_all(job)
    pipe.run_toon(job, settings=pl.ToonSettings(style="flat", turntable=False))
    assert job.blend.read_text() == "flat True True"
    assert not job.turntable.exists()  # no stale video from the earlier render
    assert pipe.client.calls == ["images", "model"]
    assert pl.Job.load(job.dir).toon_settings().style == "flat"


def test_new_pick_or_seed_resets_later_stages(make_pipe, job):
    pipe = make_pipe()
    pipe.run_all(job, pick=1)
    pipe.pick(job, 3)
    assert job.image.read_bytes() == b"anima_00003_.png"
    assert job.status("model") == job.status("toon") == "todo"
    assert job.next_stage() == "model"
    with pytest.raises(pl.PipelineError, match="1 to 4"):
        pipe.pick(job, 5)

    pipe.run_images(job, seed=7)
    assert job.data["seed"] == 7 and pipe.client.calls == ["images", "model", "images"]
    assert job.next_stage() == "pick"


def test_stage_order_is_enforced(make_pipe, job):
    pipe = make_pipe()
    with pytest.raises(pl.PipelineError, match="pick"):
        pipe.run_model(job)
    with pytest.raises(pl.PipelineError, match="3D model"):
        pipe.run_toon(job)


def test_blender_failure_keeps_the_log(make_pipe, job):
    fail = [sys.executable, "-c", "import sys; print('boom'); sys.exit(3)", "--"]
    pipe = make_pipe(blender_cmd=fail)
    with pytest.raises(pl.PipelineError, match="code 3") as info:
        pipe.run_all(job)
    assert "boom" in str(info.value)
    assert (job.dir / "blender.log").read_text().strip() == "boom"
    assert pl.Job.load(job.dir).next_stage() == "toon"


def test_run_logged_timeout(tmp_path):
    cmd = [sys.executable, "-c", "import time; time.sleep(30)"]
    with pytest.raises(pl.PipelineError, match="longer than"):
        pl.run_logged(cmd, tmp_path / "log.txt", timeout=0.5)


def test_toon_settings_match_the_blender_script(tmp_path):
    image, mask = tmp_path / "image.png", tmp_path / "mask.png"
    image.write_bytes(b"x")
    args = toon_scene.parse_args(pl.ToonSettings().to_args(Path("m.glb"), Path("o"), image, mask))
    defaults = toon_scene.parse_args(["--glb", "m.glb", "--out", "o"])
    for key in ("style", "bands", "outline", "res", "samples", "frames", "background"):
        assert getattr(args, key) == getattr(defaults, key), key
    assert args.source_image == image and args.source_mask is None  # mask file is missing
    flat = pl.ToonSettings(outline=0, turntable=False, transparent=True, projection=False)
    args = toon_scene.parse_args(flat.to_args(Path("m.glb"), Path("o"), image, mask))
    assert args.no_outline and args.no_turntable and args.transparent
    assert args.source_image is None


def test_waits_for_kernel_warmup(make_pipe, tmp_path):
    log = tmp_path / "warm.log"
    log.write_text("[warm] compiling\n")
    pipe = make_pipe()
    pipe.warm_log = log
    pipe._wait_for_kernels(timeout=0.05, interval=0.01)
    assert any("rendering anyway" in m for m in pipe.messages)
    log.write_text("[warm] ready\nWARM_DONE\n")
    pipe.messages.clear()
    pipe._wait_for_kernels(timeout=0.05, interval=0.01)
    assert pipe.messages == []


def test_zip(make_pipe, job):
    pipe = make_pipe()
    pipe.run_all(job)
    with zipfile.ZipFile(pipe.make_zip(job)) as z:
        names = set(z.namelist())
    expected = {"job.json", "image.png", "mask.png", "model.glb", "toon.blend", "turntable.mp4"}
    assert {f"{job.id}/{n}" for n in expected} <= names
    assert f"{job.id}/stills/front.png" in names
    assert not any("candidates" in n for n in names)


def test_cli_resume_redo(monkeypatch, make_pipe, tmp_path, job):
    pipe = make_pipe()
    pipe.run_all(job)
    monkeypatch.setattr(pl, "Pipeline", lambda log: pipe)
    jobs = str(tmp_path / "jobs")
    done = pl.main(["--resume", "latest", "--jobs-dir", jobs, "--style", "flat"])
    assert done.blend.read_text().startswith("flat")
    assert pipe.client.calls == ["images", "model"]
    done = pl.main(["--resume", str(job.dir), "--redo", "model"])
    assert pipe.client.calls == ["images", "model", "model"]
    assert json.loads((job.dir / "job.json").read_text())["toon_settings"]["style"] == "flat"
    with pytest.raises(SystemExit):
        pl.main([])
