"""UI helpers that do not need Gradio (issue #16)."""

from datetime import datetime

from maker3d import pipeline as pl
from maker3d import ui


def test_to_hex():
    assert ui.to_hex("#FFFFFF") == "#ffffff"
    assert ui.to_hex("#abc") == "#aabbcc"
    assert ui.to_hex("rgba(255, 128, 0, 1)") == "#ff8000"
    assert ui.to_hex("rgb(12.4, 300, -5)") == "#0cff00"
    assert ui.to_hex("") == ui.to_hex(None) == ui.to_hex("blue") == "#ffffff"


def test_job_list_summary_and_view(tmp_path):
    job = pl.Job.create("1girl", seed=5, jobs_dir=tmp_path, now=datetime(2026, 11, 7, 9))
    assert ui.job_choices(tmp_path) == [(f"{job.id} (next: images)", str(job.dir))]
    text = ui.summary(job)
    assert str(job.dir) in text and "Seed: 5" in text and "press 1" in text
    assert ui.view(job) == {"candidates": [], "pick": 1, "model": None, "stills": [], "video": None}

    job.candidates_dir.mkdir()
    (job.candidates_dir / "candidate_1.png").write_bytes(b"x")
    job.mark("images", "done", seconds=4.2)
    job.mark("model", "failed", error="model: the GPU ran out of memory.\nmore")
    text = ui.summary(job)
    assert "images done (4s)" in text and "ran out of memory" in text and "click a" in text
    assert ui.view(job)["candidates"] == [str(job.candidates_dir / "candidate_1.png")]


class _Pipe:
    def __init__(self):
        self.log = lambda message: None


def test_task_streams_log_and_catches_errors():
    pipe = _Pipe()

    def work(n):
        for i in range(n):
            pipe.log(f"line {i}")
        return "result"

    task = ui.Task(pipe, work, 3)
    list(task.follow(every=0.01))
    assert task.result == "result" and task.lines == ["line 0", "line 1", "line 2"]

    def boom():
        raise pl.PipelineError("no GPU")

    task = ui.Task(pipe, boom)
    list(task.follow(every=0.01))
    assert task.text().endswith("ERROR: no GPU")
