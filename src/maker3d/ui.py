"""Gradio front end (issue #16).

    from maker3d.ui import launch
    launch()  # in Colab: shows the app and prints a public gradio.live link

Flow: prompt -> 4 candidates -> pick one -> 3D model (TRELLIS.2) -> toon render -> zip.
Each "Generate" makes a new job folder (see pipeline.py), so earlier results stay on
Drive and can be opened again from "Earlier jobs", also after a disconnect.

Gradio is imported inside build_app(), so the helpers here are testable without it.
(No ``from __future__ import annotations``: Gradio reads the ``gr.SelectData`` type hint.)
"""

import argparse
import re
import threading
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path

from maker3d.pipeline import STAGES, Job, Pipeline, ToonSettings, default_jobs_dir, list_jobs
from maker3d.prompts import DEFAULT_HANDS, HAND_TAGS

NEXT_STEP = {
    "images": "press 1 · Generate",
    "pick": "click a candidate, then press 2 · Make the 3D model",
    "model": "press 2 · Make the 3D model",
    "toon": "press 3 · Toon render",
    None: "finished: the zip is under Download",
}
EXAMPLES = [
    "1girl, silver hair, long hair, red jacket, black shorts, thighhighs, boots",
    "1boy, spiky black hair, blue school uniform, sneakers",
    "1girl, witch, purple hat, purple dress, striped stockings",
]


# --------------------------------------------------------------------------------------
# Helpers (no Gradio)
# --------------------------------------------------------------------------------------
class Task:
    """Run a pipeline call in a thread and stream its log lines (for live status)."""

    def __init__(self, pipe: Pipeline, fn: Callable, *args, **kwargs) -> None:
        self.pipe, self.fn, self.args, self.kwargs = pipe, fn, args, kwargs
        self.lines: list[str] = []
        self.result = None
        self.error: BaseException | None = None

    def _target(self) -> None:
        try:
            self.result = self.fn(*self.args, **self.kwargs)
        except BaseException as exc:  # shown in the status box instead of crashing the UI
            self.error = exc

    def follow(self, every: float = 1.0) -> Iterator[str]:
        """Start the call; yield the log text every ``every`` seconds until it ends."""
        old = self.pipe.log

        def log(message: str) -> None:
            self.lines.append(message)
            old(message)

        self.pipe.log = log
        thread = threading.Thread(target=self._target, daemon=True)
        thread.start()
        try:
            while thread.is_alive():
                thread.join(every)
                yield self.text()
        finally:
            self.pipe.log = old

    def text(self, keep: int = 14) -> str:
        lines = self.lines[-keep:]
        if self.error is not None:
            lines.append(f"ERROR: {self.error}")
        return "\n".join(lines)


def to_hex(color: str | None, default: str = "#ffffff") -> str:
    """'#rrggbb' from a colour picker value ('#fff', '#ffffff', 'rgb(...)', 'rgba(...)')."""
    if not color:
        return default
    color = color.strip().lower()
    if re.fullmatch(r"#[0-9a-f]{6}", color):
        return color
    if re.fullmatch(r"#[0-9a-f]{3}", color):
        return "#" + "".join(c * 2 for c in color[1:])
    m = re.fullmatch(r"rgba?\(([^)]*)\)", color)
    if m:
        parts = [p.strip() for p in m.group(1).split(",")][:3]
        if len(parts) == 3:
            rgb = [max(0, min(255, round(float(p)))) for p in parts]
            return "#" + "".join(f"{v:02x}" for v in rgb)
    return default


def job_choices(jobs_dir: str | Path) -> list[tuple[str, str]]:
    """(label, folder) pairs for the 'Earlier jobs' dropdown, newest first."""
    choices = []
    for job in list_jobs(jobs_dir):
        nxt = job.next_stage()
        state = "finished" if nxt is None else f"next: {nxt}"
        choices.append((f"{job.id} ({state})", str(job.dir)))
    return choices


def summary(job: Job) -> str:
    """Status text for a job: folder, seed, every stage, what to do next."""
    lines = [f"Job folder: {job.dir}", f"Seed: {job.data['seed']}"]
    for stage in STAGES:
        info = job.data["stages"].get(stage, {})
        state = "done" if job.is_done(stage) else info.get("status", "todo")
        if state == "done" and info.get("seconds") is not None:
            state += f" ({info['seconds']:.0f}s)"
        lines.append(f"  {stage:<6} {state}")
        if info.get("error") and not job.is_done(stage):
            lines.append(f"         {info['error'].splitlines()[0][:200]}")
    lines.append(f"Next: {NEXT_STEP[job.next_stage()]}")
    return "\n".join(lines)


def view(job: Job) -> dict:
    """Paths the app shows for a job (None / [] where a stage has not run)."""
    return {
        "candidates": [str(p) for p in job.candidates()],
        "pick": job.data.get("pick") or 1,
        "model": str(job.model) if job.is_done("model") else None,
        "stills": [str(p) for p in job.stills()] if job.is_done("toon") else [],
        "video": str(job.turntable) if job.is_done("toon") and job.turntable.is_file() else None,
    }


# --------------------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------------------
def build_app(pipe: Pipeline | None = None, jobs_dir: str | Path | None = None):
    import gradio as gr

    pipe = pipe or Pipeline()
    jobs_root = Path(jobs_dir) if jobs_dir else default_jobs_dir()
    gpu = {"concurrency_limit": 1, "concurrency_id": "gpu"}  # one GPU job at a time
    keep = gr.update  # "leave this output as it is"

    def need_job(job_dir: str | None) -> Job:
        if not job_dir:
            raise gr.Error("Generate images first (step 1), or open an earlier job.")
        return Job.load(job_dir)

    with gr.Blocks(title="3D-Maker") as app:
        job_dir = gr.State(None)
        picked = gr.State(1)

        gr.Markdown(
            "# 3D-Maker\n"
            "Prompt → **Anima Turbo v1.1** anime image → **TRELLIS.2** 3D model → "
            "**Blender** toon render that looks 2D. Results are saved in "
            f"`{jobs_root}`. Anima is for non-commercial use only."
        )

        # --- 1. images ----------------------------------------------------------------
        with gr.Row():
            with gr.Column(scale=3):
                subject = gr.Textbox(
                    label="Character",
                    lines=3,
                    placeholder="Danbooru tags or plain English, e.g. " + EXAMPLES[0],
                )
                gr.Examples(EXAMPLES, inputs=subject)
            with gr.Column(scale=2):
                seed = gr.Number(-1, label="Seed (-1 = random)", precision=0)
                hands = gr.Radio(
                    sorted(HAND_TAGS),
                    value=DEFAULT_HANDS,
                    label="Hands",
                    info="fists survive the 3D step best",
                )
                friendly = gr.Checkbox(
                    True, label="3D-friendly tags (full body, A-pose, front view, white bg)"
                )
        generate = gr.Button("1 · Generate 4 images", variant="primary")
        gallery = gr.Gallery(label="Candidates: click one to pick it", columns=4, height=440)
        picked_note = gr.Markdown("Picked: candidate 1")

        # --- 2. model -----------------------------------------------------------------
        make_model = gr.Button("2 · Make the 3D model (about 2 min)", variant="primary")
        model_view = gr.Model3D(label="3D model (TRELLIS.2)", height=480)

        # --- 3. toon ------------------------------------------------------------------
        td = ToonSettings()
        with gr.Row():
            style = gr.Dropdown(["cel", "flat", "cel_rim"], value=td.style, label="Style")
            bands = gr.Radio([2, 3], value=td.bands, label="Shade bands")
            outline = gr.Slider(
                0.0, 0.012, value=td.outline, step=0.001, label="Outline (0 = none)"
            )
            background = gr.ColorPicker(td.background, label="Background")
        with gr.Row():
            smooth = gr.Slider(
                0.0, 0.05, value=td.normal_smooth, step=0.005, label="Smooth shadows (0 = off)"
            )
            turntable = gr.Checkbox(td.turntable, label="Turntable video")
            hold = gr.Radio(
                [("on twos", 2), ("on threes", 3)], value=td.hold, label="Turntable timing"
            )
        render = gr.Button("3 · Toon render (about 1 min)", variant="primary")
        stills = gr.Gallery(label="Toon stills", columns=4, height=360)
        video = gr.Video(label="Turntable")
        download = gr.File(label="Download: toon.blend, model.glb, stills, video, job.json")

        status = gr.Textbox(label="Status", lines=10, max_lines=24, interactive=False)
        with gr.Accordion("Earlier jobs", open=False):
            earlier = gr.Dropdown(choices=job_choices(jobs_root), label="Open a job", value=None)
            refresh = gr.Button("Refresh the list")

        # --- handlers -----------------------------------------------------------------
        def on_generate(subject_text, seed_value, hands_value, friendly_value):
            if not (subject_text or "").strip():
                raise gr.Error("Describe a character first.")
            seed_int = None if seed_value is None or seed_value < 0 else int(seed_value)
            job = Job.create(
                subject_text,
                seed_int,
                three_d_friendly=bool(friendly_value),
                hands=hands_value,
                jobs_dir=jobs_root,
            )
            head = f"Job folder: {job.dir}\nSeed: {job.data['seed']}\n"
            cleared = ([], "Picked: candidate 1", None, [], None, None)
            yield (str(job.dir), 1, *cleared, head + "starting...", keep())
            task = Task(pipe, pipe.run_images, job)
            for text in task.follow():
                yield (keep(),) * 8 + (head + text, keep())
            candidates = [str(p) for p in job.candidates()]
            yield (
                keep(),
                keep(),
                candidates,
                keep(),
                keep(),
                keep(),
                keep(),
                keep(),
                summary(job) if task.error is None else head + task.text(),
                gr.update(choices=job_choices(jobs_root)),
            )

        generate.click(
            on_generate,
            [subject, seed, hands, friendly],
            [job_dir, picked, gallery, picked_note, model_view, stills, video, download]
            + [status, earlier],
            **gpu,
        )

        def on_select(evt: gr.SelectData):
            number = int(evt.index) + 1
            return number, f"Picked: candidate {number}"

        gallery.select(on_select, None, [picked, picked_note])

        def on_make_model(job_dir_value, number):
            job = need_job(job_dir_value)

            def work():
                pipe.pick(job, int(number))
                return pipe.run_model(job)

            yield None, [], None, None, f"Picked candidate {number}; starting..."
            task = Task(pipe, work)
            for text in task.follow():
                yield keep(), keep(), keep(), keep(), text
            model = str(job.model) if task.error is None else None
            status_text = summary(job) if task.error is None else task.text()
            yield model, keep(), keep(), keep(), status_text

        make_model.click(
            on_make_model,
            [job_dir, picked],
            [model_view, stills, video, download, status],
            **gpu,
        )

        def on_render(
            job_dir_value, style_v, bands_v, outline_v, background_v, smooth_v, turntable_v, hold_v
        ):
            job = need_job(job_dir_value)
            settings = replace(
                job.toon_settings(),
                style=style_v,
                bands=int(bands_v),
                outline=float(outline_v),
                background=to_hex(background_v),
                normal_smooth=float(smooth_v),
                turntable=bool(turntable_v),
                hold=int(hold_v),
            )

            def work():
                pipe.run_toon(job, settings=settings)
                return pipe.make_zip(job)

            yield keep(), keep(), keep(), "starting..."
            task = Task(pipe, work)
            for text in task.follow():
                yield keep(), keep(), keep(), text
            if task.error is not None:
                yield keep(), keep(), keep(), task.text()
                return
            v = view(job)
            yield v["stills"], v["video"], str(task.result), summary(job)

        render.click(
            on_render,
            [job_dir, style, bands, outline, background, smooth, turntable, hold],
            [stills, video, download, status],
            **gpu,
        )

        def on_open(choice):
            if not choice:
                return (keep(),) * 11
            job = Job.load(choice)
            v = view(job)
            s = job.toon_settings()
            zip_path = str(pipe.make_zip(job)) if job.is_done("toon") else None
            return (
                str(job.dir),
                v["pick"],
                v["candidates"],
                f"Picked: candidate {v['pick']}",
                v["model"],
                v["stills"],
                v["video"],
                zip_path,
                summary(job),
                job.data.get("subject", ""),
                s.style,
            )

        earlier.change(
            on_open,
            [earlier],
            [job_dir, picked, gallery, picked_note, model_view, stills, video, download]
            + [status, subject, style],
        )
        refresh.click(lambda: gr.update(choices=job_choices(jobs_root)), None, [earlier])

    return app


def launch(
    share: bool = True,
    jobs_dir: str | Path | None = None,
    pipe: Pipeline | None = None,
    **launch_kwargs,
):
    """Build and start the app. In a notebook this returns right away."""
    pipe = pipe or Pipeline()
    root = Path(jobs_dir) if jobs_dir else default_jobs_dir()
    root.mkdir(parents=True, exist_ok=True)
    pipe.work_dir.mkdir(parents=True, exist_ok=True)
    app = build_app(pipe, root)
    app.queue(default_concurrency_limit=1)
    print(f"Jobs are saved in {root}", flush=True)
    allowed = [str(root), str(pipe.work_dir)]
    return app.launch(share=share, allowed_paths=allowed, **launch_kwargs)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="3D-Maker web app")
    p.add_argument("--no-share", action="store_true", help="no public gradio.live link")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--jobs-dir")
    args = p.parse_args(argv)
    launch(share=not args.no_share, jobs_dir=args.jobs_dir, server_port=args.port)


if __name__ == "__main__":
    main()
