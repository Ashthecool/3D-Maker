"""Job orchestration across the stages (issue #15).

    images  Anima Turbo v1.1      prompt -> candidates/candidate_N.png        (ComfyUI)
    pick    one candidate         -> image.png
    model   BiRefNet + TRELLIS.2  image.png -> mask.png + model.glb          (ComfyUI)
    toon    Blender toon render   model.glb -> stills/, turntable.mp4, toon.blend

Each job is one folder (on Google Drive when it is mounted) with a ``job.json`` that
records the prompt, seed, settings and the status of every stage. A stage counts as done
only when job.json says so *and* its output exists, so a Colab session that died in the
middle of a stage resumes at that stage. Re-running a stage resets the stages after it.

    python -m maker3d.pipeline "1girl, red jacket, black shorts" --seed 2 --pick 2
    python -m maker3d.pipeline --resume latest
    python -m maker3d.pipeline --resume latest --redo toon --style flat
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import threading
import time
import zipfile
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path

from maker3d import workflows
from maker3d.comfy_client import ComfyClient, OutputFile, find_node
from maker3d.config import BLENDER_DIR, DRIVE_JOBS_DIR, DRIVE_ROOT, AnimaDefaults
from maker3d.prompts import DEFAULT_HANDS, HAND_TAGS, NEGATIVE, build_prompt
from maker3d.workflows import Trellis2Settings

STAGES = ("images", "pick", "model", "toon")
TOON_SCRIPT = Path(__file__).parent / "blender" / "toon_scene.py"
LOCAL_JOBS_DIR = Path("/content/jobs")
WORK_DIR = Path("/content/work")
WARM_LOG = Path("/content/blender_warm.log")  # written by setup_colab.sh
Log = Callable[[str], None]


class PipelineError(RuntimeError):
    """A stage failed; the message says what to try next."""


# --------------------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ToonSettings:
    """Options for blender/toon_scene.py (same defaults)."""

    style: str = "cel"  # flat | cel | cel_rim
    bands: int = 2
    outline: float = 0.004  # fraction of the model height; 0 = no outline
    res: int = 1024
    samples: int = 16
    frames: int = 96  # turntable length at 24 fps
    hold: int = 2  # frames per pose: 2 = on twos, 3 = on threes (#22)
    ease: float = 0.3  # linger on the front/back views; 0 = even spin
    turntable: bool = True
    background: str = "#ffffff"
    transparent: bool = False
    normal_smooth: float = 0.02  # blur shading normals for cleaner shadows; 0 = off (#23)
    projection: bool = True  # put the crisp Anima image on front-facing surfaces
    color_match: bool = True  # recolour TRELLIS.2's darker texture to the image (#24)

    def to_args(
        self, glb: Path, out: Path, image: Path | None = None, mask: Path | None = None
    ) -> list[str]:
        args = ["--glb", str(glb), "--out", str(out), "--style", self.style]
        args += ["--bands", str(self.bands), "--res", str(self.res)]
        args += ["--samples", str(self.samples), "--frames", str(self.frames)]
        args += ["--hold", str(self.hold), "--ease", str(self.ease)]
        args += ["--normal-smooth", str(self.normal_smooth)]
        args += ["--background", self.background]
        args += ["--outline", str(self.outline)] if self.outline > 0 else ["--no-outline"]
        if not self.turntable:
            args.append("--no-turntable")
        if self.transparent:
            args.append("--transparent")
        if self.projection and image is not None and image.is_file():
            args += ["--source-image", str(image)]
            if mask is not None and mask.is_file():
                args += ["--source-mask", str(mask)]
            if not self.color_match:
                args.append("--no-color-match")
        return args


def from_dict(cls, data: dict | None):
    """Build a settings dataclass from job.json, ignoring keys it does not know."""
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in (data or {}).items() if k in names})


# --------------------------------------------------------------------------------------
# Job folder
# --------------------------------------------------------------------------------------
def slugify(text: str, max_len: int = 40) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_len].rstrip("-") or "job"


def default_jobs_dir() -> Path:
    """MyDrive/3D-Maker/jobs when Drive is mounted, else /content/jobs (lost on disconnect)."""
    return Path(DRIVE_JOBS_DIR) if Path(DRIVE_ROOT).parent.is_dir() else LOCAL_JOBS_DIR


class Job:
    FILE = "job.json"

    def __init__(self, path: str | Path, data: dict) -> None:
        self.dir = Path(path)
        self.data = data

    # --- create / load / save ---------------------------------------------------------
    @classmethod
    def create(
        cls,
        subject: str,
        seed: int | None = None,
        *,
        three_d_friendly: bool = True,
        hands: str = DEFAULT_HANDS,
        jobs_dir: str | Path | None = None,
        image_settings: AnimaDefaults | None = None,
        model_settings: Trellis2Settings | None = None,
        toon_settings: ToonSettings | None = None,
        now: datetime | None = None,
    ) -> Job:
        prompt = build_prompt(subject, three_d_friendly, hands)  # validates ``hands`` first
        now = now or datetime.now()
        seed = random.randrange(2**32) if seed is None or seed < 0 else int(seed)
        root = Path(jobs_dir) if jobs_dir else default_jobs_dir()
        base = f"{now:%Y%m%d-%H%M%S}-{slugify(subject)}"
        path, n = root / base, 2
        while path.exists():
            path, n = root / f"{base}-{n}", n + 1
        path.mkdir(parents=True)
        data = {
            "id": path.name,
            "created": now.isoformat(timespec="seconds"),
            "subject": subject,
            "three_d_friendly": three_d_friendly,
            "hands": hands,
            "prompt": prompt,
            "negative": NEGATIVE,
            "seed": seed,
            "pick": None,
            "image_settings": asdict(image_settings or AnimaDefaults()),
            "model_settings": asdict(model_settings or Trellis2Settings()),
            "toon_settings": asdict(toon_settings or ToonSettings()),
            "stages": {s: {"status": "todo"} for s in STAGES},
        }
        job = cls(path, data)
        job.save()
        return job

    @classmethod
    def load(cls, path: str | Path) -> Job:
        path = Path(path)
        if path.name == cls.FILE:
            path = path.parent
        return cls(path, json.loads((path / cls.FILE).read_text("utf-8")))

    def save(self) -> None:
        tmp = self.dir / (self.FILE + ".tmp")
        tmp.write_text(json.dumps(self.data, indent=2) + "\n", "utf-8")
        os.replace(tmp, self.dir / self.FILE)

    # --- paths ----------------------------------------------------------------------
    @property
    def id(self) -> str:
        return self.data["id"]

    @property
    def candidates_dir(self) -> Path:
        return self.dir / "candidates"

    def candidates(self) -> list[Path]:
        return sorted(self.candidates_dir.glob("candidate_*.png"), key=_number)

    image = property(lambda self: self.dir / "image.png")
    mask = property(lambda self: self.dir / "mask.png")
    model = property(lambda self: self.dir / "model.glb")
    stills_dir = property(lambda self: self.dir / "stills")
    turntable = property(lambda self: self.dir / "turntable.mp4")
    blend = property(lambda self: self.dir / "toon.blend")

    def stills(self) -> list[Path]:
        order = ("front", "three_quarter", "side", "back")
        found = {p.stem: p for p in self.stills_dir.glob("*.png")}
        return [found[n] for n in order if n in found] + [
            p for n, p in sorted(found.items()) if n not in order
        ]

    def deliverables(self) -> list[Path]:
        """What goes in the download zip."""
        files = [self.dir / self.FILE, self.image, self.mask, self.model, self.blend]
        files += [self.turntable, *self.stills()]
        return [p for p in files if p.is_file()]

    # --- settings ---------------------------------------------------------------------
    def image_settings(self) -> AnimaDefaults:
        return from_dict(AnimaDefaults, self.data.get("image_settings"))

    def model_settings(self) -> Trellis2Settings:
        return from_dict(Trellis2Settings, self.data.get("model_settings"))

    def toon_settings(self) -> ToonSettings:
        return from_dict(ToonSettings, self.data.get("toon_settings"))

    # --- stage status -----------------------------------------------------------------
    def status(self, stage: str) -> str:
        return self.data["stages"].get(stage, {}).get("status", "todo")

    def has_output(self, stage: str) -> bool:
        if stage == "images":
            return bool(self.candidates())
        path = {"pick": self.image, "model": self.model, "toon": self.blend}[stage]
        return path.is_file() and path.stat().st_size > 0

    def is_done(self, stage: str) -> bool:
        return self.status(stage) == "done" and self.has_output(stage)

    def next_stage(self) -> str | None:
        """The first stage that still has to run, or None when the job is finished."""
        return next((s for s in STAGES if not self.is_done(s)), None)

    def mark(self, stage: str, status: str, **info) -> None:
        self.data["stages"][stage] = {
            "status": status,
            **info,
            "updated": datetime.now().isoformat(timespec="seconds"),
        }
        self.save()

    def reset_after(self, stage: str) -> None:
        """Later stages depend on this one's output, so they have to run again."""
        for later in STAGES[STAGES.index(stage) + 1 :]:
            if self.status(later) != "todo":
                self.data["stages"][later] = {"status": "todo"}
        self.save()

    def __repr__(self) -> str:
        return f"Job({self.dir}, next={self.next_stage()})"


def _number(path: Path) -> int:
    m = re.search(r"(\d+)$", path.stem)
    return int(m.group(1)) if m else 0


def list_jobs(jobs_dir: str | Path | None = None) -> list[Job]:
    """Jobs in ``jobs_dir``, newest first (folder names start with a timestamp)."""
    root = Path(jobs_dir) if jobs_dir else default_jobs_dir()
    if not root.is_dir():
        return []
    paths = sorted((p for p in root.iterdir() if (p / Job.FILE).is_file()), reverse=True)
    return [Job.load(p) for p in paths]


# --------------------------------------------------------------------------------------
# Errors and subprocesses
# --------------------------------------------------------------------------------------
OOM_HINTS = {
    "images": "a smaller image size or batch size (AnimaDefaults)",
    "model": (
        "lighter TRELLIS.2 settings, e.g. upsample_resolution=1024, "
        "decimate_faces=300000, texture_size=2048"
    ),
    "toon": "a lower --res or --samples",
}


def explain_error(stage: str, exc: BaseException) -> BaseException:
    """Turn known failures into a PipelineError that says what to try."""
    text = str(exc)
    if "outofmemory" in text.lower().replace(" ", ""):
        return PipelineError(
            f"{stage}: the GPU ran out of memory. Try again with {OOM_HINTS[stage]}. ({text})"
        )
    return exc


def default_blender_cmd() -> list[str]:
    exe = shutil.which("blender") or f"{BLENDER_DIR}/blender"
    # --python-exit-code: a Python exception in the script makes Blender exit non-zero.
    return [exe, "-b", "--factory-startup", "--python-exit-code", "1"] + [
        "-P",
        str(TOON_SCRIPT),
        "--",
    ]


def run_logged(
    cmd: Sequence[str],
    log_path: Path,
    on_line: Log | None = None,
    timeout: float | None = None,
) -> None:
    """Run ``cmd``, copying its output to ``log_path`` line by line. Raises on failure."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timed_out = threading.Event()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            list(cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        def kill() -> None:
            timed_out.set()
            proc.kill()

        timer = threading.Timer(timeout, kill) if timeout else None
        if timer:
            timer.start()
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                log.write(line)
                log.flush()
                if on_line:
                    on_line(line.rstrip("\n"))
            code = proc.wait()
        finally:
            if timer:
                timer.cancel()
    if timed_out.is_set():
        raise PipelineError(f"{Path(cmd[0]).name} took longer than {timeout:.0f}s; see {log_path}")
    if code != 0:
        tail = "\n".join(log_path.read_text("utf-8", errors="replace").splitlines()[-15:])
        raise PipelineError(f"{Path(cmd[0]).name} exited with code {code}:\n{tail}")


# --------------------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------------------
class Pipeline:
    """Runs the stages of a Job. ``client`` and ``blender_cmd`` are swappable for tests."""

    def __init__(
        self,
        client: ComfyClient | None = None,
        blender_cmd: Sequence[str] | None = None,
        work_dir: str | Path = WORK_DIR,
        warm_log: str | Path | None = WARM_LOG,
        log: Log = print,
    ) -> None:
        self.client = client or ComfyClient()
        self.blender_cmd = list(blender_cmd) if blender_cmd else default_blender_cmd()
        self.work_dir = Path(work_dir)
        self.warm_log = Path(warm_log) if warm_log else None
        self.log = log

    # --- helpers --------------------------------------------------------------------
    @contextmanager
    def _stage(self, job: Job, stage: str) -> Iterator[None]:
        job.reset_after(stage)
        job.mark(stage, "running")
        t0 = time.monotonic()
        try:
            yield
        except BaseException as exc:
            err = explain_error(stage, exc)
            seconds = round(time.monotonic() - t0, 1)
            job.mark(stage, "failed", error=str(err)[:2000], seconds=seconds)
            if err is not exc:
                raise err from exc
            raise
        seconds = round(time.monotonic() - t0, 1)
        job.mark(stage, "done", seconds=seconds)
        self.log(f"[{stage}] done in {seconds:.0f}s")

    def _poller(self, stage: str, every: float = 15.0) -> Callable[[float], None]:
        last = [0.0]

        def on_poll(elapsed: float) -> None:
            if elapsed - last[0] >= every:
                last[0] = elapsed
                self.log(f"[{stage}] still running ({elapsed:.0f}s)")

        return on_poll

    def _fetch(self, file: OutputFile, dest: Path) -> Path:
        """Download a ComfyUI output next to ``dest``, then rename it (no partial files)."""
        tmp_dir = dest.parent / ".download"
        got = self.client.download(file, tmp_dir)
        got.replace(dest)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return dest

    def _free_vram(self) -> None:
        # Models move to CPU RAM but stay cached, so the next job reloads them quickly.
        try:
            self.client.free(unload_models=True, free_memory=False)
        except Exception as exc:  # not fatal: Blender usually fits next to the models
            self.log(f"[toon] could not free VRAM ({exc}); continuing")

    def _wait_for_kernels(self, timeout: float = 900.0, interval: float = 5.0) -> None:
        """Wait for setup's background Cycles kernel compile, so it is not done twice."""
        log = self.warm_log
        if log is None or not log.is_file() or "WARM_DONE" in log.read_text(errors="replace"):
            return
        self.log("[toon] waiting for the Cycles GPU kernels (first run on this VM only)")
        deadline = time.monotonic() + timeout
        while "WARM_DONE" not in log.read_text(errors="replace"):
            if time.monotonic() >= deadline:
                self.log("[toon] kernel warm-up still running; rendering anyway")
                return
            time.sleep(interval)

    # --- stages ---------------------------------------------------------------------
    def run_images(self, job: Job, force: bool = False, seed: int | None = None) -> list[Path]:
        """Stage 1: Anima Turbo candidates. A new ``seed`` replaces the job's seed."""
        if seed is not None and seed != job.data["seed"]:
            job.data["seed"] = random.randrange(2**32) if seed < 0 else int(seed)
            force = True
        if job.is_done("images") and not force:
            return job.candidates()
        with self._stage(job, "images"):
            wf = workflows.anima_t2i(
                job.data["prompt"],
                job.data["seed"],
                negative=job.data["negative"],
                params=job.image_settings(),
                filename_prefix=f"maker3d/{job.id}/anima",
            )
            self.log(f"[images] Anima Turbo v1.1, seed {job.data['seed']}")
            files = [f for f in self.client.run(wf, timeout=900) if f.filename.endswith(".png")]
            if not files:
                raise PipelineError("Anima returned no images")
            job.candidates_dir.mkdir(exist_ok=True)
            for old in job.candidates():
                old.unlink()
            for i, file in enumerate(files, 1):
                self._fetch(file, job.candidates_dir / f"candidate_{i}.png")
        return job.candidates()

    def pick(self, job: Job, number: int) -> Path:
        """Choose candidate ``number`` (1-based) as the image for the 3D stage."""
        candidates = job.candidates()
        if not 1 <= number <= len(candidates):
            raise PipelineError(f"pick a candidate from 1 to {len(candidates)}, not {number}")
        if job.is_done("pick") and job.data.get("pick") == number:
            return job.image
        with self._stage(job, "pick"):
            shutil.copyfile(candidates[number - 1], job.image)
            job.data["pick"] = number
        return job.image

    def run_model(
        self, job: Job, force: bool = False, settings: Trellis2Settings | None = None
    ) -> Path:
        """Stage 2: BiRefNet mask + TRELLIS.2 GLB from image.png."""
        if settings is not None and asdict(settings) != job.data.get("model_settings"):
            job.data["model_settings"] = asdict(settings)
            force = True
        if job.is_done("model") and not force:
            return job.model
        if not job.is_done("pick"):
            raise PipelineError("pick a candidate image first")
        with self._stage(job, "model"):
            s = job.model_settings()
            name = self.client.upload_image(job.image, subfolder=f"maker3d/{job.id}")
            wf = workflows.trellis2_i2m(
                name, job.data["seed"], settings=s, filename_prefix=f"maker3d/{job.id}/model"
            )
            mask_node = find_node(wf, "mask") if s.remove_background else None
            self.log("[model] BiRefNet + TRELLIS.2 (about 1.5 min on an A100, more on an L4)")
            files = self.client.run(wf, timeout=1800, on_poll=self._poller("model"))
            glbs = [f for f in files if f.filename.lower().endswith(".glb")]
            if not glbs:
                raise PipelineError("TRELLIS.2 wrote no GLB file")
            masks = [f for f in files if f.node_id == mask_node]
            if masks:
                self._fetch(masks[0], job.mask)
            self._fetch(glbs[-1], job.model)
        return job.model

    def run_toon(self, job: Job, force: bool = False, settings: ToonSettings | None = None) -> Path:
        """Stage 3: Blender toon render of model.glb (stills, turntable, toon.blend)."""
        if settings is not None and asdict(settings) != job.data.get("toon_settings"):
            job.data["toon_settings"] = asdict(settings)
            force = True
        if job.is_done("toon") and not force:
            return job.blend
        if not job.is_done("model"):
            raise PipelineError("make the 3D model first")
        with self._stage(job, "toon"):
            self._free_vram()
            self._wait_for_kernels()
            # Render on the VM's disk, then copy the results (Drive is slow for many files).
            work = self.work_dir / job.id / "toon"
            shutil.rmtree(work, ignore_errors=True)
            work.mkdir(parents=True)
            cmd = self.blender_cmd + job.toon_settings().to_args(
                job.model, work, job.image, job.mask
            )
            self.log("[toon] Blender toon render")
            try:
                run_logged(
                    cmd,
                    work / "blender.log",
                    on_line=lambda line: self.log(line) if line.startswith("[toon]") else None,
                    timeout=3600,
                )
            finally:
                if (work / "blender.log").is_file():
                    shutil.copyfile(work / "blender.log", job.dir / "blender.log")
            if not (work / "toon.blend").is_file():
                raise PipelineError("Blender wrote no toon.blend; see blender.log in the job")
            shutil.rmtree(job.stills_dir, ignore_errors=True)
            if (work / "stills").is_dir():
                shutil.copytree(work / "stills", job.stills_dir)
            job.turntable.unlink(missing_ok=True)
            if (work / "turntable.mp4").is_file():
                shutil.copyfile(work / "turntable.mp4", job.turntable)
            tmp = job.dir / "toon.blend.tmp"
            shutil.copyfile(work / "toon.blend", tmp)
            tmp.replace(job.blend)
        return job.blend

    def run_all(self, job: Job, pick: int | None = None) -> Job:
        """Run every stage that is not done yet (resumes an interrupted job).

        ``pick``: candidate number; default is the job's earlier pick, else 1.
        """
        self.run_images(job)
        self.pick(job, pick or job.data.get("pick") or 1)
        self.run_model(job)
        self.run_toon(job)
        return job

    def make_zip(self, job: Job) -> Path:
        """Zip of the deliverables, on the VM's disk (for a download button)."""
        dest = self.work_dir / f"{job.id}.zip"
        dest.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(dest, "w") as z:  # PNG/GLB/MP4/.blend are compressed already
            for path in job.deliverables():
                z.write(path, f"{job.id}/{path.relative_to(job.dir).as_posix()}")
        return dest


# --------------------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------------------
def _resolve(resume: str, jobs_dir: str | None) -> Job:
    if resume == "latest":
        jobs = list_jobs(jobs_dir)
        if not jobs:
            raise SystemExit(f"no jobs in {jobs_dir or default_jobs_dir()}")
        return jobs[0]
    return Job.load(resume)


def main(argv: list[str] | None = None) -> Job:
    p = argparse.ArgumentParser(description="Prompt -> anime image -> 3D model -> toon render")
    p.add_argument("subject", nargs="?", help="character description (Danbooru tags or text)")
    p.add_argument("--resume", metavar="JOB", help="job folder, job.json or 'latest'")
    p.add_argument("--seed", type=int, help="-1 or empty = random")
    p.add_argument("--pick", type=int, help="which candidate to turn into 3D (default 1)")
    p.add_argument("--hands", choices=sorted(HAND_TAGS), default=DEFAULT_HANDS)
    p.add_argument("--no-3d-tags", action="store_true", help="leave out the 3D-friendly tags")
    p.add_argument("--redo", choices=STAGES, help="run this stage again (and the ones after)")
    p.add_argument("--style", choices=("flat", "cel", "cel_rim"))
    p.add_argument("--jobs-dir", help=f"default: {DRIVE_JOBS_DIR} if Drive is mounted")
    p.add_argument("--zip", action="store_true", help="also write a zip of the results")
    args = p.parse_args(argv)
    if bool(args.subject) == bool(args.resume):
        p.error("give a subject or --resume, not both")

    if args.resume:
        job = _resolve(args.resume, args.jobs_dir)  # a different --pick re-runs model + toon
    else:
        job = Job.create(
            args.subject,
            args.seed,
            three_d_friendly=not args.no_3d_tags,
            hands=args.hands,
            jobs_dir=args.jobs_dir,
        )
    if args.style:
        job.data["toon_settings"] = {**job.data["toon_settings"], "style": args.style}
        job.reset_after("model")
    if args.redo:
        job.data["stages"][args.redo] = {"status": "todo"}
        job.reset_after(args.redo)
    print(f"job {job.dir} (next: {job.next_stage()})", flush=True)

    pipe = Pipeline(log=lambda m: print(m, flush=True))
    pipe.run_all(job, pick=args.pick)
    print(f"done: {job.blend}", flush=True)
    if args.zip:
        print(f"zip: {pipe.make_zip(job)}", flush=True)
    return job


if __name__ == "__main__":
    main()
