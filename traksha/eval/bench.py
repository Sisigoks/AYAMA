"""Throughput harness: how fast does this run, on this machine, at this size.

Answers the three questions that decide the demo plan.

  - Which backbone can we afford at full resolution?
  - What batch size fits in this GPU's memory?
  - How long does the 3D view have to wait after the button is pressed?

Reports wall time per stage, chips per second, megapixels per second and peak
VRAM, as JSON and as a markdown table. Numbers from a warmed-up model only: the
first forward pass carries kernel autotuning and weight upload and is timed
separately rather than being allowed to pollute the average.
"""
from __future__ import annotations

import os
import platform
import time
from typing import Callable, Optional, Sequence

import numpy as np

from ..core.ingest import ingest
from ..core.types import Scene
from ..depth.backbones import get_backbone
from ..depth.infer import n_chips, predict_depth


def device_report() -> dict:
    """Everything about the machine that changes a timing number.

    TRAKSHA runs on the CPU only, so this reports cores and thread count rather
    than accelerators - see the note in `traksha.depth.backbones.hf`.
    """
    rep = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import torch

        rep["torch"] = torch.__version__
        rep["threads"] = torch.get_num_threads()
    except ImportError:
        rep["torch"] = None
    try:
        import rasterio

        rep["rasterio"] = rasterio.__version__
        rep["gdal"] = rasterio.__gdal_version__
    except ImportError:
        rep["rasterio"] = None
    return rep


def sample_scene_of(size: int, gsd_m: float = 0.5) -> Scene:
    """The bundled real sample at a requested size, for size-scaling curves."""
    from ..data.sample import load_sample_scene

    return load_sample_scene(size=size).as_scene()


def bench_backbone(
    scene: Scene,
    backbone: str,
    chip: int = 1024,
    overlap: float = 0.25,
    batch_size: int = 1,
    repeats: int = 1,
) -> dict:
    model = get_backbone(backbone)

    t0 = time.time()
    model.load()
    load_s = time.time() - t0

    if batch_size <= 0:
        batch_size = getattr(model, "suggest_batch_size", lambda _c: 1)(chip)

    # Warm-up: one real chip through the real path, timed but not counted.
    warm_patch = scene.rgb[:chip, :chip]
    if warm_patch.shape[0] < chip or warm_patch.shape[1] < chip:
        warm_patch = np.pad(
            warm_patch,
            ((0, max(0, chip - warm_patch.shape[0])), (0, max(0, chip - warm_patch.shape[1])), (0, 0)),
            mode="edge")
    t0 = time.time()
    model.infer(warm_patch)
    warmup_s = time.time() - t0

    chips = n_chips(scene.shape, chip, overlap)
    runs = []
    for _ in range(max(1, repeats)):
        t0 = time.time()
        predict_depth(scene, model, chip=chip, overlap=overlap, batch_size=batch_size)
        runs.append(time.time() - t0)

    wall = float(np.median(runs))
    px = scene.shape[0] * scene.shape[1]
    return {
        "backbone": backbone,
        "image_px": [int(scene.shape[0]), int(scene.shape[1])],
        "chip": chip,
        "overlap": overlap,
        "batch_size": int(batch_size),
        "n_chips": int(chips),
        "load_s": round(load_s, 2),
        "warmup_s": round(warmup_s, 2),
        "wall_s": round(wall, 2),
        "s_per_chip": round(wall / max(chips, 1), 3),
        "chips_per_s": round(chips / max(wall, 1e-6), 2),
        "mpix_per_s": round(px / 1e6 / max(wall, 1e-6), 2),
        "runs_s": [round(r, 2) for r in runs],
        "model": model.stats(),
    }


def _run_case_isolated(kwargs: dict) -> dict:
    """Run one bench case in a child process so a native crash is survivable.

    Torch can segfault outright when a second model is loaded into a warm
    process, and an out-of-memory on a GPU can take the context with it. Neither
    is catchable in-process, so the case runs in a child and a dead child is
    recorded as a result for that cell.
    """
    import concurrent.futures as cf

    image = kwargs.pop("_image", None)
    size = kwargs.pop("_size", 1024)
    try:
        with cf.ProcessPoolExecutor(max_workers=1, max_tasks_per_child=1) as ex:
            return ex.submit(_bench_case_entry, image, size, kwargs).result()
    except Exception as exc:
        name = type(exc).__name__
        detail = "process died (segfault or OOM kill)" if "BrokenProcess" in name else str(exc)
        return {"backbone": kwargs.get("backbone"), "chip": kwargs.get("chip"),
                "batch_size": kwargs.get("batch_size"), "error": f"{name}: {detail}"}


def _bench_case_entry(image, size, kwargs) -> dict:
    """Module-level entry point; must be importable for spawn-based children."""
    from ..core.ingest import ingest

    scene = ingest(image) if image else sample_scene_of(size)
    return bench_backbone(scene, **kwargs)


def sweep(
    image: Optional[str] = None,
    size: int = 1024,
    backbones: Sequence[str] = ("dav2-vits",),
    chips: Sequence[int] = (512, 1024),
    batches: Sequence[int] = (1,),
    overlap: float = 0.25,
    repeats: int = 1,
    on_case: Optional[Callable[[str, int, int], None]] = None,
    isolate: bool = True,
) -> dict:
    results = []
    for bb in backbones:
        for chip in chips:
            for batch in batches:
                if on_case:
                    on_case(bb, chip, batch)
                case = dict(backbone=bb, chip=chip, overlap=overlap, batch_size=batch,
                            repeats=repeats, _image=image, _size=size)
                if isolate:
                    results.append(_run_case_isolated(case))
                    continue
                case.pop("_image"); case.pop("_size")
                try:
                    scene = ingest(image) if image else sample_scene_of(size)
                    results.append(bench_backbone(scene, **case))
                except Exception as exc:                     # OOM is a result, not a crash
                    results.append({"backbone": bb, "chip": chip, "batch_size": batch,
                                    "error": f"{type(exc).__name__}: {exc}"})
    return {
        "environment": device_report(),
        "source": image or f"bundled sample scene, {size}x{size}",
        "results": results,
    }


def format_bench(report: dict) -> str:
    env = report["environment"]
    lines = [
        f"TRAKSHA throughput   {report['source']}",
        f"  {env.get('cpu_count')} cores, torch {env.get('torch')} "
        f"on {env.get('threads')} thread(s)",
        "",
        f"  {'backbone':<12}{'chip':>6}{'batch':>6}{'chips':>7}"
        f"{'wall s':>9}{'s/chip':>9}{'MPix/s':>9}",
        "  " + "-" * 58,
    ]
    for r in report["results"]:
        if "error" in r:
            lines.append(f"  {r['backbone']:<12}{r['chip']:>6}{r['batch_size']:>6}"
                         f"    {r['error'][:44]}")
            continue
        lines.append(
            f"  {r['backbone']:<12}{r['chip']:>6}{r['batch_size']:>6}{r['n_chips']:>7}"
            f"{r['wall_s']:>9.2f}{r['s_per_chip']:>9.3f}{r['mpix_per_s']:>9.2f}"
        )
    return "\n".join(lines)


def save(report: dict, path: str) -> str:
    from ..core.jsonio import save_json

    return save_json(report, path)
