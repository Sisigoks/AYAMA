"""Live progress for long runs.

A 2048 px ViT-L study is minutes of silence per scene without this. Silence
during a demo is indistinguishable from a hang, and on a rented GPU it is
indistinguishable from money being wasted, so every stage that can take more
than a few seconds reports where it is, how fast it is going, and when it will
be done.

Three rendering modes, chosen automatically:

  rich   an interactive terminal. One line, rewritten in place, with a bar,
         rate, ETA and live VRAM.
  plain  a notebook, a CI log, or a redirect to a file. Carriage returns would
         produce thousands of unreadable lines, so it prints a timestamped line
         at most every `plain_interval` seconds plus one on completion.
  none   quiet.

Nesting matters for reading it: the study knows it is on scene 2 of 3, the
pipeline knows it is in the depth stage, and the tiler knows it is on chip 24
of 36. The line shows all three at once, because "24/36" alone does not tell
anyone how much of the run is left.
"""
from __future__ import annotations

import os
import shutil
import sys
import time
from typing import Optional

_BAR_FULL = "█"     # █
_BAR_EMPTY = "░"    # ░
_BAR_L = "▏"        # ▏
_BAR_R = "▕"        # ▕


def supports_unicode(stream) -> bool:
    enc = (getattr(stream, "encoding", None) or "").lower()
    return "utf" in enc


def gpu_stats() -> Optional[dict]:
    """Live VRAM, or None when there is no CUDA device.

    Reads the driver's free/total rather than torch's allocator counters: the
    number that matters when choosing a batch size is what the card actually has
    left, including whatever else is resident on it.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free, total = torch.cuda.mem_get_info()
        return {
            "name": torch.cuda.get_device_properties(torch.cuda.current_device()).name,
            "used_gb": (total - free) / 1024 ** 3,
            "total_gb": total / 1024 ** 3,
            "reserved_gb": torch.cuda.memory_reserved() / 1024 ** 3,
        }
    except Exception:
        return None


def _fmt_dur(seconds: float) -> str:
    if seconds is None or seconds != seconds or seconds < 0:
        return "--"
    seconds = float(seconds)
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"
    return f"{int(seconds // 3600)}h{int((seconds % 3600) // 60):02d}m"


class Task:
    """One unit of work that can report a fraction of itself done."""

    def __init__(self, live: "Live", name: str, total: Optional[int], unit: str):
        self.live = live
        self.name = name
        self.total = total
        self.unit = unit
        self.done = 0
        self.detail = ""
        self.started = time.time()
        self.finished: Optional[float] = None

    # -- progress -----------------------------------------------------------
    def advance(self, n: int = 1, detail: str = "") -> None:
        self.done += n
        if detail:
            self.detail = detail
        self.live._render()

    def set(self, done: int, total: Optional[int] = None, detail: str = "") -> None:
        self.done = done
        if total is not None:
            self.total = total
        if detail:
            self.detail = detail
        self.live._render()

    def note(self, detail: str) -> None:
        self.detail = detail
        self.live._render()

    # -- rendering ----------------------------------------------------------
    @property
    def elapsed(self) -> float:
        return (self.finished or time.time()) - self.started

    @property
    def rate(self) -> Optional[float]:
        e = self.elapsed
        return (self.done / e) if (e > 0.2 and self.done > 0) else None

    @property
    def eta(self) -> Optional[float]:
        if not self.total or not self.rate or self.done >= self.total:
            return None
        return (self.total - self.done) / self.rate

    def describe(self, with_name: bool = True) -> str:
        bits = [self.name] if with_name else []
        if self.total:
            bits.append(f"{self.done}/{self.total} {self.unit}".strip())
        elif self.done:
            bits.append(f"{self.done} {self.unit}".strip())
        if self.detail:
            bits.append(self.detail)
        return "  ".join(bits)


class Live:
    """Renders the innermost active task, with its ancestors as context."""

    def __init__(self, mode: str = "auto", stream=None, min_interval: float = 0.12,
                 plain_interval: float = 20.0, show_gpu: bool = True):
        self.stream = stream or sys.stdout
        self.mode = self._resolve(mode)
        self.min_interval = float(min_interval)
        self.plain_interval = float(plain_interval)
        self.show_gpu = show_gpu
        self.stack: list[Task] = []
        self._last_render = 0.0
        self._last_gpu_poll = 0.0
        self._gpu: Optional[dict] = None
        self._dirty = False
        self._unicode = supports_unicode(self.stream)

    def _resolve(self, mode: str) -> str:
        if mode and mode != "auto":
            return mode
        if os.environ.get("UNNAT_PROGRESS"):
            return os.environ["UNNAT_PROGRESS"]
        try:
            return "rich" if self.stream.isatty() else "plain"
        except Exception:
            return "plain"

    # -- task lifecycle -----------------------------------------------------
    def task(self, name: str, total: Optional[int] = None, unit: str = "") -> "TaskCtx":
        return TaskCtx(self, name, total, unit)

    def _push(self, task: Task) -> None:
        self.stack.append(task)
        self._last_render = 0.0        # always show a new task immediately
        self._render(force=True)

    def _pop(self, task: Task, summary: str = "") -> None:
        task.finished = time.time()
        if task in self.stack:
            self.stack.remove(task)
        if self.mode == "none":
            return
        body = summary or task.describe(with_name=False) or "done"
        line = f"  {task.name:<14} {body}   [{_fmt_dur(task.elapsed)}]"
        self._clear_line()
        print(line, file=self.stream, flush=True)
        self._last_render = 0.0
        self._render(force=True)

    # -- rendering ----------------------------------------------------------
    def _poll_gpu(self) -> None:
        now = time.time()
        if not self.show_gpu or now - self._last_gpu_poll < 1.0:
            return
        self._last_gpu_poll = now
        self._gpu = gpu_stats()

    def _bar(self, frac: float, width: int = 18) -> str:
        filled = int(round(frac * width))
        if self._unicode:
            return f"{_BAR_R}{_BAR_FULL * filled}{_BAR_EMPTY * (width - filled)}{_BAR_L}"
        return f"[{'#' * filled}{'-' * (width - filled)}]"

    def _line(self) -> str:
        if not self.stack:
            return ""
        head = self.stack[-1]
        context = " ".join(f"[{t.name} {t.done}/{t.total}]"
                           for t in self.stack[:-1] if t.total)
        parts = []
        if context:
            parts.append(context)
        parts.append(head.name)
        if head.total:
            frac = min(1.0, head.done / max(head.total, 1))
            parts.append(self._bar(frac))
            parts.append(f"{head.done}/{head.total} {head.unit}".strip())
        elif head.done:
            parts.append(f"{head.done} {head.unit}".strip())
        if head.detail:
            parts.append(head.detail)
        if head.rate:
            unit = head.unit or "it"
            parts.append(f"{head.rate:.2f} {unit}/s" if head.rate < 100
                         else f"{head.rate:.0f} {unit}/s")
        if head.eta:
            parts.append(f"ETA {_fmt_dur(head.eta)}")
        self._poll_gpu()
        if self._gpu:
            parts.append(f"VRAM {self._gpu['used_gb']:.1f}/{self._gpu['total_gb']:.1f} GB")
        return "  ".join(parts)

    def _clear_line(self) -> None:
        if self.mode != "rich" or not self._dirty:
            return
        width = shutil.get_terminal_size((100, 24)).columns
        self.stream.write("\r" + " " * (width - 1) + "\r")
        self.stream.flush()
        self._dirty = False

    def _render(self, force: bool = False) -> None:
        if self.mode == "none" or not self.stack:
            return
        now = time.time()

        if self.mode == "rich":
            if not force and now - self._last_render < self.min_interval:
                return
            self._last_render = now
            width = shutil.get_terminal_size((100, 24)).columns
            line = ("  " + self._line())[: max(20, width - 2)]
            self.stream.write("\r" + line.ljust(width - 2)[: width - 2])
            self.stream.flush()
            self._dirty = True
            return

        # plain: a timestamped line at a readable cadence, never a carriage return
        if not force and now - self._last_render < self.plain_interval:
            return
        self._last_render = now
        head = self.stack[-1]
        pct = f" ({100 * head.done / head.total:.0f}%)" if head.total else ""
        print(f"  [{time.strftime('%H:%M:%S')}] {self._line()}{pct}",
              file=self.stream, flush=True)

    # -- convenience --------------------------------------------------------
    def log(self, message: str) -> None:
        """Print a line without fighting the live one."""
        if self.mode == "none":
            return
        self._clear_line()
        print(message, file=self.stream, flush=True)
        self._render(force=True)

    def banner(self) -> str:
        g = gpu_stats()
        if g:
            return f"{g['name']}  {g['total_gb']:.1f} GB VRAM"
        return "CPU only"


class TaskCtx:
    """Context manager wrapper so a stage always pops, even on an exception."""

    def __init__(self, live: Live, name: str, total: Optional[int], unit: str):
        self.live = live
        self.task = Task(live, name, total, unit)
        self.summary = ""

    def __enter__(self) -> Task:
        self.live._push(self.task)
        return self.task

    def done(self, summary: str) -> None:
        self.summary = summary

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.live._pop(self.task, self.summary or (self.task.detail if exc is None
                                                   else f"FAILED: {exc}"))
        return False


# A module-level default so callers that do not thread one through still work.
_default: Optional[Live] = None


def get_live(mode: str = "auto") -> Live:
    global _default
    if _default is None:
        _default = Live(mode=mode)
    return _default


def set_live(live: Optional[Live]) -> None:
    global _default
    _default = live
