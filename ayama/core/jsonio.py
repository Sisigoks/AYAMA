"""JSON that a browser will actually parse.

Python's `json.dump` emits bare `NaN` and `Infinity` for non-finite floats. That
is valid Python and invalid JSON: `JSON.parse` rejects it outright, so a single
un-computed metric anywhere in a results file renders the whole file unreadable
to every browser and every strict parser. Metrics legitimately come back
non-finite - delta1 with no valid pixels, ECE with too few samples - so this is
a case that happens on ordinary runs, not an edge case.

Non-finite values become `null`, which is what "no value" means in JSON, and
`allow_nan=False` makes any future leak an exception instead of a silently
broken artifact.
"""
from __future__ import annotations

import json
import math
import os
from typing import Any

import numpy as np


def json_safe(obj: Any) -> Any:
    """Recursively convert an object into something json.dump can emit strictly."""
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, (np.integer, int)) and not isinstance(obj, bool):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if obj is None or isinstance(obj, str):
        return obj
    return str(obj)


def dumps(obj: Any, indent: int = 2) -> str:
    return json.dumps(json_safe(obj), indent=indent, allow_nan=False)


def save_json(obj: Any, path: str, indent: int = 2) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(dumps(obj, indent=indent))
    return path
