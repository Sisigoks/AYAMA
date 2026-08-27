"""Backbone registry. `get_backbone(name)` is the only entry point."""
from __future__ import annotations

from .base import DepthBackbone
from .hf import CHECKPOINTS as _HF_CHECKPOINTS
from .synthetic import SyntheticBackbone

BACKBONES = ("synthetic",) + tuple(_HF_CHECKPOINTS)

# What the UI shows in the backbone dropdown.
LABELS = {
    "dav2-vits": "Depth Anything V2 - ViT-S (fast)",
    "dav2-vitb": "Depth Anything V2 - ViT-B",
    "dav2-vitl": "Depth Anything V2 - ViT-L (primary)",
    "dpt-large": "DPT-Large (comparison backbone)",
    "dpt-hybrid": "DPT-Hybrid MiDaS",
    "synthetic": "Synthetic placeholder (no weights)",
}


def get_backbone(name: str, device: str = "auto", dtype: str = "auto") -> DepthBackbone:
    key = (name or "").strip().lower()
    if key in ("synthetic", "none", "stub"):
        return SyntheticBackbone()
    if key in _HF_CHECKPOINTS:
        from .hf import make

        return make(key, device=device, dtype=dtype)
    raise KeyError(f"unknown backbone '{name}'. Available: {', '.join(BACKBONES)}")
