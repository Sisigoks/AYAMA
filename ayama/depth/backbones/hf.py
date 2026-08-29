"""Hugging Face depth adapters: Depth Anything V2 and DPT.

Both models are served through transformers' AutoModelForDepthEstimation, so a
single implementation covers them and the only difference is the checkpoint id.

Deliberate choice: the RELATIVE Depth Anything V2 checkpoints, not the metric
ones. The metric variants carry a scale prior fitted to ground-level outdoor
scenes, which is actively wrong for nadir imagery. We want the clean unitless
surface and our own calibration on top of it (see ayama.chhaya).

Inference runs on the CPU. That is a measured decision rather than a
limitation nobody got round to lifting: an order of magnitude more backbone
moved recovered relief from 0.05 m to 0.17 m against a true 14.4 m, because the
bottleneck is metric scale rather than perception (README section 3.2). The
thing that actually recovers structure - a scale fitted once over a dataset,
`ayama.learn` - costs milliseconds. Carrying device selection, autocast and VRAM
budgeting for a speed-up that changes no result is upkeep with no return.
"""
from __future__ import annotations


import numpy as np

from .base import DepthBackbone

CHECKPOINTS = {
    "dav2-vits": "depth-anything/Depth-Anything-V2-Small-hf",
    "dav2-vitb": "depth-anything/Depth-Anything-V2-Base-hf",
    "dav2-vitl": "depth-anything/Depth-Anything-V2-Large-hf",
    "dpt-large": "Intel/dpt-large",
    "dpt-hybrid": "Intel/dpt-hybrid-midas",
}

# Rough activation cost per chip at 518 px, in MB of VRAM, measured shape-wise
# rather than empirically; used only to pick a safe default batch size.
_ACTIVATION_MB = {"dav2-vits": 260, "dav2-vitb": 620, "dav2-vitl": 1400,
                  "dpt-large": 1500, "dpt-hybrid": 700}


class HFDepthBackbone(DepthBackbone):
    """A frozen Hugging Face depth checkpoint, run on the CPU.

    ĀYĀMA is CPU-only by design, not by accident. The one component that could
    profit from a GPU is inference, and measuring it showed the profit does not
    reach the result: an order of magnitude more backbone moved recovered relief
    from 0.05 m to 0.17 m against a true 14.4 m, because the bottleneck is the
    metric scale, not perception. What actually fixed it - a scale fitted once
    over a dataset, `ayama.learn` - is a handful of dot products. So the device
    machinery is gone rather than left as an unused option that has to be kept
    working and honestly documented.
    """

    def __init__(self, name: str, checkpoint: str, native: int = 518,
                 dtype: str = "float32"):
        self.name = name
        self.checkpoint = checkpoint
        self.native = native
        self._model = None
        self._processor = None
        self.device = "cpu"
        self.dtype_name = "float32"

    # ------------------------------------------------------------------ load
    def load(self) -> "HFDepthBackbone":
        if self._model is not None:
            return self
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                f"backbone '{self.name}' needs torch + transformers.\n"
                "  pip install -r requirements-torch.txt\n"
                "There is no weightless fallback: a placeholder that fabricates "
                "a depth field is how an invented number reaches a results table."
            ) from exc


        self._processor = AutoImageProcessor.from_pretrained(self.checkpoint)
        self._model = AutoModelForDepthEstimation.from_pretrained(self.checkpoint)
        self._model.to(self.device).eval()
        torch.set_grad_enabled(False)
        return self

    # ----------------------------------------------------------------- infer
    def infer(self, patch: np.ndarray) -> np.ndarray:
        return self.infer_batch([patch])[0]

    def infer_batch(self, patches: list) -> list:
        import torch

        if self._model is None:
            self.load()
        if not patches:
            return []
        sizes = [(p.shape[0], p.shape[1]) for p in patches]
        inputs = self._processor(images=list(patches), return_tensors="pt")
        with torch.no_grad():
            out = self._model(**inputs).predicted_depth
        out = out.float()
        if out.ndim == 3:
            out = out.unsqueeze(1)

        results = []
        for i, (h, w) in enumerate(sizes):
            r = torch.nn.functional.interpolate(
                out[i:i + 1], size=(h, w), mode="bicubic", align_corners=False
            )
            results.append(r[0, 0].detach().cpu().numpy().astype(np.float32))
        return results

    # ----------------------------------------------------------------- meta
    def describe(self) -> str:
        return f"{self.name} ({self.checkpoint}) on {self.device}/{self.dtype_name}"

    def stats(self) -> dict:
        d = {"name": self.name, "checkpoint": self.checkpoint,
             "device": self.device, "dtype": self.dtype_name}
        if self._model is not None:
            d["parameters_m"] = round(
                sum(p.numel() for p in self._model.parameters()) / 1e6, 1)
        return d

    def suggest_batch_size(self, chip: int) -> int:
        """One. Batching a transformer on CPU rarely pays and can thrash cache."""
        return 1


def make(name: str) -> HFDepthBackbone:
    if name not in CHECKPOINTS:
        raise KeyError(name)
    native = 518 if name.startswith("dav2") else 384
    return HFDepthBackbone(name, CHECKPOINTS[name], native=native)
