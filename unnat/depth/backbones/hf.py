"""Hugging Face depth adapters: Depth Anything V2 and DPT.

Both models are served through transformers' AutoModelForDepthEstimation, so a
single implementation covers them and the only difference is the checkpoint id.

Deliberate choice: the RELATIVE Depth Anything V2 checkpoints, not the metric
ones. The metric variants carry a scale prior fitted to ground-level outdoor
scenes, which is actively wrong for nadir imagery. We want the clean unitless
surface and our own calibration on top of it (see unnat.chhaya).

GPU notes, because this is where the throughput is:
  - chips are batched, so a 4k tile is a handful of forward passes and not
    thirty-six;
  - autocast to fp16 on CUDA, which is roughly a 2x win on this workload and
    changes the depth field by far less than the calibration residual;
  - the batch size that fits is a function of chip size and VRAM, so
    `suggest_batch_size` estimates it rather than making the user guess.
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
    def __init__(self, name: str, checkpoint: str, native: int = 518,
                 device: str = "auto", dtype: str = "auto"):
        self.name = name
        self.checkpoint = checkpoint
        self.native = native
        self._device_pref = device
        self._dtype_pref = dtype
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
                "Until then run with --backbone synthetic to exercise the pipeline."
            ) from exc

        self.device = resolve_device(self._device_pref)
        self.dtype_name = resolve_dtype(self._dtype_pref, self.device)

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
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        use_amp = self.dtype_name == "float16" and self.device.startswith("cuda")
        with torch.no_grad():
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out = self._model(**inputs).predicted_depth
            else:
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
        """Largest batch that should fit, or 1 on CPU where batching rarely pays."""
        if not self.device.startswith("cuda"):
            return 1
        import torch

        free, _total = torch.cuda.mem_get_info()
        budget_mb = free / (1024 ** 2) * 0.6          # leave headroom
        per_chip = _ACTIVATION_MB.get(self.name, 800) * (max(chip, 1) / 518.0) ** 2
        if self.dtype_name == "float16":
            per_chip *= 0.55
        return int(np.clip(budget_mb // max(per_chip, 1.0), 1, 16))


def resolve_device(pref: str = "auto") -> str:
    import torch

    if pref and pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(pref: str, device: str) -> str:
    """fp16 on CUDA by default; fp32 everywhere else, where fp16 is slower."""
    if pref and pref != "auto":
        return pref
    return "float16" if device.startswith("cuda") else "float32"


def make(name: str, device: str = "auto", dtype: str = "auto") -> HFDepthBackbone:
    if name not in CHECKPOINTS:
        raise KeyError(name)
    native = 518 if name.startswith("dav2") else 384
    return HFDepthBackbone(name, CHECKPOINTS[name], native=native, device=device, dtype=dtype)
