"""GPU harness: the checks that can only be made where CUDA exists.

Run on the GPU box with:

    pytest tests/test_gpu.py -v -m gpu

Everything here is skipped without a CUDA device, and the skip reason says so.
The point of these tests is not that the GPU is fast; it is that the GPU path
produces the SAME surface as the CPU path. A batching or fp16 change that
quietly alters the depth field would otherwise show up as an unexplained
metrics shift three stages later.
"""
from __future__ import annotations

import numpy as np
import pytest

from unnat.depth.infer import predict_depth


class _DeterministicBackbone:
    """Depth from pixel content alone, so batching cannot change the answer."""

    name = "deterministic"
    native = None
    calls = 0

    def load(self):
        return self

    def infer(self, patch):
        return patch.astype(np.float32).mean(axis=2)

    def infer_batch(self, patches):
        type(self).calls += 1
        return [self.infer(p) for p in patches]


# --------------------------------------------------------------- batching (CPU)
def test_batching_does_not_change_the_mosaic(scene):
    """A batch is a scheduling decision, not a numerical one."""
    one = predict_depth(scene, _DeterministicBackbone(), chip=128, overlap=0.25, batch_size=1)
    four = predict_depth(scene, _DeterministicBackbone(), chip=128, overlap=0.25, batch_size=4)
    assert np.allclose(one.relative, four.relative, atol=1e-6)


def test_batch_path_is_actually_used(scene):
    _DeterministicBackbone.calls = 0
    predict_depth(scene, _DeterministicBackbone(), chip=128, overlap=0.25, batch_size=4)
    assert _DeterministicBackbone.calls > 0, "infer_batch was never called"


# ------------------------------------------------------------------------- gpu
@pytest.mark.gpu
def test_cuda_is_visible_and_reports_memory():
    import torch

    assert torch.cuda.is_available()
    free, total = torch.cuda.mem_get_info()
    assert total > 0 and free > 0
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    print(f"\n  {props.name}  {total / 1024**3:.1f} GB  cc {props.major}.{props.minor}")


@pytest.mark.gpu
def test_device_resolution_prefers_cuda():
    from unnat.depth.backbones.hf import resolve_device, resolve_dtype

    assert resolve_device("auto") == "cuda"
    assert resolve_dtype("auto", "cuda") == "float16"
    assert resolve_dtype("auto", "cpu") == "float32"
    assert resolve_device("cpu") == "cpu"


@pytest.mark.gpu
@pytest.mark.slow
def test_backbone_loads_onto_the_gpu():
    from unnat.depth.backbones import get_backbone

    model = get_backbone("dav2-vits", device="cuda").load()
    assert model.device.startswith("cuda")
    stats = model.stats()
    assert stats["parameters_m"] > 10
    print(f"\n  {model.describe()}  {stats['parameters_m']}M params")


@pytest.mark.gpu
@pytest.mark.slow
def test_gpu_and_cpu_agree_on_the_same_chip():
    """fp16 on GPU must not move the surface more than the calibration residual."""
    from unnat.depth.backbones import get_backbone
    from unnat.eval.synthetic_scene import make_scene

    sc = make_scene(size=518, gsd_m=0.5, seed=5)
    patch = sc.rgb[:518, :518]

    gpu = get_backbone("dav2-vits", device="cuda", dtype="float16").load().infer(patch)
    cpu = get_backbone("dav2-vits", device="cpu", dtype="float32").load().infer(patch)

    gpu_n = (gpu - gpu.mean()) / max(gpu.std(), 1e-6)
    cpu_n = (cpu - cpu.mean()) / max(cpu.std(), 1e-6)
    r = float(np.corrcoef(gpu_n.ravel(), cpu_n.ravel())[0, 1])
    print(f"\n  fp16(gpu) vs fp32(cpu) correlation: {r:.5f}")
    assert r > 0.995, f"fp16 GPU path diverges from fp32 CPU path (r={r:.4f})"


@pytest.mark.gpu
@pytest.mark.slow
def test_gpu_batching_matches_single_chip_inference():
    from unnat.depth.backbones import get_backbone
    from unnat.eval.synthetic_scene import make_scene

    sc = make_scene(size=518, gsd_m=0.5, seed=6)
    patches = [sc.rgb[:518, :518], np.flipud(sc.rgb[:518, :518]).copy()]

    model = get_backbone("dav2-vits", device="cuda").load()
    singles = [model.infer(p) for p in patches]
    batched = model.infer_batch(patches)
    for s, b in zip(singles, batched):
        r = float(np.corrcoef(s.ravel(), b.ravel())[0, 1])
        assert r > 0.999, f"batched inference differs from single (r={r:.5f})"


@pytest.mark.gpu
@pytest.mark.slow
def test_suggested_batch_size_fits_in_memory():
    from unnat.depth.backbones import get_backbone
    from unnat.eval.synthetic_scene import make_scene

    model = get_backbone("dav2-vits", device="cuda").load()
    n = model.suggest_batch_size(518)
    assert n >= 1
    sc = make_scene(size=518, gsd_m=0.5, seed=7)
    patches = [sc.rgb[:518, :518]] * n
    out = model.infer_batch(patches)          # must not OOM
    assert len(out) == n
    print(f"\n  suggested batch {n} at chip 518 - fits")


@pytest.mark.gpu
@pytest.mark.slow
def test_full_pipeline_runs_on_gpu(tmp_path):
    """The whole thing, end to end, on the device the demo will use."""
    from unnat.api.pipeline import run
    from unnat.core.types import Config
    from unnat.dsm.cog import write_cog, write_rgb
    from unnat.eval.synthetic_scene import make_scene

    sc = make_scene(size=512, gsd_m=0.5, seed=8)
    img = str(tmp_path / "scene.tif")
    dtm = str(tmp_path / "dtm.tif")
    dsm = str(tmp_path / "dsm.tif")
    write_rgb(img, sc.rgb, sc.meta)
    write_cog(dtm, sc.dtm_m, sc.meta)
    write_cog(dsm, sc.dsm_m, sc.meta)

    cfg = Config(backbone="dav2-vits", chip=512, dem_source=f"sim:{dtm}", reference=dsm,
                 n_bootstrap=4, extras={"device": "cuda", "batch_size": 2})
    res = run(img, cfg, out_dir=str(tmp_path / "out"))

    assert res.surface is not None
    assert np.isfinite(res.surface.dsm_m).all()
    assert res.metrics["n_px"] > 0
    assert res.anchors_used > 0
    print(f"\n  tier {res.tier.value}  MAE {res.metrics['mae_m']:.2f} m  "
          f"depth {res.timings_s['depth']:.1f}s")
