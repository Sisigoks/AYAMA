"""Shared fixtures and the GPU marker.

GPU tests are skipped unless CUDA is actually present, so the same suite runs
on a laptop and on the GPU box and only reports what it really checked. Nothing
is silently passed over: a skipped GPU test says why.
"""
from __future__ import annotations

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requires a working CUDA device")
    config.addinivalue_line("markers", "slow: loads real model weights")


def has_cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def has_torch() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def pytest_collection_modifyitems(config, items):
    skip_gpu = pytest.mark.skip(reason="no CUDA device available")
    skip_slow = pytest.mark.skip(reason="torch/transformers not installed")
    for item in items:
        if "gpu" in item.keywords and not has_cuda():
            item.add_marker(skip_gpu)
        if "slow" in item.keywords and not has_torch():
            item.add_marker(skip_slow)


@pytest.fixture(scope="session")
def synthetic_scene():
    """A small deterministic scene with known ground truth."""
    from unnat.eval.synthetic_scene import make_scene

    return make_scene(size=384, gsd_m=0.5, seed=17)


@pytest.fixture(scope="session")
def scene(synthetic_scene):
    from unnat.core.types import Scene

    return Scene(rgb=synthetic_scene.rgb, meta=synthetic_scene.meta, raw_dtype="uint8")
