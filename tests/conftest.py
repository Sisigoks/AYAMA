"""Shared fixtures, backed by real imagery.

This suite used to build its rasters with a renderer. It no longer does, and
that is deliberate: every number this project publishes comes from real imagery,
and a suite that exercises the pipeline on invented pixels is testing a
different pipeline from the one that ships.

The scene is the one bundled with the package - see `ayama/data/sample.py` for
what it is and `ayama/data/fixture/ATTRIBUTION.md` for the swisstopo licence.
It is committed, so a fresh clone runs the whole suite with no network.

It carries **no sun angles**, because none is published. Tests needing shadow
physics pass `sun=(az, el)` to `load_sample_scene`, which states at the call
site that the angle is a test parameter rather than a measurement.
"""
from __future__ import annotations

import pytest

from ayama.data.sample import load_sample_scene  # noqa: F401  (re-exported for tests)


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: loads real model weights")


def has_torch() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def pytest_collection_modifyitems(config, items):
    skip_slow = pytest.mark.skip(reason="torch/transformers not installed")
    for item in items:
        if "slow" in item.keywords and not has_torch():
            item.add_marker(skip_slow)


@pytest.fixture(scope="session")
def real_scene():
    """The bundled crop: 384 px of central Zurich, no sun."""
    return load_sample_scene(size=384)


@pytest.fixture(scope="session")
def scene(real_scene):
    return real_scene.as_scene()
