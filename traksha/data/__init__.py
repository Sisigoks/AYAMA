"""Real-dataset support: discovery and evaluation on imagery we did not render.

`traksha.eval` scores a run against whatever truth it was given. This package is
the counterpart for data that came from somewhere else, and it downloads
nothing - see `datasets.py` for why.
"""
from .datasets import (SceneRef, aggregate, discover, discover_generic,
                       discover_us3d, run_scene)

__all__ = ["SceneRef", "discover", "discover_us3d", "discover_generic",
           "run_scene", "aggregate"]
