"""The training-free refiner: Sat2City v2's frozen appearance path on our mesh.

The diffusion call has never run here - this machine is `torch 2.13.0+cpu` - so
what is tested is everything around it, and one thing in particular.

**The frame transform must be exactly invertible.** That is the whole safety
argument of this module. TRELLIS.2 normalises a mesh into [-0.5, 0.5] with a
similarity transform and an axis swap, and because that is invertible, this
stage can *assert* the surface it gets back is the one it sent rather than
trusting a flag. threefiner could not offer that - it ran `clean_mesh`, which
merges vertices - and the assertion is what makes "geometry is measured" a
checkable claim rather than a promise.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from traksha.mesh import structural as S
from traksha.mesh import trellis as T


GSD = 0.5


def scene(h=64, w=64, ground=100.0):
    return np.full((h, w), ground, np.float32)


def block(dsm, r0, c0, r1, c1, height):
    dsm[r0:r1, c0:c1] = dsm.min() + height
    m = np.zeros(dsm.shape, bool)
    m[r0:r1, c0:c1] = True
    return m


def built(h=64, w=64):
    dsm = scene(h, w)
    mask = block(dsm, 16, 16, 44, 44, 20.0)
    dtm = np.full(dsm.shape, 100.0, np.float32)
    b = S.measure(1, mask, dsm, dtm)
    assert b is not None
    return S.build(dsm, (dsm - dtm).astype(np.float32), [b], GSD), [b], mask


# ------------------------------------------------------------- the guarantee
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_the_frame_transform_round_trips_exactly(seed):
    """Invertibility is why the geometry guard can be an assertion."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(400, 3)) * [40, 25, 15] + [2.68e6, 1.24e6, 410.0]
    norm, restore = T.frame(v)
    assert np.abs(restore(norm) - v).max() < 1e-9


def test_the_normalised_mesh_satisfies_the_pipelines_own_assert():
    """TRELLIS.2's preprocess_mesh asserts every vertex is inside [-0.5, 0.5]."""
    rng = np.random.default_rng(7)
    v = rng.normal(size=(200, 3)) * [60, 5, 30]
    norm, _ = T.frame(v)
    assert norm.min() >= -0.5 and norm.max() <= 0.5


def test_the_frame_swaps_z_up_for_y_up():
    """y' = -z and z' = y, which is what the pipeline does to its input."""
    v = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0], [1.0, 0.0, 0.0]])
    norm, _ = T.frame(v)
    # The vertex that was highest in z must now be lowest in y.
    assert norm[1, 1] < norm[0, 1]


def test_a_degenerate_extent_does_not_divide_by_zero():
    v = np.zeros((10, 3))
    norm, restore = T.frame(v)
    assert np.isfinite(norm).all()
    assert np.abs(restore(norm) - v).max() < 1e-9


# -------------------------------------------------------------- extraction
def test_a_building_is_extracted_with_its_own_triangles_renumbered():
    mesh, _, _ = built()
    got = T.extract(mesh, 1)
    assert got is not None
    verts, faces = got
    assert faces.min() >= 0 and faces.max() < len(verts)


def test_extraction_does_not_move_a_vertex():
    """This selects and renumbers. It is not allowed to be a transform."""
    mesh, _, _ = built()
    verts, _ = T.extract(mesh, 1)
    V = np.asarray(mesh["vertices"])
    for v in verts[:50]:
        assert np.isclose(np.abs(V - v).sum(1).min(), 0.0)


def test_an_absent_building_extracts_to_none():
    mesh, _, _ = built()
    assert T.extract(mesh, 999) is None


def test_a_building_below_the_face_floor_is_declined():
    mesh, _, _ = built()
    hit = next(g for g in mesh["groups"] if g[0] == "building_1")
    tiny = dict(mesh, groups=[("building_1", hit[1], 8)])
    assert T.extract(tiny, 1) is None


# ------------------------------------------------------------------ the crop
def test_the_crop_carries_the_footprint_as_alpha():
    """TRELLIS.2's background remover is built for object photos and would cut
    a nadir city crop along whatever contrast it found. Our alpha pre-empts it."""
    mesh, blds, mask = built()
    verts, _ = T.extract(mesh, 1)
    rgb = np.full((64, 64, 3), 128, np.uint8)
    img = T.crop(rgb, verts, GSD, rgb.shape[:2], mask)
    assert img is not None and img.mode == "RGBA"
    assert np.asarray(img)[..., 3].max() == 255


def test_without_a_footprint_the_crop_is_plain_rgb():
    mesh, _, _ = built()
    verts, _ = T.extract(mesh, 1)
    rgb = np.full((64, 64, 3), 128, np.uint8)
    assert T.crop(rgb, verts, GSD, rgb.shape[:2]).mode == "RGB"


def test_a_footprint_too_small_to_condition_on_is_declined():
    mesh, _, _ = built()
    verts, _ = T.extract(mesh, 1)
    rgb = np.full((64, 64, 3), 128, np.uint8)
    tiny = np.zeros((64, 64), bool)
    tiny[30:32, 30:32] = True
    assert T.crop(rgb, verts, GSD, rgb.shape[:2], tiny) is None


def test_the_crop_stays_inside_the_raster():
    """A building on the scene edge must not index outside the image."""
    dsm = scene()
    mask = block(dsm, 0, 0, 24, 24, 20.0)
    dtm = np.full(dsm.shape, 100.0, np.float32)
    b = S.measure(1, mask, dsm, dtm)
    mesh = S.build(dsm, (dsm - dtm).astype(np.float32), [b], GSD)
    verts, _ = T.extract(mesh, 1)
    rgb = np.full((64, 64, 3), 128, np.uint8)
    img = T.crop(rgb, verts, GSD, rgb.shape[:2], mask)
    assert img is not None


# ------------------------------------------------------------- the dry path
def test_preflight_never_raises_and_names_what_is_missing():
    got = T.preflight()
    assert isinstance(got, dict) and "missing" in got
    if not got["ok"]:
        assert got["reason"].startswith("missing ")


def test_a_machine_without_cuda_is_told_that():
    got = T.preflight()
    try:
        import torch

        if not torch.cuda.is_available():
            assert any("CUDA" in m for m in got["missing"])
    except ImportError:
        assert "torch" in got["missing"]


def test_loading_the_pipeline_without_the_dependencies_raises_a_reason():
    with pytest.raises(T.TrellisUnavailable, match="missing"):
        T.load_pipeline()


def test_a_dry_run_selects_the_largest_buildings_and_costs_them(tmp_path):
    mesh, blds, _ = built()
    rgb = np.full((64, 64, 3), 128, np.uint8)
    rec = T.refine(mesh, rgb, GSD, str(tmp_path), buildings=blds,
                   limit=4, dry_run=True)
    assert rec["skipped"] == "dry run"
    assert rec["candidates"] >= 1
    assert len(rec["estimate_s"]) == 2


def test_an_unavailable_machine_records_the_reason_rather_than_raising(tmp_path):
    mesh, blds, _ = built()
    rgb = np.full((64, 64, 3), 128, np.uint8)
    rec = T.refine(mesh, rgb, GSD, str(tmp_path), buildings=blds, limit=2)
    assert "skipped" in rec
    assert rec["refined"] == 0 and rec["attempted"] == 0


def test_the_record_has_the_same_shape_whether_or_not_the_stage_ran(tmp_path):
    """Zero refined is a fact a consumer can read; a missing key is a question."""
    mesh, blds, _ = built()
    rgb = np.full((64, 64, 3), 128, np.uint8)
    rec = T.refine(mesh, rgb, GSD, str(tmp_path), buildings=blds, limit=1,
                   dry_run=True)
    for k in ("refined", "attempted", "candidates", "selected", "buildings"):
        assert k in rec, k


# ---------------------------------------------------------------- the label
def test_the_manifest_says_the_texture_is_synthesised_and_geometry_is_not(tmp_path):
    path = T.write_manifest(str(tmp_path), {"refined": 3})
    with open(path, encoding="utf-8") as fh:
        got = json.load(fh)
    assert got["synthesised"] is True
    assert got["geometry_measured"] is True
    assert "not a photograph" in got["warning"]
    assert "structural.obj are untouched" in got["warning"]


def test_the_record_states_that_nothing_was_trained():
    """The whole point: every module in this path is frozen TRELLIS.2."""
    mesh, blds, _ = built()
    rgb = np.full((64, 64, 3), 128, np.uint8)
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        rec = T.refine(mesh, rgb, GSD, d, buildings=blds, limit=1, dry_run=True)
    assert rec["trained"] is False
    assert rec["checkpoint"] == "microsoft/TRELLIS.2-4B"


def test_the_description_names_which_half_of_sat2city_this_is():
    text = T.describe()
    assert "Sat2City v2" in text and "frozen" in text
    assert "geometry flow" in text


# ------------------------------------------------------------- the GPU floor
# Two hardware floors decide this stage and neither is about speed: flash-attn
# builds only for compute capability 8.0+, and bfloat16 is 8.0+ as well. A
# Tesla P100 - Kaggle's default accelerator - is Pascal at 6.0, two generations
# below both. Catching that in preflight is the difference between a sentence
# and a failed source build twenty minutes in.
class _Props:
    def __init__(self, mem):
        self.total_memory = mem


def _as_gpu(monkeypatch, cap, mem_gb, name="fake"):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i=0: name)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i=0: cap)
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda i=0: _Props(mem_gb * 1e9))


def _blockers(check):
    """The architecture refusals, as opposed to the missing-package ones."""
    return [m for m in check["missing"] if "flash-attn" in m]


@pytest.mark.parametrize("cap,name", [((6, 0), "P100"), ((6, 1), "P40")])
def test_pascal_is_blocked_because_it_has_no_documented_route(monkeypatch, cap, name):
    """TRELLIS.2's xformers fallback is written for V100 and up. Pascal has none."""
    _as_gpu(monkeypatch, cap, 16.0, name)
    got = T.preflight()
    assert _blockers(got), f"{name} at {cap} should be refused outright"
    assert "flash-attn" in _blockers(got)[0]


@pytest.mark.parametrize("cap,name", [((7, 0), "V100"), ((7, 5), "T4")])
def test_volta_and_turing_are_warned_not_blocked(monkeypatch, cap, name):
    """Blocking these was wrong: TRELLIS.2's README names the xformers route."""
    _as_gpu(monkeypatch, cap, 16.0, name)
    got = T.preflight()
    assert not _blockers(got), f"{name} has a documented backend and must not be refused"
    assert any("xformers" in n for n in got["notes"])


def test_the_attention_backend_variable_is_named_and_its_state_reported(monkeypatch):
    _as_gpu(monkeypatch, (7, 5), 15.8, "T4")
    monkeypatch.delenv(T.ATTN_BACKEND_ENV, raising=False)
    assert any("unset" in n for n in T.preflight()["notes"])
    monkeypatch.setenv(T.ATTN_BACKEND_ENV, "xformers")
    assert any("already set" in n for n in T.preflight()["notes"])


def test_a_second_card_is_reported_as_throughput_not_capacity(monkeypatch):
    """Kaggle's "T4 x2" is two 15 GB devices, not one 30 GB device."""
    torch = pytest.importorskip("torch")
    _as_gpu(monkeypatch, (7, 5), 15.8, "T4")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    notes = " ".join(T.preflight()["notes"])
    assert "do not pool" in notes
    assert "one building each" in notes


def test_the_memory_shortfall_says_a_second_card_does_not_help(monkeypatch):
    _as_gpu(monkeypatch, (7, 5), 15.8, "T4")
    notes = " ".join(T.preflight()["notes"])
    assert "second card of the same size does not help" in notes
    assert "no smaller variant" in notes


def test_a_blocked_card_is_not_told_to_try_a_smaller_resolution(monkeypatch):
    """Resolution cannot move an architecture wall, and saying so wastes a day."""
    _as_gpu(monkeypatch, (6, 0), 16.0, "P100")
    got = T.preflight()
    assert not any("resolution 512" in n for n in got["notes"])


def test_a_blocked_card_is_told_what_would_work(monkeypatch):
    _as_gpu(monkeypatch, (6, 0), 16.0, "P100")
    reason = _blockers(T.preflight())[0]
    assert "L4" in reason


@pytest.mark.parametrize("cap,vram", [((8, 0), 40.0), ((8, 9), 23.0), ((9, 0), 80.0)])
def test_ampere_and_newer_clear_the_gpu_gate(monkeypatch, cap, vram):
    _as_gpu(monkeypatch, cap, vram)
    got = T.preflight()
    assert not _blockers(got)
    assert not any("VRAM" in n for n in got["notes"])


def test_a_supported_card_that_is_short_on_memory_is_warned(monkeypatch):
    _as_gpu(monkeypatch, (8, 6), 12.0, "RTX 3060")
    got = T.preflight()
    assert not _blockers(got)
    assert any("VRAM" in n for n in got["notes"])


def test_the_capability_and_family_are_reported(monkeypatch):
    _as_gpu(monkeypatch, (8, 9), 23.0, "L4")
    got = T.preflight()
    assert got["capability"] == "8.9"
    assert "Ada" in got["gpu_family"]
