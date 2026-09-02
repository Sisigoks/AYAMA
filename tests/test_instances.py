"""Structural segmentation: the mask generator's arithmetic and its artifact.

The model itself is exercised in one `slow` test, because downloading weights to
assert that a neural network still segments is a test of the network, not of
this code. What is tested here without weights is everything that decides *which*
masks survive and *what* the rest of the pipeline then reads - the part where a
mistake is silent, because a wrong instance map still looks like an instance map.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from traksha.semantics import instances as I
from traksha.semantics import sam2 as S


def make_mask(x0, y0, w, h, shape=(64, 64), iou=0.9, stab=0.95):
    seg = np.zeros(shape, bool)
    seg[y0:y0 + h, x0:x0 + w] = True
    return S.Mask(segmentation=seg, area_px=int(seg.sum()),
                  bbox=(x0, y0, w, h), predicted_iou=iou,
                  stability_score=stab, point=(x0 + w / 2, y0 + h / 2))


# ------------------------------------------------------------- the arithmetic
def test_the_point_grid_covers_the_image_and_stays_inside_it():
    g = S._point_grid(4, 100, 200)
    assert g.shape == (16, 2)
    assert g[:, 0].min() > 0 and g[:, 0].max() < 200
    assert g[:, 1].min() > 0 and g[:, 1].max() < 100


def test_a_bounding_box_is_inclusive_of_its_last_pixel():
    """An off-by-one here shrinks every mask's box and corrupts NMS quietly."""
    m = np.zeros((10, 10), bool)
    m[2:5, 3:8] = True
    assert S._bbox(m) == (3, 2, 5, 3)


def test_an_empty_mask_has_an_empty_box_rather_than_raising():
    assert S._bbox(np.zeros((8, 8), bool)) == (0, 0, 0, 0)


def test_the_stability_score_is_one_for_a_mask_with_a_hard_edge():
    """A confident mask does not change area when the threshold moves."""
    torch = pytest.importorskip("torch")
    logits = torch.full((1, 1, 8, 8), -50.0)
    logits[..., :4, :] = 50.0
    assert float(S._stability_score(logits)[0, 0]) == pytest.approx(1.0)


def test_the_stability_score_falls_for_a_mask_sitting_on_a_gradient():
    torch = pytest.importorskip("torch")
    ramp = torch.linspace(-2.0, 2.0, 64).reshape(1, 1, 8, 8)
    assert float(S._stability_score(ramp)[0, 0]) < 0.9


def test_deduplication_keeps_distinct_objects_and_drops_repeats():
    pytest.importorskip("torchvision")
    seg = S.Sam2Segmenter()
    a = make_mask(0, 0, 20, 20)
    a_again = make_mask(0, 0, 20, 20, iou=0.8)          # the same object, twice
    b = make_mask(40, 40, 20, 20)                       # a different one
    kept = seg._deduplicate([a, a_again, b])
    assert len(kept) == 2, "a duplicate survived, or a distinct object was lost"


def test_deduplication_returns_largest_first():
    pytest.importorskip("torchvision")
    seg = S.Sam2Segmenter()
    kept = seg._deduplicate([make_mask(0, 0, 8, 8), make_mask(20, 20, 30, 30)])
    assert [m.area_px for m in kept] == sorted([m.area_px for m in kept], reverse=True)


def test_an_unknown_variant_names_the_ones_that_exist():
    with pytest.raises(S.Sam2Unavailable, match="Available"):
        _ = S.Sam2Segmenter(variant="sam2-enormous").checkpoint


# ---------------------------------------------------------------- the artifact
def test_instance_ids_are_unique_and_start_at_one():
    f = I.from_masks([make_mask(0, 0, 10, 10), make_mask(30, 30, 20, 20)], (64, 64))
    ids = [r["id"] for r in f.records]
    assert ids == sorted(set(ids)) and ids[0] == 1
    assert f.count == 2


def test_zero_means_no_instance():
    f = I.from_masks([make_mask(0, 0, 10, 10)], (64, 64))
    assert f.instance_map[40, 40] == I.UNASSIGNED
    assert not f.mask(I.UNASSIGNED).all()


def test_a_small_mask_inside_a_large_one_survives():
    """SAM proposes nested masks: a roof section inside a roof inside a block.

    Painting the large one last would erase every small instance behind it, and
    the finer structure is exactly what a building-scale reconstruction needs.
    """
    big = make_mask(0, 0, 40, 40)
    small = make_mask(10, 10, 8, 8)
    f = I.from_masks([small, big], (64, 64))          # deliberately wrong order in
    assert f.instance_map[14, 14] != f.instance_map[35, 35]
    assert f.records[0]["area_px"] > f.records[1]["area_px"], "not painted largest first"
    assert f.records[1]["visible_px"] == 64, "the nested instance was overwritten"


def test_visible_area_is_recorded_when_an_instance_is_covered_over():
    """An instance with no pixels left must not promise a mask nobody can index."""
    big = make_mask(0, 0, 40, 40)
    over = make_mask(0, 0, 40, 40)                    # identical, painted after
    f = I.from_masks([big, over], (64, 64))
    assert f.records[0]["visible_px"] == 0
    assert f.records[1]["visible_px"] == 1600


def test_boundaries_land_where_the_instance_changes():
    f = I.from_masks([make_mask(10, 10, 20, 20)], (64, 64))
    assert f.boundary[10, 15] and f.boundary[9, 15]   # both sides of the edge
    assert not f.boundary[0, 0] and not f.boundary[20, 20]


def test_confidence_is_carried_per_pixel_and_stays_in_range():
    f = I.from_masks([make_mask(0, 0, 10, 10, iou=0.8, stab=0.5)], (64, 64))
    assert f.confidence[5, 5] == pytest.approx(0.4)
    assert f.confidence[40, 40] == 0.0
    assert f.confidence.min() >= 0.0 and f.confidence.max() <= 1.0


def test_masks_of_the_wrong_shape_are_refused_rather_than_broadcast():
    odd = make_mask(0, 0, 4, 4, shape=(32, 32))
    f = I.from_masks([odd, make_mask(0, 0, 10, 10)], (64, 64))
    assert f.count == 1, "a mask from a different image was painted in"


def test_coverage_reports_what_fraction_carries_an_instance():
    f = I.from_masks([make_mask(0, 0, 32, 32)], (64, 64))
    assert f.coverage == pytest.approx(0.25)


def test_a_skipped_stage_still_produces_a_readable_artifact():
    """Every consumer would otherwise grow a `None` branch and lose the reason."""
    f = I.empty((16, 16), "no network")
    assert f.count == 0 and f.coverage == 0.0
    assert f.instance_map.shape == (16, 16)
    assert "no network" in f.provenance["skipped"]


def test_the_artifact_writes_metadata_that_reproduces_the_run(tmp_path):
    f = I.from_masks([make_mask(0, 0, 10, 10)], (64, 64),
                     provenance={"model": "sam2-tiny", "points_per_side": 16})
    art = f.save(str(tmp_path))
    assert os.path.exists(art["metadata"])
    meta = json.load(open(art["metadata"], encoding="utf-8"))
    assert meta["schema"] == I.SCHEMA_VERSION
    assert meta["instances"] == 1
    assert meta["provenance"]["model"] == "sam2-tiny"
    assert meta["provenance"]["written_utc"].endswith("Z")
    assert meta["records"][0]["bbox"] == [0, 0, 10, 10]


# ------------------------------------------------------------------ the model
@pytest.mark.slow
def test_sam2_really_segments_the_bundled_scene():
    """One end-to-end pass, on the real weights. Marked slow: it downloads."""
    from traksha.data.sample import load_sample_scene

    scene = load_sample_scene(size=256)
    seg = S.Sam2Segmenter(points_per_side=6, points_per_batch=6)
    try:
        masks = seg.generate(scene.rgb)
    except S.Sam2Unavailable as exc:
        pytest.skip(f"SAM 2 unavailable: {exc}")
    finally:
        seg.unload()

    assert masks, "the generator returned nothing on a real scene"
    f = I.from_masks(masks, scene.rgb.shape[:2])
    assert f.count == len(masks)
    assert 0.0 < f.coverage <= 1.0
    assert f.instance_map.max() == f.count
