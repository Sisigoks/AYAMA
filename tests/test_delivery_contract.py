"""The delivery benchmark is evidence, so its shape and its honesty are a contract.

`results/DELIVERY.md` is rendered from `results/delivery.json`, and the README
quotes both. These tests run a miniature benchmark and assert the keys exist,
the numbers are self-consistent, and - the part that matters - that the sweeps
cannot report a saving which is really a deleted measurement.

That last one is not hypothetical. The first version of the quantisation sweep
stepped by fractions of the mean sigma and announced a 99.8% saving on a layer
it had flattened to a constant, and the second version quantised in value space
and produced a smaller file at 16 bits than at 12. Both are pinned here.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("rasterio")

from ayama.eval import delivery as D  # noqa: E402


@pytest.fixture(scope="module")
def mini(tmp_path_factory):
    """A small run and its benchmark. Sweeps trimmed so the suite stays fast."""
    from ayama.core.types import SceneMeta
    from ayama.dsm.cog import write_cog, write_rgb

    d = tmp_path_factory.mktemp("run")
    h = w = 128
    rr, cc = np.mgrid[0:h, 0:w]
    dtm = 400.0 + 0.05 * rr + 0.02 * cc
    ndsm = np.zeros((h, w), np.float32)
    ndsm[30:60, 30:60] = 9.0
    dsm = (dtm + ndsm).astype(np.float32)
    sem = np.zeros((h, w), np.float32)
    sem[30:60, 30:60] = 2
    meta = SceneMeta(crs="EPSG:32644", transform=(0.5, 0, 5e5, 0, -0.5, 2e6), gsd_m=0.5)
    write_cog(str(d / "dsm.tif"), dsm, meta)
    write_cog(str(d / "ndsm.tif"), ndsm, meta)
    # Not a constant: a real sigma field varies with anchor density, and a
    # constant one is a degenerate case the sweep now flags separately.
    sigma = (3.0 + 0.4 * np.sin(rr / 9.0) * np.cos(cc / 7.0)).astype(np.float32)
    write_cog(str(d / "sigma.tif"), sigma, meta)
    write_cog(str(d / "error.tif"), (dsm * 0.01).astype(np.float32), meta)
    write_cog(str(d / "sem.tif"), sem, meta, dtype="uint8", nodata=255)
    write_rgb(str(d / "texture.jpg"), np.full((h, w, 3), 128, np.uint8))

    out = str(tmp_path_factory.mktemp("delivery"))
    rep = D.run_delivery(str(d), out, tile=64, tiles=(64, 128), obj_strides=(2, 4),
                         repeats=1, work_dir=out)
    return rep


# ------------------------------------------------------------------- shape
def test_report_carries_every_top_level_block(mini):
    for key in ("ayama_delivery_version", "generated_utc", "environment", "scene",
                "config", "build", "encode", "stages", "tile_sweep", "obj_sweep",
                "quantisation", "roundtrip", "payload", "viewer", "wall_s"):
        assert key in mini, f"delivery.json is missing '{key}'"


def test_report_is_strict_json(mini, tmp_path):
    """A bare NaN parses in Python and breaks every strict reader downstream."""
    from ayama.core.jsonio import dumps

    import json

    json.loads(dumps(mini))


def test_markdown_renders_from_the_json(mini, tmp_path):
    path = D.write_report(mini, str(tmp_path / "DELIVERY.md"))
    text = open(path, encoding="utf-8").read()
    for heading in ("# AYAMA delivery benchmark", "## Headline", "## Encoding throughput",
                    "## Tile size", "## Mesh decimation", "## What full precision costs",
                    "## The surface survives the trip", "## Payload", "## Viewer CPU"):
        assert heading in text, f"report lost the '{heading}' section"
    assert "GPU rasterisation is not measured" in text


# ------------------------------------------------------------ self-consistency
def test_build_timings_are_comparable(mini):
    """`obj_s` is a difference, so both builds must have been timed alike.

    An earlier version timed one build beside the results and the other in the
    system temp directory, which on a machine with two disks made their
    difference meaningless - and negative on a fast enough scratch drive.
    """
    b = mini["build"]
    assert b["full_s"] >= b["tiles_only_s"] > 0
    assert b["obj_s"] == pytest.approx(b["full_s"] - b["tiles_only_s"], abs=0.02)
    assert mini["config"]["timed_in"], "the report must say where it timed"


def test_payload_adds_up(mini):
    p = mini["payload"]
    assert p["total_bytes"] >= p["tiles_bytes"] + p["mesh_bytes"] - 4096
    assert sum(p["by_layer"].values()) == p["tiles_bytes"]
    assert 0 < p["first_paint_bytes"] <= p["tiles_bytes"]


def test_coarser_lods_are_smaller(mini):
    by_lod = list(mini["payload"]["by_lod"].values())
    assert by_lod == sorted(by_lod, reverse=True), "a coarser LOD weighed more"


def test_every_layer_survives_the_round_trip(mini):
    assert mini["roundtrip"], "nothing was checked"
    bad = [r for r in mini["roundtrip"] if not r["within_half_a_step"]]
    assert not bad, f"delivery altered the surface: {bad}"


# ------------------------------------------------- the sweeps cannot lie
def test_fewer_bits_always_costs_more_error(mini):
    """Error must grow as bits are taken away. True at any image size."""
    by_layer: dict = {}
    for r in mini["quantisation"]:
        by_layer.setdefault(r["layer"], []).append(r)
    for layer, rows in by_layer.items():
        rows = sorted(rows, key=lambda r: -r["bits"])
        errs = [r["max_error_m"] for r in rows]
        assert errs == sorted(errs), f"{layer}: fewer bits produced a smaller error"


REAL_RUN = Path(__file__).resolve().parents[1] / "results" / "seed7" / "run"


@pytest.mark.skipif(not (REAL_RUN / "dsm.tif").exists(),
                    reason="results/seed7/run not present")
def test_fewer_bits_is_never_bigger_on_a_real_run():
    """Monotonicity in bytes, asserted where the claim is actually made.

    This caught a real bug: quantising in value space and re-encoding left the
    low byte noisy, and 12-bit came out larger than 16-bit. Keeping the top N
    bits of the code and zeroing the rest is what a narrower field really does.

    It runs against the real run rather than the fixture because below a few
    tens of kB a PNG is mostly header and filter choice - four bit depths of a
    128 px layer came out 174, 172, 177 and 171 bytes, which measures nothing.
    That threshold is what the `meaningful` flag records.
    """
    from ayama.mesh.build import load_run

    rows = D.quantisation_sweep(load_run(str(REAL_RUN)))
    by_layer: dict = {}
    for r in rows:
        if r.get("meaningful") and not r.get("degenerate"):
            by_layer.setdefault(r["layer"], []).append(r)
    assert by_layer, "no layer was large enough for the byte count to mean anything"
    for layer, rs in by_layer.items():
        rs = sorted(rs, key=lambda r: -r["bits"])
        sizes = [r["bytes"] for r in rs]
        assert sizes == sorted(sizes, reverse=True), (
            f"{layer}: fewer bits produced a larger file: "
            f"{[(r['bits'], r['kb']) for r in rs]}")


def test_a_saving_that_destroys_the_layer_is_never_recommended(mini):
    """The bug this whole test file exists for.

    Stepping by sigma flattened a 0.28 m layer to a constant and called the
    resulting 99.8% byte saving a win. Any recommended variant must still
    resolve its layer to a small fraction of that layer's own range.
    """
    rec = [r for r in mini["quantisation"] if r.get("recommended")]
    assert rec, "no precision was recommended for any layer"
    for r in rec:
        assert not r["degenerate"], f"{r['layer']} never varies; nothing to recommend"
        assert r["error_over_span"] <= 1e-3, (
            f"{r['layer']} at {r['bits']} bits loses "
            f"{100 * r['error_over_span']:.1f}% of its own range")
        assert r["max_error_m"] < r["span_m"], f"{r['layer']} was flattened"


def test_quantisation_reports_error_beside_every_saving(mini):
    for r in mini["quantisation"]:
        for key in ("bytes", "max_error_m", "error_over_span", "span_m",
                    "vs_24bit", "degenerate", "meaningful"):
            assert key in r, f"quantisation row without '{key}' invites a false win"


def test_tile_sweep_covers_the_requested_sizes(mini):
    assert [r["tile"] for r in mini["tile_sweep"]] == [64, 128]
    for r in mini["tile_sweep"]:
        assert r["n_tiles"] > 0 and r["bytes"] > 0 and r["seconds"] > 0


def test_obj_size_tracks_triangle_count(mini):
    rows = sorted(mini["obj_sweep"], key=lambda r: r["stride"])
    assert rows[0]["triangles"] > rows[-1]["triangles"]
    assert rows[0]["bytes"] > rows[-1]["bytes"]


def test_encode_bench_reports_a_rate_for_every_op(mini):
    assert len(mini["encode"]) >= 5
    for r in mini["encode"]:
        assert r["seconds"] > 0 and r["mpix_per_s"] > 0


# ---------------------------------------------------------------- the viewer
def test_viewer_block_is_measured_or_says_why(mini):
    """Never a silent zero: either node ran, or the reason it did not is recorded."""
    v = mini["viewer"]
    assert isinstance(v, dict)
    if "skipped" in v:
        assert v["skipped"], "skipped without a reason"
    else:
        assert v.get("ops"), "viewer block with no measurements and no skip reason"
        assert v["totals_ms"]["first_paint_cpu"] > 0


def test_quantise_to_bits_zeroes_the_low_bits():
    """The property the whole sweep rests on, checked directly."""
    a = np.linspace(0.0, 1.0, 4096).reshape(64, 64)
    for bits in (16, 12, 8):
        rgb = D._quantise_to_bits(a, 0.0, 1.0, bits)
        code = (rgb[..., 0].astype(np.uint32) << 16 | rgb[..., 1].astype(np.uint32) << 8
                | rgb[..., 2].astype(np.uint32))
        assert np.all(code % (1 << (24 - bits)) == 0), f"{bits}-bit left low bits set"
        assert len(np.unique(code)) <= (1 << bits)
