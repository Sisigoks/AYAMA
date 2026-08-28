"""Phase 3/4 integration: the tileset a run produces, and the page that reads it.

Two contracts are under test.

`build_tileset` must not alter the surface: a tile decoded back must equal the
raster it came from, at every LOD, with every pixel accounted for.

`web/` must agree with `ayama/mesh/`. The manifest is the only thing joining a
Python writer to a JavaScript reader, and nothing else in the suite would notice
if the two drifted apart - the page would simply render a wrong surface without
complaining. The checks here are structural (every element the script reaches
for exists; every manifest key it reads is written); `scripts/check_app.js`
executes the page for the rest.
"""
from __future__ import annotations

import json
import os
import re

import numpy as np
import pytest

from ayama.core.types import SceneMeta
from ayama.dsm.cog import write_cog, write_rgb
from ayama.mesh.build import build_tileset, derive_notes, load_run
from ayama.mesh.encode import decode_linear, decode_terrain_rgb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB = os.path.join(ROOT, "web")
REAL_RUN = os.path.join(ROOT, "results", "seed7", "run")


# --------------------------------------------------------------------- fixture
@pytest.fixture(scope="module")
def fake_run(tmp_path_factory):
    """A small run directory with the artifacts `ayama run` writes."""
    d = tmp_path_factory.mktemp("run")
    h = w = 96
    rr, cc = np.mgrid[0:h, 0:w]
    dtm = 400.0 + 0.05 * rr + 0.02 * cc
    ndsm = np.zeros((h, w), np.float32)
    ndsm[20:40, 20:40] = 12.0                       # a building
    dsm = (dtm + ndsm).astype(np.float32)
    sem = np.zeros((h, w), np.float32)
    sem[20:40, 20:40] = 2                           # BUILDING
    sigma = np.full((h, w), 3.0, np.float32)
    err = (dsm - dtm - ndsm).astype(np.float32)

    meta = SceneMeta(crs="EPSG:32644", transform=(0.5, 0, 500000.0, 0, -0.5, 2000000.0),
                     gsd_m=0.5, sun_azimuth_deg=138.4, sun_elevation_deg=61.2)
    write_cog(str(d / "dsm.tif"), dsm, meta, description="DSM (m)")
    write_cog(str(d / "ndsm.tif"), ndsm, meta)
    write_cog(str(d / "sigma.tif"), sigma, meta)
    write_cog(str(d / "error.tif"), err, meta)
    write_cog(str(d / "sem.tif"), sem, meta, dtype="uint8", nodata=255)
    write_rgb(str(d / "texture.jpg"), np.full((h, w, 3), 128, np.uint8))
    with open(d / "provenance.json", "w", encoding="utf-8") as fh:
        json.dump({"backbone": "synthetic", "segmentation": "heuristic",
                   "dem": "simulated copernicus", "tier": "A"}, fh)
    return str(d), {"dsm": dsm, "ndsm": ndsm, "sigma": sigma, "sem": sem}


@pytest.fixture(scope="module")
def built(fake_run, tmp_path_factory):
    run_dir, arrays = fake_run
    out = str(tmp_path_factory.mktemp("tiles"))
    # `lods` is forced: a 96 px fixture is below the size at which the builder
    # would derive more than one level, and the LOD contract still needs testing.
    manifest = build_tileset(run_dir, out, tile=32, pad=1, obj_stride=2, lods=3)
    return manifest, out, arrays


# ------------------------------------------------------------------- manifest
def test_manifest_is_strict_json(built):
    """A bare NaN parses in Python and makes JSON.parse throw, blanking the page."""
    _, out, _ = built
    raw = open(os.path.join(out, "tileset.json"), encoding="utf-8").read()
    assert "NaN" not in raw and "Infinity" not in raw
    json.loads(raw)


def test_manifest_carries_the_grid_and_georeference(built):
    m, _, _ = built
    assert m["ayama_tileset_version"] >= 1
    assert m["grid"] == {"width": 96, "height": 96, "gsd_m": 0.5, "tile": 32, "pad": 1,
                         "extent_m": [48.0, 48.0], "quantise_bits": 24}
    assert m["crs"] == "EPSG:32644"
    assert len(m["transform"]) == 6


def test_every_declared_tile_file_exists(built):
    m, out, _ = built
    n = 0
    for lod in m["lods"]:
        for t in lod["tiles"]:
            for rel in t["layers"].values():
                assert os.path.exists(os.path.join(out, rel)), rel
                n += 1
    assert n > 0


def test_tiles_cover_every_pixel_at_every_lod(built):
    m, _, _ = built
    for lod in m["lods"]:
        seen = np.zeros((lod["height"], lod["width"]), np.int32)
        for t in lod["tiles"]:
            seen[t["y0"]:t["y0"] + t["height"], t["x0"]:t["x0"] + t["width"]] += 1
        assert seen.min() == 1 and seen.max() == 1


def test_lod_count_is_derived_from_the_raster_size():
    from ayama.mesh.build import _lod_count

    assert _lod_count((96, 96), 32) == 1            # already near the floor
    assert _lod_count((1024, 1024), 512) == 4       # 1024 / 512 / 256 / 128
    assert _lod_count((4096, 4096), 512) == 6       # capped, not unbounded


def test_lods_halve_resolution_and_double_gsd(built):
    m, _, _ = built
    assert len(m["lods"]) == 3
    for i, lod in enumerate(m["lods"]):
        assert lod["stride"] == 2 ** i
        assert lod["gsd_m"] == pytest.approx(0.5 * 2 ** i)
        assert lod["width"] == 96 // (2 ** i)


def test_layer_ranges_are_shared_across_lods(built):
    """Per-LOD ranges would make one elevation decode differently at each zoom."""
    m, _, _ = built
    for key in ("ndsm", "sigma", "error"):
        assert m["layers"][key]["vmin"] is not None
        assert m["layers"][key]["vmax"] > m["layers"][key]["vmin"]
        assert m["layers"][key]["encoding"] == "linear"
    assert m["layers"]["dsm"]["encoding"] == "terrain-rgb"
    assert m["layers"]["dsm"]["step_m"] == 0.1


# ------------------------------------------------------- the surface is intact
def _tile_pixels(out, rel):
    from PIL import Image

    return np.asarray(Image.open(os.path.join(out, rel)).convert("RGB"), np.uint8)


def test_decoded_dsm_tiles_equal_the_source_raster(built):
    m, out, arrays = built
    lod0 = m["lods"][0]
    for t in lod0["tiles"]:
        got = decode_terrain_rgb(_tile_pixels(out, t["layers"]["dsm"]))
        want = arrays["dsm"][t["y0"]:t["y0"] + t["height"], t["x0"]:t["x0"] + t["width"]]
        assert got.shape == want.shape
        assert np.abs(got - want).max() <= 0.05 + 1e-3


def test_decoded_ndsm_keeps_the_building(built):
    m, out, arrays = built
    spec = m["layers"]["ndsm"]
    lod0 = m["lods"][0]
    rebuilt = np.zeros((lod0["height"], lod0["width"]), np.float32)
    for t in lod0["tiles"]:
        v = decode_linear(_tile_pixels(out, t["layers"]["ndsm"]), spec["vmin"], spec["vmax"])
        rebuilt[t["y0"]:t["y0"] + t["height"], t["x0"]:t["x0"] + t["width"]] = v
    assert np.abs(rebuilt - arrays["ndsm"]).max() < 1e-3
    assert rebuilt[30, 30] == pytest.approx(12.0, abs=1e-3)


def test_coarser_lods_are_decimations_of_the_source(built):
    m, out, arrays = built
    lod1 = m["lods"][1]
    rebuilt = np.zeros((lod1["height"], lod1["width"]), np.float32)
    for t in lod1["tiles"]:
        rebuilt[t["y0"]:t["y0"] + t["height"], t["x0"]:t["x0"] + t["width"]] = \
            decode_terrain_rgb(_tile_pixels(out, t["layers"]["dsm"]))
    assert np.abs(rebuilt - arrays["dsm"][::2, ::2]).max() <= 0.05 + 1e-3


def test_obj_is_written_and_referenced_relatively(built):
    m, out, _ = built
    assert m["mesh"]["triangles"] > 0
    assert not os.path.isabs(m["mesh"]["obj"])
    assert os.path.exists(os.path.join(out, m["mesh"]["obj"]))
    assert os.path.exists(os.path.join(out, m["mesh"]["mtl"]))


# ------------------------------------------------------------------ the notes
def test_a_flat_surface_raises_a_critical_note(fake_run):
    """The check that stops a flattened city being shipped as a 3D deliverable."""
    run_dir, _ = fake_run
    run = load_run(run_dir)
    run["ndsm"] = np.zeros_like(run["ndsm"])          # what Phase 2 currently emits
    notes = derive_notes(run, {})
    flat = [n for n in notes if n["id"] == "flat_surface"]
    assert len(flat) == 1
    assert flat[0]["level"] == "critical"


def test_a_real_surface_raises_no_critical_note(fake_run):
    run_dir, _ = fake_run
    run = load_run(run_dir)
    notes = derive_notes(run, {})
    assert not [n for n in notes if n["level"] == "critical"]


def test_simulated_inputs_are_declared(built):
    m, _, _ = built
    ids = {n["id"] for n in m["notes"]}
    assert "simulated_dem" in ids
    assert "heuristic_segmentation" in ids


def test_load_run_rejects_a_directory_without_a_dsm(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_run(str(tmp_path))


# ----------------------------------------------- the page agrees with the data
def _read(name):
    return open(os.path.join(WEB, name), encoding="utf-8").read()


def test_web_assets_exist():
    for f in ("index.html", "app.js", "style.css"):
        assert os.path.exists(os.path.join(WEB, f)), f


def test_every_element_the_script_reaches_for_exists_in_the_html():
    """Both scripts: app.js draws the tileset, upload.js drives the job."""
    app, html = _read("app.js"), _read("index.html")
    app += _read("upload.js")
    ids = set(re.findall(r"getElementById\('([\w-]+)'\)", app))
    ids |= set(re.findall(r"\$\('([\w-]+)'\)", _read("upload.js")))
    ids |= set(re.findall(r"\$\('#([\w-]+)'\)", app))
    ids |= set(re.findall(r"querySelector\('#([\w-]+)'\)", app))
    assert ids, "found no element lookups - the extraction regex is stale"
    present = set(re.findall(r'id="([\w-]+)"', html))
    assert ids <= present, f"app.js reaches for missing ids: {sorted(ids - present)}"


def test_the_page_reads_only_manifest_keys_the_builder_writes(built):
    """Both halves of the contract, checked against a manifest that really exists."""
    m, _, _ = built
    app = _read("app.js")
    for key in ("lods", "layers", "notes", "grid", "default_layer", "provenance",
                "metrics", "mesh", "transform", "source_run", "generated_utc"):
        assert key in m, f"builder stopped writing {key}"
        assert key in app, f"app.js stopped reading {key}"
    lod = m["lods"][0]
    for key in ("tiles", "width", "height", "gsd_m", "lod"):
        assert key in lod
    for key in ("x0", "y0", "layers"):
        assert key in lod["tiles"][0]


def test_the_page_and_the_encoder_share_the_terrain_rgb_constants():
    """Two implementations of one packing. Drift here is silent and total."""
    from ayama.mesh.encode import MAX_CODE, TERRAIN_BASE_M, TERRAIN_STEP_M

    app = _read("app.js")
    assert f"TERRAIN_BASE = {TERRAIN_BASE_M}" in app
    assert f"TERRAIN_STEP = {TERRAIN_STEP_M}" in app
    assert "256 * 256 * 256 - 1" in app and MAX_CODE == 256 ** 3 - 1


def test_the_page_declares_no_external_resources():
    """No CDN, no build step: the viewer has to work with the network off.

    Only fetching attributes are inspected. The inline SVG favicon legitimately
    contains an `xmlns` of http://www.w3.org/2000/svg, which is a namespace name
    and is never dereferenced - a blanket search for "http" would flag it.
    """
    html = _read("index.html")
    external = re.findall(r'(?:src|href)\s*=\s*"(https?://[^"]*)"', html)
    assert not external, f"index.html loads from outside: {external}"
    for bad in ("cdn.", "unpkg", "jsdelivr", "googleapis"):
        assert bad not in html, f"index.html reaches outside for {bad}"
    assert '<script src="app.js">' in html
    assert '@import' not in _read("style.css")


def test_colour_ramps_match_the_png_previews():
    """The page and cog.py must not disagree about what a height looks like."""
    from ayama.dsm.cog import _fallback_lut

    app = _read("app.js")
    for name, key in (("viridis", "viridis"), ("magma", "magma"), ("terrain", "terrain")):
        first = _fallback_lut(name)[0][:3]
        assert f"[{first[0]}, {first[1]}, {first[2]}]" in app, f"{key} ramp drifted"


# --------------------------------------------------- against the real Phase 2
@pytest.mark.skipif(not os.path.isdir(REAL_RUN), reason="results/seed7/run not present")
def test_the_real_phase2_run_tiles_and_reports_its_own_defect(tmp_path):
    """End to end on the CPU study's own output, not a fixture.

    This is the run the README reports, so the tileset it produces must carry
    the same numbers - and must raise the flat-surface note, because that run
    really does have one.
    """
    m = build_tileset(REAL_RUN, str(tmp_path / "t"), tile=512, write_mesh=False)
    assert m["grid"]["width"] == 1024 and m["grid"]["gsd_m"] == pytest.approx(0.5)
    assert m["crs"] == "EPSG:32644"

    nd = m["layers"]["ndsm"]["stats"]
    assert nd["max"] < 1.0, "the run stopped being flat - update README section 5"
    assert [n for n in m["notes"] if n["id"] == "flat_surface"]

    # Metrics come from the study, unchanged.
    assert m["tier"] == "A"
    assert m["metrics"]["mae_m"] == pytest.approx(3.394, abs=0.01)
    assert m["metrics"]["edge_f1"] == pytest.approx(0.276, abs=0.01)

    # And the sun the viewer lights from is the scene's own.
    assert m["provenance"]["sun_elevation_deg"] == pytest.approx(61.2, abs=0.1)


# ------------------------------------------------ the published web tileset
def test_a_quantised_build_still_round_trips(fake_run, tmp_path):
    """12-bit is what the published demo ships, so it gets the same guarantee."""
    run_dir, arrays = fake_run
    out = str(tmp_path / "q")
    m = build_tileset(run_dir, out, tile=32, pad=1, write_mesh=False, quantise_bits=12)

    assert m["grid"]["quantise_bits"] == 12
    for key in ("ndsm", "sigma", "error"):
        assert m["layers"][key]["bits"] == 12
        assert "data_range_m" in m["layers"][key]

    spec = m["layers"]["ndsm"]
    lod0 = m["lods"][0]
    rebuilt = np.zeros((lod0["height"], lod0["width"]), np.float32)
    for t in lod0["tiles"]:
        rebuilt[t["y0"]:t["y0"] + t["height"], t["x0"]:t["x0"] + t["width"]] = \
            decode_linear(_tile_pixels(out, t["layers"]["ndsm"]),
                          spec["vmin"], spec["vmax"])
    assert np.abs(rebuilt - arrays["ndsm"]).max() <= spec["step_m"] / 2 + 1e-6


# The committed demo tileset the published site embeds. It lives inside web/
# so that web/ is self-contained: any static server rooted there serves a
# working viewer, and the Pages workflow just copies the directory.
DEMO = os.path.join(ROOT, "web", "data")


@pytest.mark.skipif(not os.path.exists(os.path.join(DEMO, "tileset.json")),
                    reason="web/data not built")
def test_the_published_demo_tileset_is_web_sized_and_intact():
    """The tileset the GitHub Pages site serves. It is committed, so it is tested.

    Two things matter for a published demo: that it is small enough to load,
    and that shrinking it did not quietly change the surface.
    """
    with open(os.path.join(DEMO, "tileset.json"), encoding="utf-8") as fh:
        m = json.load(fh)

    total = sum(os.path.getsize(os.path.join(b, f))
                for b, _, fs in os.walk(DEMO) for f in fs)
    assert total < 6e6, f"demo tileset is {total / 1e6:.1f} MB; too heavy for a page"
    assert m["mesh"] is None, "the demo must not ship the 139 MB OBJ"
    assert m["grid"]["quantise_bits"] == 12

    for lod in m["lods"]:
        for t in lod["tiles"]:
            for rel in t["layers"].values():
                assert os.path.exists(os.path.join(DEMO, rel)), rel

    # The flat-surface note is the whole reason the demo is honest.
    assert [n for n in m["notes"] if n["id"] == "flat_surface"]
    assert m["metrics"]["mae_m"] == pytest.approx(3.394, abs=0.01)
