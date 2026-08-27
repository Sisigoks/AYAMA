"""Phase 3 contract tests: encodings, tiling, OBJ export.

The theme is that Phase 3 must not change the surface. Every test here is some
form of "what came out equals what went in", because a delivery layer that
quietly alters elevation is worse than one that fails - the failure is visible
and the alteration is not.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from ayama.mesh.encode import (MAX_CODE, TERRAIN_BASE_M, TERRAIN_STEP_M,
                               decode_linear, decode_normal_map,
                               decode_terrain_rgb, encode_linear,
                               encode_linear_bits, encode_terrain_rgb,
                               linear_range_for_bits, linear_step,
                               normal_map, quantisation_step)
from ayama.mesh.obj import write_obj
from ayama.mesh.tiles import (cut, grid_size, interior, pyramid, reassemble,
                              tile_specs)


# --------------------------------------------------------------------- encode
def test_terrain_rgb_roundtrips_within_half_a_step():
    rng = np.random.default_rng(3)
    dsm = rng.uniform(-400.0, 8800.0, (64, 64)).astype(np.float32)
    back = decode_terrain_rgb(encode_terrain_rgb(dsm))
    # Half a step from the quantiser, plus float32's own spacing at 8800 m
    # (~1e-3), because the decoder returns float32 to match the raster dtype.
    assert np.abs(back - dsm).max() <= TERRAIN_STEP_M / 2 + 2e-3


def test_terrain_rgb_matches_the_mapbox_constants():
    # Other tools decode with exactly this expression; a changed base or step
    # would still round-trip through our own decoder and be wrong everywhere else.
    rgb = np.array([[[1, 2, 3]]], np.uint8)
    code = (1 << 16) | (2 << 8) | 3
    assert decode_terrain_rgb(rgb)[0, 0] == pytest.approx(
        TERRAIN_BASE_M + code * TERRAIN_STEP_M)


def test_terrain_rgb_saturates_instead_of_wrapping():
    """An out-of-range height must clamp, not wrap around to a plausible one.

    Wrapping is the dangerous failure: 40000 m would come back as a small
    positive elevation and render as ordinary terrain.
    """
    rgb = encode_terrain_rgb(np.array([[1e9]], np.float32))
    assert _code(rgb)[0, 0] == MAX_CODE
    rgb_low = encode_terrain_rgb(np.array([[-1e9]], np.float32))
    assert _code(rgb_low)[0, 0] == 0


def _code(rgb):
    a = rgb.astype(np.uint32)
    return (a[..., 0] << 16) | (a[..., 1] << 8) | a[..., 2]


def test_non_finite_encodes_to_the_nodata_floor():
    rgb = encode_terrain_rgb(np.array([[np.nan, np.inf, 100.0]], np.float32))
    assert decode_terrain_rgb(rgb)[0, 0] == pytest.approx(TERRAIN_BASE_M)
    assert decode_terrain_rgb(rgb)[0, 2] == pytest.approx(100.0, abs=0.05)


def test_linear_encoding_beats_terrain_rgb_on_a_tiny_range():
    """The reason two encodings exist.

    Phase 2 currently emits an nDSM spanning well under a metre. Terrain-RGB's
    fixed 0.1 m step turns that into a handful of terraces; the linear encoding
    spends all 24 bits on the range that is actually there.
    """
    ndsm = np.linspace(0.0, 0.28, 4096).reshape(64, 64).astype(np.float32)

    rgb, vmin, vmax = encode_linear(ndsm)
    linear_err = np.abs(decode_linear(rgb, vmin, vmax) - ndsm).max()
    terrain_err = np.abs(decode_terrain_rgb(encode_terrain_rgb(ndsm)) - ndsm).max()

    assert linear_err < 1e-6
    assert terrain_err > 1000 * linear_err
    assert quantisation_step(vmin, vmax) < 1e-7


def test_linear_encoding_handles_a_constant_layer():
    rgb, vmin, vmax = encode_linear(np.full((8, 8), 3.0, np.float32))
    assert vmax > vmin                       # widened, not divided by zero
    assert np.all(np.isfinite(decode_linear(rgb, vmin, vmax)))


def test_linear_range_is_honoured_when_given():
    a = np.linspace(0, 10, 64).reshape(8, 8).astype(np.float32)
    rgb, vmin, vmax = encode_linear(a, -100.0, 100.0)
    assert (vmin, vmax) == (-100.0, 100.0)
    assert np.abs(decode_linear(rgb, vmin, vmax) - a).max() < 1e-4


# -------------------------------------------------------------------- normals
def test_flat_ground_points_straight_up():
    n = decode_normal_map(normal_map(np.full((16, 16), 42.0, np.float32), 0.5))
    assert n[..., 2].min() > 0.99
    assert np.abs(n[..., 0]).max() < 0.02
    assert np.abs(n[..., 1]).max() < 0.02


def test_normals_are_unit_length():
    rng = np.random.default_rng(11)
    dsm = rng.normal(0, 3, (32, 32)).astype(np.float32)
    n = decode_normal_map(normal_map(dsm, 0.5))
    assert np.abs(np.linalg.norm(n, axis=2) - 1.0).max() < 0.02


def test_a_slope_rising_east_tilts_its_normal_west():
    # Elevation increasing with +col (east) must give a normal leaning -X.
    dsm = np.tile(np.arange(32, dtype=np.float32) * 1.0, (32, 1))
    n = decode_normal_map(normal_map(dsm, 1.0))
    assert n[16, 16, 0] < -0.3


def test_normals_respond_to_gsd():
    """The same height step is a gentler slope when pixels are further apart."""
    dsm = np.tile(np.arange(16, dtype=np.float32), (16, 1))
    steep = decode_normal_map(normal_map(dsm, 0.5))[8, 8, 2]
    gentle = decode_normal_map(normal_map(dsm, 5.0))[8, 8, 2]
    assert gentle > steep


# ---------------------------------------------------------------------- tiles
def test_tiles_cover_the_raster_exactly_once():
    specs = tile_specs((1000, 700), tile=256, pad=1)
    seen = np.zeros((1000, 700), np.int32)
    for s in specs:
        seen[s.y0:s.y1, s.x0:s.x1] += 1
    assert seen.min() == 1 and seen.max() == 1
    assert grid_size((1000, 700), 256) == (4, 3)


def test_interiors_reassemble_into_the_original():
    rng = np.random.default_rng(5)
    a = rng.normal(400, 5, (300, 420)).astype(np.float32)
    specs = tile_specs(a.shape, tile=128, pad=2)
    parts = {s.key: interior(cut(a, s), s) for s in specs}
    assert np.array_equal(reassemble(parts, a.shape), a)


def test_padding_reads_the_neighbours_pixels():
    """The point of the pad: a tile's halo is its neighbour's data, not a copy."""
    a = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
    specs = tile_specs(a.shape, tile=32, pad=2)
    right = [s for s in specs if (s.row, s.col) == (0, 1)][0]
    padded = cut(a, right)
    top, left = right.inset                      # always (pad, pad) after cut
    assert (top, left) == (2, 2)
    # The two columns left of this tile's interior are the last two of tile 0_0.
    assert np.array_equal(padded[top:top + 32, left - 2:left], a[0:32, 30:32])
    assert np.array_equal(interior(padded, right), a[0:32, 32:64])


def test_border_tiles_replicate_instead_of_inventing():
    a = np.arange(32 * 32, dtype=np.float32).reshape(32, 32)
    spec = tile_specs(a.shape, tile=32, pad=3)[0]
    padded = cut(a, spec)
    assert padded.shape == (38, 38)
    # Replicated edge, so the gradient across the border is zero, not wrong.
    assert np.array_equal(padded[0], padded[3])
    assert np.array_equal(padded[:, 0], padded[:, 3])


def test_seam_free_normals_are_why_padding_exists():
    """Normals from padded tiles must equal normals of the whole raster."""
    rng = np.random.default_rng(9)
    dsm = rng.normal(0, 4, (128, 128)).astype(np.float32)
    whole = normal_map(dsm, 0.5)

    specs = tile_specs(dsm.shape, tile=64, pad=1)
    parts = {s.key: interior(normal_map(cut(dsm, s), 0.5), s) for s in specs}
    stitched = reassemble(parts, dsm.shape, dtype=np.uint8)

    # Byte-exact across every internal seam. The raster's own outer ring is
    # excluded because the two paths use different (both defensible) edge
    # conventions there: np.gradient takes a one-sided difference on the whole
    # raster, while a border tile sees a replicated row and reads a zero
    # gradient. That disagreement is one pixel wide at the scene edge and
    # cannot produce a seam, which is what this test is about.
    assert np.array_equal(stitched[1:-1, 1:-1], whole[1:-1, 1:-1])


def test_unpadded_tiling_leaves_visible_seams():
    """The control for the test above: without a pad, seams really do appear."""
    rng = np.random.default_rng(9)
    dsm = rng.normal(0, 4, (128, 128)).astype(np.float32)
    whole = normal_map(dsm, 0.5)
    specs = tile_specs(dsm.shape, tile=64, pad=0)
    parts = {s.key: normal_map(cut(dsm, s), 0.5) for s in specs}
    stitched = reassemble(parts, dsm.shape, dtype=np.uint8)
    assert not np.array_equal(stitched[1:-1, 1:-1], whole[1:-1, 1:-1])

    # And the damage is exactly where a seam would show: the tile boundaries.
    differs = (stitched != whole).any(axis=2)
    assert differs[63, 5] and differs[64, 5]        # the horizontal seam
    assert not differs[30, 5]                       # tile interiors are clean


def test_pyramid_decimates_rather_than_averages():
    a = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
    levels = list(pyramid(a, 3))
    assert [l[0] for l in levels] == [0, 1, 2]
    assert levels[1][1].shape == (32, 32)
    # Every value at every level is a value that exists in the source surface.
    assert np.isin(levels[2][1], a).all()


# ------------------------------------------------------------------------ obj
def test_obj_geometry_is_well_formed(tmp_path):
    dsm = np.fromfunction(lambda r, c: 100 + 0.1 * r + 0.05 * c, (16, 20), dtype=np.float32)
    info = write_obj(str(tmp_path / "s.obj"), dsm, gsd_m=0.5, texture_name="t.jpg")

    assert info["vertices"] == 16 * 20
    assert info["triangles"] == 2 * 15 * 19
    assert os.path.exists(info["mtl"])

    verts, uvs, faces = _parse_obj(info["obj"])
    assert len(verts) == info["vertices"]
    assert len(uvs) == info["vertices"]
    assert len(faces) == info["triangles"]
    flat = [i for f in faces for i in f]
    assert min(flat) >= 1 and max(flat) <= len(verts)      # 1-based, in range
    assert all(0.0 <= u <= 1.0 and 0.0 <= v <= 1.0 for u, v in uvs)


def test_obj_axes_put_north_up_and_elevation_in_z(tmp_path):
    dsm = np.zeros((4, 4), np.float32)
    dsm[0, 0] = 50.0                        # north-west corner
    info = write_obj(str(tmp_path / "s.obj"), dsm, gsd_m=2.0, texture_name=None)
    verts, _, _ = _parse_obj(info["obj"])

    # Row 0 is northernmost, so it must carry the largest Y.
    assert verts[0][1] == pytest.approx(3 * 2.0)
    assert verts[0][0] == pytest.approx(0.0)
    assert verts[0][2] == pytest.approx(50.0)
    assert verts[-1][1] == pytest.approx(0.0)


def test_obj_skips_faces_over_nodata(tmp_path):
    dsm = np.full((8, 8), 10.0, np.float32)
    dsm[4, 4] = np.nan
    info = write_obj(str(tmp_path / "s.obj"), dsm, gsd_m=1.0)
    # The NaN corner belongs to four quads, all of which must be dropped.
    assert info["dropped_quads"] == 4
    assert info["triangles"] == 2 * (7 * 7 - 4)


def test_obj_stride_decimates(tmp_path):
    dsm = np.zeros((32, 32), np.float32)
    full = write_obj(str(tmp_path / "a.obj"), dsm, 0.5, stride=1)
    half = write_obj(str(tmp_path / "b.obj"), dsm, 0.5, stride=2)
    assert half["vertices"] == 16 * 16
    assert half["step_m"] == pytest.approx(1.0)
    assert half["vertices"] * 4 == full["vertices"]


def test_obj_refuses_a_grid_too_small_to_mesh(tmp_path):
    with pytest.raises(ValueError):
        write_obj(str(tmp_path / "s.obj"), np.zeros((4, 4), np.float32), 1.0, stride=8)


def _parse_obj(path: str):
    verts, uvs, faces = [], [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "v":
                verts.append(tuple(float(x) for x in parts[1:4]))
            elif parts[0] == "vt":
                uvs.append(tuple(float(x) for x in parts[1:3]))
            elif parts[0] == "f":
                faces.append(tuple(int(p.split("/")[0]) for p in parts[1:4]))
    return verts, uvs, faces


# ------------------------------------------------------- quantised encoding
def _codes(rgb):
    a = rgb.astype(np.uint32)
    return (a[..., 0] << 16) | (a[..., 1] << 8) | a[..., 2]


@pytest.mark.parametrize("bits", [16, 12, 8])
def test_quantised_encoding_zeroes_the_low_bits(bits):
    """The property the payload saving rests on: a narrower field, really.

    Rounding in value space and re-encoding leaves the low byte noisy through
    floating-point jitter, and PNG cannot collapse it - which once produced a
    larger file at 12 bits than at 16.
    """
    a = np.linspace(0.0, 5.0, 4096).reshape(64, 64)
    code = _codes(encode_linear_bits(a, 0.0, 5.0, bits))
    assert np.all(code % (1 << (24 - bits)) == 0)
    assert len(np.unique(code)) <= (1 << bits)


@pytest.mark.parametrize("bits", [24, 16, 12, 8])
def test_quantised_encoding_round_trips_within_half_a_step(bits):  # noqa: D401
    """With the manifest range the encoder chooses, the decode stays exact.

    The bit shift leaves the largest emittable code short of 2^24 - 1, so a
    plain decode would read the top of the range systematically low - 0.024% at
    12 bits, a compression toward vmin rather than a rounding error.
    `linear_range_for_bits` widens the recorded vmax by exactly that ratio.
    """
    rng = np.random.default_rng(4)
    a = rng.uniform(-40.0, 15.0, (64, 64))
    lo, hi = float(a.min()), float(a.max())
    rgb = encode_linear_bits(a, lo, hi, bits)
    enc_min, enc_max = linear_range_for_bits(lo, hi, bits)
    back = decode_linear(rgb, enc_min, enc_max)
    step = linear_step(lo, hi, bits)
    # decode_linear returns float32 to match the raster dtype, so at 24 bits the
    # storage type is the floor rather than the quantiser: float32 spacing at 40
    # is about 4e-6, which is larger than a 24-bit step over this range.
    float32_floor = abs(hi) * 1e-6 + 1e-6
    assert np.abs(back - a).max() <= step / 2 + float32_floor


def test_ignoring_the_encoder_range_biases_the_result_low():
    """Why the range is in the manifest: decoding with the raw range is wrong."""
    a = np.linspace(-40.0, 15.0, 4096).reshape(64, 64)
    rgb = encode_linear_bits(a, -40.0, 15.0, 12)
    naive = decode_linear(rgb, -40.0, 15.0)          # the raw data range
    enc_min, enc_max = linear_range_for_bits(-40.0, 15.0, 12)
    correct = decode_linear(rgb, enc_min, enc_max)
    assert np.abs(correct - a).max() < np.abs(naive - a).max()
    assert (naive - a).mean() < 0                    # biased toward vmin


def test_fewer_bits_never_costs_less_error():
    """24 bits is excluded: there the float32 return type sets the error, not
    the quantiser, so the ordering says nothing about the encoding."""
    # Deliberately irregular. linspace(0, 9, 4096) lands exactly on the 12-bit
    # level grid, so 12 bits would encode it losslessly and beat 16 - which says
    # something about that array, not about the encoding.
    a = np.random.default_rng(23).uniform(0.0, 9.0, (64, 64))
    worst = []
    for bits in (16, 12, 8):
        rgb = encode_linear_bits(a, 0.0, 9.0, bits)
        lo, hi = linear_range_for_bits(0.0, 9.0, bits)
        worst.append(float(np.abs(decode_linear(rgb, lo, hi) - a).max()))
    assert worst == sorted(worst)


def test_quantising_shrinks_the_encoded_png():
    """The payload claim, measured where the byte count means something.

    Not on a small tile: below a few tens of kB a PNG is mostly header and
    filter choice, and 12 bits can come out larger than 24 by pure noise.
    """
    from ayama.dsm.cog import _apply_cmap  # noqa: F401  (PIL is a core dep)

    import io as _io

    from PIL import Image

    rng = np.random.default_rng(17)
    a = np.cumsum(rng.normal(0, 0.05, (512, 512)), axis=0)   # smooth, like a surface

    def png_bytes(bits):
        rgb = encode_linear_bits(a, a.min(), a.max(), bits)
        buf = _io.BytesIO()
        Image.fromarray(np.ascontiguousarray(rgb)).save(buf, format="PNG", optimize=True)
        return buf.tell()

    full, quantised = png_bytes(24), png_bytes(12)
    assert quantised < full * 0.75, f"12-bit saved only {100 * (1 - quantised / full):.0f}%"


def test_full_bits_matches_the_unquantised_encoder():
    a = np.linspace(0.0, 3.0, 1024).reshape(32, 32)
    plain, lo, hi = encode_linear(a)
    assert np.array_equal(encode_linear_bits(a, lo, hi, 24), plain)
