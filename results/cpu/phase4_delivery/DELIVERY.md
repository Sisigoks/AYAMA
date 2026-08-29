# AYAMA delivery benchmark

Phase 3 and Phase 4 on CPU. Generated 2026-08-29T06:50:23Z by
`python -m ayama.cli delivery`.

Every number is measured on the machine below, against a real Phase 2 run.
**GPU rasterisation is not measured** and is not claimed: everything in the
viewer section is CPU work the browser does before a triangle is drawn.

## Environment

- Windows-11-10.0.26200-SP0, python 3.13.5, 8 cpus
- CUDA available: False  ·  rasterio 1.5.1
- node v20.19.4
- source run `D:\UNNAT\results\cpu\real_vitl_h1\zurich`
- scene 1024 x 1024 px at 0.5 m (1.05 Mpix, 512 x 512 m)

## Headline

| | |
|---|---|
| tileset build, tiles only | **7.35 s** (0.14 Mpix/s) |
| tileset build, with the OBJ | **13.92 s** (OBJ alone 6.56 s) |
| output | 4 LODs, 7 tiles |
| payload | **44.9 MB** total, 10.8 MB tiles + 34.0 MB mesh |
| first paint, bytes | 5.18 MB |
| first paint, viewer CPU | **66 ms** |
| whole benchmark | 175 s |

> Build timings are disk-bound and vary with where they are written; an on-access virus scanner can change them several-fold. Every timed build here used one scratch directory, reported above.

## Encoding throughput

The packing arithmetic in isolation, no disk. This is what says whether the
encoding could ever be the bottleneck.

| operation | Mpix/s | seconds |
|---|---|---|
| encode terrain-rgb | 15 | 0.0679 |
| decode terrain-rgb | 18 | 0.0588 |
| encode linear | 11 | 0.0935 |
| decode linear | 17 | 0.0626 |
| normal map | 4 | 0.2557 |
| png encode, terrain-rgb | 0 | 2.2474 |
| png encode, linear | 1 | 0.7362 |

**PNG compression dominates.** Packing pixels runs at 15 Mpix/s;
compressing them runs at 1 Mpix/s - a factor of 11.
Nothing in the encoder is worth optimising until the compressor is.

## Where one build's time goes

LOD 0 only, 4 tiles of 512 px (+1 px pad), 8.42 s accounted for.

| stage | seconds | share |
|---|---|---|
| png write normals | 2.790 | 33.1% |
| png write dsm | 2.648 | 31.5% |
| png write linear layers | 2.020 | 24.0% |
| encode linear layers | 0.487 | 5.8% |
| normals | 0.357 | 4.2% |
| encode dsm | 0.109 | 1.3% |
| cut | 0.006 | 0.1% |
| crop | 0.000 | 0.0% |

## Tile size

| tile px | LODs | tiles | files | seconds | payload |
|---|---|---|---|---|---|
| 128 | 4 | 85 | 510 | 9.58 | 10.94 MB |
| 256 | 4 | 22 | 132 | 9.30 | 10.84 MB |
| 512 | 4 | 7 | 42 | 9.90 | 10.83 MB |
| 1024 | 4 | 4 | 24 | 9.39 | 10.85 MB |

**Tile size barely moves the payload.** Across a 21x range in file count (24 to 510 files), the total varies by only 1.0% (10.83 to 10.94 MB) and build time by 0.60 s.

PNG headers and per-file compression contexts were expected to punish small
tiles; at these sizes they do not, because the pixel data dominates either
way. So tile size is free to be chosen for what it actually affects - how
much a viewer can cull, and how many requests it makes - rather than for
bytes. The default of 512 keeps the file count in double digits.

## Mesh decimation

| stride | vertices | triangles | seconds | size | bytes/triangle |
|---|---|---|---|---|---|
| 2 | 262,144 | 522,242 | 5.58 | 33.6 MB | 64 |
| 4 | 65,536 | 130,050 | 1.30 | 7.8 MB | 60 |

OBJ is a text format, so size tracks triangle count almost exactly. This is
the argument for glTF, and the reason `--obj-stride` and `--no-mesh` exist.

## What full precision costs

The linear encoding spends all 24 bits, so its low byte is incompressible
noise. Each row below keeps only the top N bits of the code and zeroes the
rest, which is what a narrower field would really store - and what lets PNG
collapse them.

| layer | precision | step | size | vs 24-bit | worst error |
|---|---|---|---|---|---|
| ndsm | 24-bit (shipped) | 2.862e-07 m | 1274 kB | 1.00x | 2.38e-07 m |
| ndsm | 16-bit | 7.328e-05 m | 852 kB | 0.67x | 0.000107 m |
| ndsm | 12-bit | 0.001173 m | 576 kB | 0.45x | 0.00175 m |
| ndsm | 8-bit | 0.01883 m | 230 kB | 0.18x | 0.0274 m |
| sigma | 24-bit (shipped) | 3.568e-09 m | 2364 kB | 1.00x | 0 m |
| sigma | 16-bit | 9.135e-07 m | 1168 kB | 0.49x | 1.43e-06 m |
| sigma | 12-bit | 1.462e-05 m | 538 kB | 0.23x | 2.15e-05 m |
| sigma | 8-bit | 0.0002348 m | 137 kB | 0.06x | 0.000342 m |
| error | 24-bit (shipped) | 2.661e-06 m | 2745 kB | 1.00x | 1.91e-06 m |
| error | 16-bit | 0.0006813 m | 1780 kB | 0.65x | 0.000999 m |
| error | 12-bit | 0.0109 m | 1196 kB | 0.44x | 0.0161 m |
| error | 8-bit | 0.1751 m | 525 kB | 0.19x | 0.259 m |

**ndsm at 12 bits, sigma at 12 bits, error at 12 bits** resolves every layer to better than 0.1% of its
own range, and takes the linear layers from 6.38 MB to
2.31 MB - a saving of **4.07 MB**, 64% of their bytes.

The precision column is not decoration. An earlier version of this sweep
stepped by fractions of the mean sigma instead of by the layer's own range,
and reported a 99.8% saving on nDSM - by rounding a layer that spans 0.28 m
with a 0.75 m step, which flattens it to a constant. A byte count alone
cannot tell a compression win from a deleted measurement.

## The surface survives the trip

Every tile decoded back and compared against the raster it came from.

| LOD | layer | encoding | step | worst error | within half a step |
|---|---|---|---|---|---|
| 0 | dsm | terrain-rgb | 0.1 | 0.05 m | yes |
| 0 | ndsm | linear | 2.86e-07 | 2.38e-07 m | yes |
| 0 | sigma | linear | 3.57e-09 | 0 m | yes |
| 0 | error | linear | 2.66e-06 | 1.91e-06 m | yes |
| 1 | dsm | terrain-rgb | 0.1 | 0.05 m | yes |
| 1 | ndsm | linear | 2.86e-07 | 2.38e-07 m | yes |
| 1 | sigma | linear | 3.57e-09 | 0 m | yes |
| 1 | error | linear | 2.66e-06 | 1.91e-06 m | yes |
| 2 | dsm | terrain-rgb | 0.1 | 0.05 m | yes |
| 2 | ndsm | linear | 2.86e-07 | 2.38e-07 m | yes |
| 2 | sigma | linear | 3.57e-09 | 0 m | yes |
| 2 | error | linear | 2.66e-06 | 1.91e-06 m | yes |
| 3 | dsm | terrain-rgb | 0.1 | 0.05 m | yes |
| 3 | ndsm | linear | 2.86e-07 | 2.38e-07 m | yes |
| 3 | sigma | linear | 3.57e-09 | 0 m | yes |
| 3 | error | linear | 2.66e-06 | 1.91e-06 m | yes |

## Payload

| layer | bytes | share of tiles |
|---|---|---|
| error | 3.68 MB | 34.0% |
| sigma | 3.19 MB | 29.5% |
| ndsm | 1.74 MB | 16.1% |
| normal | 1.34 MB | 12.3% |
| texture | 0.54 MB | 4.9% |
| dsm | 0.34 MB | 3.1% |
| **mesh** | 34.04 MB | - |

| LOD | bytes |
|---|---|
| lod0 | 7.92 MB |
| lod1 | 2.15 MB |
| lod2 | 0.59 MB |
| lod3 | 0.16 MB |

**First paint** fetches 5.18 MB: geometry,
normals, the default drape and the two layers the cursor readout needs.

## Viewer CPU

Measured against the real `web/app.js` under node, best of five with a
warm-up. This is the work the browser does before the GPU is involved.

| operation | ms |
|---|---|
| parse tileset.json | 0.06 |
| decode terrain-rgb, whole scene | 8.43 |
| decode linear, whole scene | 8.03 |
| decode terrain-rgb, reusing the buffer | 7.06 |
| tileGeometry, one tile | 4.85 |
| gridIndices, one tile | 3.67 |
| colourize, one tile | 3.94 |
| build a 256-entry LUT | 0.14 |
| renderPanels (jsdom) (jsdom not installed) | - |

| for the whole scene | ms |
|---|---|
| decode every data layer | 32.1 |
| build geometry for every tile | 34.1 |
| **CPU before first paint** | **66.2** |
| re-colour on a layer switch | 15.8 |

Decode throughput: **124 Mpix/s** terrain-rgb, 131 Mpix/s linear.

**Reusing the output buffer is 16% faster** (7.1 ms against 8.4 ms). The viewer currently
allocates a fresh `Float32Array` per tile per layer; it does not have to.

## What the tileset says about itself

- **!** Height above ground reaches 4.80 m; structures look under-built.
-  Semantics came from the colour heuristic, not a trained model.

## Reproducing this

```bash
python -m ayama.cli delivery results/seed7/run --out results
```

