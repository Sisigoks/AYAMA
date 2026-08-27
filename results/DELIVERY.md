# AYAMA delivery benchmark

Phase 3 and Phase 4 on CPU. Generated 2026-08-27T05:21:31Z by
`python -m ayama.cli delivery`.

Every number is measured on the machine below, against a real Phase 2 run.
**GPU rasterisation is not measured** and is not claimed: everything in the
viewer section is CPU work the browser does before a triangle is drawn.

## Environment

- Windows-11-10.0.26200-SP0, python 3.13.5, 8 cpus
- CUDA available: False  ·  rasterio 1.5.1
- node v20.19.4
- source run `D:\UNNAT\results\seed7\run`
- scene 1024 x 1024 px at 0.5 m (1.05 Mpix, 512 x 512 m)

## Headline

| | |
|---|---|
| tileset build, tiles only | **2.59 s** (0.40 Mpix/s) |
| tileset build, with the OBJ | **5.07 s** (OBJ alone 2.47 s) |
| output | 4 LODs, 7 tiles |
| payload | **43.1 MB** total, 9.1 MB tiles + 34.0 MB mesh |
| first paint, bytes | 4.34 MB |
| first paint, viewer CPU | **35 ms** |
| whole benchmark | 68 s |

> Build timings are disk-bound and vary with where they are written; an on-access virus scanner can change them several-fold. Every timed build here used one scratch directory, reported above.

## Encoding throughput

The packing arithmetic in isolation, no disk. This is what says whether the
encoding could ever be the bottleneck.

| operation | Mpix/s | seconds |
|---|---|---|
| encode terrain-rgb | 33 | 0.0322 |
| decode terrain-rgb | 47 | 0.0224 |
| encode linear | 22 | 0.0477 |
| decode linear | 44 | 0.0239 |
| normal map | 11 | 0.0917 |
| png encode, terrain-rgb | 3 | 0.3687 |
| png encode, linear | 5 | 0.2076 |

**PNG compression dominates.** Packing pixels runs at 33 Mpix/s;
compressing them runs at 5 Mpix/s - a factor of 6.
Nothing in the encoder is worth optimising until the compressor is.

## Where one build's time goes

LOD 0 only, 4 tiles of 512 px (+1 px pad), 1.46 s accounted for.

| stage | seconds | share |
|---|---|---|
| png write linear layers | 0.708 | 48.4% |
| png write dsm | 0.337 | 23.1% |
| normals | 0.150 | 10.3% |
| encode linear layers | 0.123 | 8.4% |
| png write normals | 0.110 | 7.5% |
| encode dsm | 0.032 | 2.2% |
| cut | 0.002 | 0.1% |
| crop | 0.000 | 0.0% |

## Tile size

| tile px | LODs | tiles | files | seconds | payload |
|---|---|---|---|---|---|
| 128 | 4 | 85 | 510 | 2.85 | 9.15 MB |
| 256 | 4 | 22 | 132 | 2.81 | 9.08 MB |
| 512 | 4 | 7 | 42 | 3.45 | 9.08 MB |
| 1024 | 4 | 4 | 24 | 3.06 | 9.10 MB |

**Tile size barely moves the payload.** Across a 21x range in file count (24 to 510 files), the total varies by only 0.8% (9.08 to 9.15 MB) and build time by 0.64 s.

PNG headers and per-file compression contexts were expected to punish small
tiles; at these sizes they do not, because the pixel data dominates either
way. So tile size is free to be chosen for what it actually affects - how
much a viewer can cull, and how many requests it makes - rather than for
bytes. The default of 512 keeps the file count in double digits.

## Mesh decimation

| stride | vertices | triangles | seconds | size | bytes/triangle |
|---|---|---|---|---|---|
| 1 | 1,048,576 | 2,093,058 | 9.59 | 139.1 MB | 66 |
| 2 | 262,144 | 522,242 | 2.40 | 33.6 MB | 64 |
| 4 | 65,536 | 130,050 | 0.55 | 7.8 MB | 60 |
| 8 | 16,384 | 32,258 | 0.14 | 1.8 MB | 57 |

OBJ is a text format, so size tracks triangle count almost exactly. This is
the argument for glTF, and the reason `--obj-stride` and `--no-mesh` exist.

## What full precision costs

The linear encoding spends all 24 bits, so its low byte is incompressible
noise. Each row below keeps only the top N bits of the code and zeroes the
rest, which is what a narrower field would really store - and what lets PNG
collapse them.

| layer | precision | step | size | vs 24-bit | worst error |
|---|---|---|---|---|---|
| ndsm | 24-bit (shipped) | 1.646e-08 m | 1436 kB | 1.00x | 1.49e-08 m |
| ndsm | 16-bit | 4.214e-06 m | 934 kB | 0.65x | 5.84e-06 m |
| ndsm | 12-bit | 6.744e-05 m | 579 kB | 0.40x | 9.29e-05 m |
| ndsm | 8-bit | 0.001083 m | 190 kB | 0.13x | 0.00148 m |
| sigma | 24-bit (shipped) | 2.652e-09 m | 2416 kB | 1.00x | 0 m |
| sigma | 16-bit | 6.79e-07 m | 1066 kB | 0.44x | 9.54e-07 m |
| sigma | 12-bit | 1.087e-05 m | 476 kB | 0.20x | 1.6e-05 m |
| sigma | 8-bit | 0.0001745 m | 123 kB | 0.05x | 0.000255 m |
| error | 24-bit (shipped) | 3.437e-06 m | 2368 kB | 1.00x | 1.91e-06 m |
| error | 16-bit | 0.0008799 m | 816 kB | 0.34x | 0.00131 m |
| error | 12-bit | 0.01408 m | 407 kB | 0.17x | 0.021 m |
| error | 8-bit | 0.2261 m | 113 kB | 0.05x | 0.337 m |

**ndsm at 12 bits, sigma at 12 bits, error at 12 bits** resolves every layer to better than 0.1% of its
own range, and takes the linear layers from 6.22 MB to
1.46 MB - a saving of **4.76 MB**, 76% of their bytes.

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
| 0 | ndsm | linear | 1.65e-08 | 1.49e-08 m | yes |
| 0 | sigma | linear | 2.65e-09 | 0 m | yes |
| 0 | error | linear | 3.44e-06 | 1.91e-06 m | yes |
| 1 | dsm | terrain-rgb | 0.1 | 0.05 m | yes |
| 1 | ndsm | linear | 1.65e-08 | 1.49e-08 m | yes |
| 1 | sigma | linear | 2.65e-09 | 0 m | yes |
| 1 | error | linear | 3.44e-06 | 1.91e-06 m | yes |
| 2 | dsm | terrain-rgb | 0.1 | 0.05 m | yes |
| 2 | ndsm | linear | 1.65e-08 | 1.49e-08 m | yes |
| 2 | sigma | linear | 2.65e-09 | 0 m | yes |
| 2 | error | linear | 3.44e-06 | 1.91e-06 m | yes |
| 3 | dsm | terrain-rgb | 0.1 | 0.05 m | yes |
| 3 | ndsm | linear | 1.65e-08 | 1.49e-08 m | yes |
| 3 | sigma | linear | 2.65e-09 | 0 m | yes |
| 3 | error | linear | 3.44e-06 | 1.91e-06 m | yes |

## Payload

| layer | bytes | share of tiles |
|---|---|---|
| sigma | 3.25 MB | 35.8% |
| error | 3.19 MB | 35.2% |
| ndsm | 1.95 MB | 21.5% |
| texture | 0.45 MB | 5.0% |
| dsm | 0.14 MB | 1.5% |
| normal | 0.09 MB | 1.0% |
| **mesh** | 33.98 MB | - |

| LOD | bytes |
|---|---|
| lod0 | 6.70 MB |
| lod1 | 1.77 MB |
| lod2 | 0.48 MB |
| lod3 | 0.13 MB |

**First paint** fetches 4.34 MB: geometry,
normals, the default drape and the two layers the cursor readout needs.

## Viewer CPU

Measured against the real `web/app.js` under node, best of five with a
warm-up. This is the work the browser does before the GPU is involved.

| operation | ms |
|---|---|
| parse tileset.json | 0.08 |
| decode terrain-rgb, whole scene | 4.29 |
| decode linear, whole scene | 4.06 |
| decode terrain-rgb, reusing the buffer | 3.44 |
| tileGeometry, one tile | 2.47 |
| gridIndices, one tile | 2.25 |
| colourize, one tile | 1.85 |
| build a 256-entry LUT | 0.08 |
| renderPanels (jsdom) | 3.36 |

| for the whole scene | ms |
|---|---|
| decode every data layer | 16.2 |
| build geometry for every tile | 18.8 |
| **CPU before first paint** | **35.1** |
| re-colour on a layer switch | 7.4 |

Decode throughput: **244 Mpix/s** terrain-rgb, 258 Mpix/s linear.

**Reusing the output buffer is 20% faster** (3.4 ms against 4.3 ms). The viewer currently
allocates a fresh `Float32Array` per tile per layer; it does not have to.

## What the tileset says about itself

- **!!** Predicted height above ground reaches only 0.28 m (99th percentile 0.17 m) on a scene where 3.0% of pixels are classified as building. The calibration scale field has collapsed to its floor, so this surface is terrain with the structures flattened. See README section 5. Raise the vertical exaggeration to see what little relief there is - it is a defect, not a rendering choice.
-  Semantics came from the colour heuristic, not a trained model.
-  Anchored to a simulated DEM (simulated copernicus from scene_dtm.tif), not a real product.

## Reproducing this

```bash
python -m ayama.cli delivery results/seed7/run --out results
```

