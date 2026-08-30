# TRAKSHA delivery benchmark

Phase 3 and Phase 4 on CPU. Generated 2026-08-30T12:51:19Z by
`python -m traksha.cli delivery`.

Every number is measured on the machine below, against a real Phase 2 run.
**Browser GPU rasterisation is not measured** and is not claimed: everything in the
viewer section is CPU work the browser does before a triangle is drawn.

## Environment

- Windows-11-10.0.26200-SP0, python 3.13.5, 8 cpus
- 8 CPU cores  ·  rasterio 1.5.1
- node v20.19.4
- source run `D:\UNNAT\results\zurich`
- scene 1024 x 1024 px at 0.5 m (1.05 Mpix, 512 x 512 m)

## Headline

| | |
|---|---|
| tileset build, tiles only | **4.83 s** (0.22 Mpix/s) |
| tileset build, with the OBJ | **10.50 s** (OBJ alone 5.67 s) |
| output | 4 LODs, 7 tiles |
| payload | **49.5 MB** total, 12.1 MB tiles + 37.3 MB mesh |
| first paint, bytes | 6.08 MB |
| first paint, viewer CPU | **50 ms** |
| whole benchmark | 101 s |

> Build timings are disk-bound and vary with where they are written; an on-access virus scanner can change them several-fold. Every timed build here used one scratch directory, reported above.

## Encoding throughput

The packing arithmetic in isolation, no disk. This is what says whether the
encoding could ever be the bottleneck.

| operation | Mpix/s | seconds |
|---|---|---|
| encode terrain-rgb | 32 | 0.0333 |
| decode terrain-rgb | 55 | 0.0191 |
| encode linear | 26 | 0.0398 |
| decode linear | 47 | 0.0223 |
| normal map | 11 | 0.0946 |
| png encode, terrain-rgb | 1 | 1.5672 |
| png encode, linear | 4 | 0.2754 |

**PNG compression dominates.** Packing pixels runs at 32 Mpix/s;
compressing them runs at 4 Mpix/s - a factor of 8.
Nothing in the encoder is worth optimising until the compressor is.

## Where one build's time goes

LOD 0 only, 4 tiles of 512 px (+1 px pad), 3.33 s accounted for.

| stage | seconds | share |
|---|---|---|
| png write dsm | 1.715 | 51.4% |
| png write linear layers | 0.822 | 24.6% |
| png write normals | 0.488 | 14.6% |
| normals | 0.146 | 4.4% |
| encode linear layers | 0.127 | 3.8% |
| encode dsm | 0.035 | 1.0% |
| cut | 0.002 | 0.1% |
| crop | 0.000 | 0.0% |

## Tile size

| tile px | LODs | tiles | files | seconds | payload |
|---|---|---|---|---|---|
| 128 | 4 | 85 | 510 | 6.82 | 12.20 MB |
| 256 | 4 | 22 | 132 | 5.05 | 12.11 MB |
| 512 | 4 | 7 | 42 | 5.48 | 12.14 MB |
| 1024 | 4 | 4 | 24 | 7.01 | 12.17 MB |

**Tile size barely moves the payload.** Across a 21x range in file count (24 to 510 files), the total varies by only 0.7% (12.11 to 12.20 MB) and build time by 1.96 s.

PNG headers and per-file compression contexts were expected to punish small
tiles; at these sizes they do not, because the pixel data dominates either
way. So tile size is free to be chosen for what it actually affects - how
much a viewer can cull, and how many requests it makes - rather than for
bytes. The default of 512 keeps the file count in double digits.

## Mesh decimation

| stride | vertices | triangles | seconds | size | bytes/triangle |
|---|---|---|---|---|---|
| 2 | 262,144 | 522,242 | 2.96 | 31.2 MB | 60 |
| 4 | 65,536 | 130,050 | 0.67 | 7.2 MB | 55 |

OBJ is a text format, so size tracks triangle count almost exactly. This is
the argument for glTF, and the reason `--obj-stride` and `--no-mesh` exist.

## What full precision costs

The linear encoding spends all 24 bits, so its low byte is incompressible
noise. Each row below keeps only the top N bits of the code and zeroes the
rest, which is what a narrower field would really store - and what lets PNG
collapse them.

| layer | precision | step | size | vs 24-bit | worst error |
|---|---|---|---|---|---|
| ndsm | 24-bit (shipped) | 3.193e-06 m | 1264 kB | 1.00x | 1.91e-06 m |
| ndsm | 16-bit | 0.0008174 m | 844 kB | 0.67x | 0.00118 m |
| ndsm | 12-bit | 0.01308 m | 567 kB | 0.45x | 0.0189 m |
| ndsm | 8-bit | 0.2101 m | 224 kB | 0.18x | 0.303 m |
| sigma | 24-bit (shipped) | 3.049e-09 m | 2162 kB | 1.00x | 0 m |
| sigma | 16-bit | 7.806e-07 m | 886 kB | 0.41x | 9.54e-07 m |
| sigma | 12-bit | 1.249e-05 m | 382 kB | 0.18x | 1.81e-05 m |
| sigma | 8-bit | 0.0002006 m | 93 kB | 0.04x | 0.000293 m |
| error | 24-bit (shipped) | 5.384e-06 m | 2744 kB | 1.00x | 3.81e-06 m |
| error | 16-bit | 0.001378 m | 1789 kB | 0.65x | 0.00204 m |
| error | 12-bit | 0.02206 m | 1166 kB | 0.42x | 0.033 m |
| error | 8-bit | 0.3542 m | 490 kB | 0.18x | 0.524 m |

**ndsm at 12 bits, sigma at 12 bits, error at 12 bits** resolves every layer to better than 0.1% of its
own range, and takes the linear layers from 6.17 MB to
2.12 MB - a saving of **4.06 MB**, 66% of their bytes.

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
| 0 | ndsm | linear | 3.19e-06 | 1.91e-06 m | yes |
| 0 | sigma | linear | 3.05e-09 | 0 m | yes |
| 0 | error | linear | 5.38e-06 | 3.81e-06 m | yes |
| 1 | dsm | terrain-rgb | 0.1 | 0.05 m | yes |
| 1 | ndsm | linear | 3.19e-06 | 1.91e-06 m | yes |
| 1 | sigma | linear | 3.05e-09 | 0 m | yes |
| 1 | error | linear | 5.38e-06 | 3.81e-06 m | yes |
| 2 | dsm | terrain-rgb | 0.1 | 0.05 m | yes |
| 2 | ndsm | linear | 3.19e-06 | 1.91e-06 m | yes |
| 2 | sigma | linear | 3.05e-09 | 0 m | yes |
| 2 | error | linear | 5.38e-06 | 3.81e-06 m | yes |
| 3 | dsm | terrain-rgb | 0.1 | 0.05 m | yes |
| 3 | ndsm | linear | 3.19e-06 | 1.91e-06 m | yes |
| 3 | sigma | linear | 3.05e-09 | 0 m | yes |
| 3 | error | linear | 5.38e-06 | 3.81e-06 m | yes |

## Payload

| layer | bytes | share of tiles |
|---|---|---|
| error | 3.68 MB | 30.4% |
| sigma | 2.94 MB | 24.3% |
| normal | 2.46 MB | 20.3% |
| ndsm | 1.73 MB | 14.3% |
| dsm | 0.78 MB | 6.4% |
| texture | 0.54 MB | 4.4% |
| **mesh** | 37.34 MB | - |

| LOD | bytes |
|---|---|
| lod0 | 8.82 MB |
| lod1 | 2.45 MB |
| lod2 | 0.67 MB |
| lod3 | 0.18 MB |

**First paint** fetches 6.08 MB: geometry,
normals, the default drape and the two layers the cursor readout needs.

## Viewer CPU

Measured against the real `web/app.js` under node, best of five with a
warm-up. This is the work the browser does before the GPU is involved.

| operation | ms |
|---|---|
| parse tileset.json | 0.23 |
| decode terrain-rgb, whole scene | 6.81 |
| decode linear, whole scene | 5.84 |
| decode terrain-rgb, reusing the buffer | 3.48 |
| tileGeometry, one tile | 2.40 |
| gridIndices, one tile | 4.38 |
| colourize, one tile | 4.82 |
| build a 256-entry LUT | 0.19 |
| renderPanels (jsdom) (jsdom not installed) | - |

| for the whole scene | ms |
|---|---|
| decode every data layer | 23.4 |
| build geometry for every tile | 27.1 |
| **CPU before first paint** | **50.5** |
| re-colour on a layer switch | 19.3 |

Decode throughput: **154 Mpix/s** terrain-rgb, 180 Mpix/s linear.

**Reusing the output buffer is 49% faster** (3.5 ms against 6.8 ms). The viewer currently
allocates a fresh `Float32Array` per tile per layer; it does not have to.

## What the tileset says about itself

-  Semantics came from the colour heuristic, not a trained model.

## Reproducing this

```bash
python -m traksha.cli delivery results/zurich
```

