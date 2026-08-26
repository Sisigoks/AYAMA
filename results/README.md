# UNNAT results

Generated 2026-08-26T09:31:49Z by `python -m unnat.cli study`.

Every number here is measured against synthetic scenes whose exact DSM is
known, using only the RGB image plus a simulated public DEM as input. They
test the method end to end; they are **not** a claim about real satellite
imagery, which needs a scene with a reference DSM we do not yet have.

## Environment

- Windows-11-10.0.26200-SP0, python 3.13.5, 8 cpus
- torch 2.13.0+cpu · CUDA available: False
- rasterio 1.5.1 · GDAL 3.12.4
- backbone `dav2-vits`, 1024x1024 px at 0.5 m, chip 512, 3 scenes

## Headline

| metric | AGMC | global affine | DEM alone (floor) |
|---|---|---|---|
| MAE (m) | **3.30 ± 0.08** | 5.49 ± 0.85 | 3.49 ± 0.05 |
| RMSE (m) | 5.49 ± 0.14 | 7.80 ± 0.82 | 5.37 ± 0.12 |
| Pearson r | **0.71 ± 0.09** | 0.16 ± 0.15 | 0.71 ± 0.09 |
| Spearman rho | 0.80 ± 0.06 | - | - |
| bias (m) | -0.60 ± 0.08 | - | - |
| 1σ coverage | 0.67 ± 0.02 | - | - |
| ECE (m) | 2.36 ± 0.14 | - | - |
| δ < 1.25 | - | - | - |
| edge F1 | 0.26 ± 0.01 | - | - |
| slope MAE (deg) | 7.06 ± 0.13 | - | - |

1σ coverage is the honest test of the uncertainty field: for a Gaussian it
should sit near 0.68. A σ that does not predict error is decoration.

**Read the third column first.** `DEM alone` is the public DEM resampled onto
the image grid, with no depth model involved at all. If the method does not
clear that floor by a clear margin on more than one metric, the depth model is
contributing little and the other two columns are measuring a DEM interpolator.

`edge F1` and `δ < 1.25` are the rows that expose this. Both describe structure:
edge F1 asks whether height discontinuities land in the right place, and δ is a
ratio of heights above ground. A surface that reproduces terrain and flattens every
building scores well on MAE and collapses on both of those.

## Error by class

| class | MAE (m) |
|---|---|
| road | 1.79 ± 0.12 |
| bare ground | 2.79 ± 0.10 |
| vegetation | 5.51 ± 0.19 |
| water | 7.93 ± 0.32 |
| building | 12.94 ± 0.22 |

Terrain is close to solved; buildings and canopy dominate the error. That is
where a monocular method fails and saying so is the point of this table.

## Ablation

One inference per scene, every variant re-solving only the calibration, so
each row sees the identical depth field.

| variant | MAE (m) | RMSE (m) | r | anchors |
|---|---|---|---|---|
| `dem_only` | 3.49 | 5.37 | 0.708 | 4021 |
| `global_affine` | 5.50 | 7.80 | 0.162 | 4021 |
| `agmc_no_gate` | 3.30 | 5.50 | 0.709 | 4231 |
| `agmc_no_shadow` | 3.30 | 5.49 | 0.711 | 3955 |
| `agmc_no_water` | 3.35 | 5.59 | 0.692 | 3950 |
| `agmc` | 3.30 | 5.49 | 0.711 | 4021 |
| `agmc_bootstrap` | 3.30 | 5.49 | 0.711 | 4021 |

## Shadow physics window

Shadow-derived building height against sun elevation, with no depth model
involved. The usable band the literature quotes is roughly 20-75 degrees.

| sun elev (deg) | shadow F1 | anchors | median height error (m) | mean weight |
|---|---|---|---|---|
| 15 | 0.29 | 0 | - | - |
| 20 | 0.34 | 0 | - | - |
| 25 | 0.46 | 3 | 8.69 | 0.28 |
| 30 | 0.71 | 50 | 6.41 | 0.64 |
| 40 | 0.75 | 57 | 4.44 | 0.74 |
| 50 | 0.81 | 57 | 1.68 | 0.80 |
| 60 | 0.81 | 58 | 1.70 | 0.87 |
| 70 | 0.73 | 48 | 2.33 | 0.45 |
| 75 | 0.60 | 0 | - | - |
| 80 | 0.38 | 0 | - | - |

## Calibration parameter sensitivity

AGMC has one free parameter, the smoothness weight. If it needed tuning per
scene it would not be a method, it would be a knob.

| lambda | MAE (m) | r |
|---|---|---|
| baseline | 5.75 | 0.271 |
| 0.05 | 3.35 | 0.761 |
| 0.1 | 3.30 | 0.765 |
| 0.25 | 3.24 | 0.768 |
| 0.5 | 3.25 | 0.765 |
| 1.0 | 3.40 | 0.747 |
| 2.0 | 3.68 | 0.719 |
| 4.0 | 4.08 | 0.686 |
| 10.0 | 4.69 | 0.649 |
| 50.0 | 5.78 | 0.626 |

## Throughput

| backbone | chip | batch | chips | wall (s) | s/chip | MPix/s | peak VRAM (MB) |
|---|---|---|---|---|---|---|---|
| dav2-vits | 512 | 1 | 9 | 18.84 | 2.093 | 0.06 | - |
| dav2-vits | 1024 | 1 | 1 | 1.53 | 1.531 | 0.68 | - |

## Mean stage timings

| stage | seconds |
|---|---|
| ingest | 0.0 |
| depth | 22.1 |
| segmentation | 1.0 |
| shadow | 0.3 |
| anchors | 2.3 |
| calibration | 0.3 |
| uncertainty | 8.2 |
| assemble | 0.5 |
| artifacts | 4.5 |
| validation | 7.5 |

## Reproducing this

```bash
bash scripts/setup_gpu.sh          # or scripts/setup.ps1 on Windows
python -m unnat.cli study --backbone dav2-vits --size 1024 --seeds 7,21,33 --out results
```

Took 450s on the machine above. Every artifact is a COG that
opens in QGIS without UNNAT installed.
