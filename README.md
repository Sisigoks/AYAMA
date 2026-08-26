# UNNAT — उन्नत

**Metric elevation from a single image.** Relative depth from a pretrained
backbone, converted to metres by **Chhaya** (छाया), an anchor-graph calibration
engine, then delivered as a COG DSM and a textured 3D mesh.

Build order is the phase ladder in the spec: every phase ends with something
that runs on its own, and nothing depends on the next phase working.

| Phase | Output | Status |
|---|---|---|
| P1 Baseline | relative depth raster | **done** |
| P2 Calibration | metric DSM + σ + metrics | **done, measured** |
| P3 3D | textured mesh in the browser | not started |
| P4 Product | web app | not started |

## Measured results

Full report in [results/README.md](results/README.md); raw data in
`results/study.json`; rendered interactively by [site/](site/).

| | AGMC | global affine | **DEM alone (floor)** |
|---|---|---|---|
| MAE | **3.30 ± 0.08 m** | 5.49 ± 0.85 m | 3.49 ± 0.05 m |
| RMSE | 5.49 ± 0.14 m | 7.80 ± 0.82 m | **5.37 ± 0.12 m** |
| Pearson r | 0.71 ± 0.09 | 0.16 ± 0.15 | 0.71 ± 0.09 |
| 1σ coverage | **0.67 ± 0.02** (target 0.68) | – | – |
| edge F1 | 0.26 ± 0.01 | – | – |

Three independent synthetic scenes at 1024×1024 / 0.5 m with exact ground truth,
`dav2-vits` on CPU, simulated Copernicus GLO-30. 450 s to regenerate:
`python -m unnat.cli study --out results`.

**Read the third column.** The reconstruction beats the public DEM it was
anchored to by 5% on MAE, loses to it by 2% on RMSE, and matches its correlation
exactly. On these scenes the anchor graph is carrying the result and the depth
model contributes little. The large gain over the global-affine baseline measures
how badly one scale fits a contaminated depth field — not how much that field
knows. `edge F1` is the row that gives it away: 0.26 is what a surface with the
terrain right and the structures flattened looks like.

The uncertainty field is the part that is genuinely working: 0.67 measured
against a Gaussian's 0.68 means the error bars mean what they say.

---

## Setup

```bash
bash scripts/setup_gpu.sh                                   # Linux/WSL, detects CUDA
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1  # Windows
```

By hand:

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt                   # core pipeline
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
.venv/bin/pip install transformers
```

The core install has no torch in it on purpose: ingest, tiling, blending,
raster IO, calibration, metrics and the synthetic scene all run without it, so
nobody is blocked on a 2 GB download. `--backbone synthetic` exercises the
entire path with no weights at all — a plumbing check, never a result.

## Commands

```bash
python -m unnat.cli doctor --load dav2-vits   # is this machine ready, and how fast
python -m unnat.cli synth --out data/scene.tif --size 2048   # town + known DSM + real shadows
python -m unnat.cli info  data/scene.tif      # everything we can read about an image
python -m unnat.cli depth data/scene.tif --out out/depth.tif --preview     # Phase 1
python -m unnat.cli run   data/scene.tif --out out/run \
    --dem sim:data/scene_dtm.tif --ref data/scene_dsm.tif --json out/run.json
python -m unnat.cli bench  --backbones dav2-vits,dav2-vitl --chips 518,1024 --batches 1,2,4
python -m unnat.cli ablate data/scene.tif --ref data/scene_dsm.tif --dem sim:data/scene_dtm.tif
```

**Full harness in one command:** `bash scripts/harness.sh` — doctor, tests,
throughput sweep, full run, ablation table, all written to `out/harness/`.

**The study that produces `results/`:**

```bash
python -m unnat.cli study --out results                      # CPU defaults
# on a GPU box:
python -m unnat.cli study --out results --backbone dav2-vitl \
    --device cuda --batch 0 --size 2048
```

See [docs/GPU.md](docs/GPU.md) for the GPU box, Docker and Colab paths.

## Site

`site/` is a static page that renders `results/study.json` — it invents nothing,
so re-running the study and pushing updates every number and image on it.

```bash
python scripts/serve.py        # assemble exactly as Pages does, and serve it
node scripts/check_site.js     # headless render check (needs: npm install jsdom)
```

`.github/workflows/pages.yml` publishes it on every push that touches `site/` or
`results/`. Enable it once under **Settings → Pages → Source: GitHub Actions**.

## Tests

```bash
python -m pytest tests -q                 # GPU tests skip with a reason
python -m pytest tests -q -m gpu -v       # GPU-only, on the GPU box
```

## Layout

```
unnat/
  core/        types.py (the contracts), geo.py, solar.py, ingest.py
  depth/       backbones/{base,hf,synthetic}.py, infer.py
  semantics/   segment.py (heuristic or raster), shadow.py
  chhaya/      agmc.py, anchors.py, ladder.py, uncertainty.py
  physics/     soft shadow, test-time refinement            (next)
  dsm/         assemble.py, cog.py — every artifact QGIS can open
  measure/     derive.py — slope, roughness, profile, buildings
  mesh/        terrain-rgb, normals, padding                (P3)
  eval/        metrics, ablation, bench, synthetic_scene, simulate
  api/         pipeline.py — the whole method in one file
web/           React + Vite + Three.js                      (P3/P4)
scripts/       setup_gpu.sh, setup.ps1, harness.sh
notebooks/     unnat_gpu_harness.ipynb (Colab)
```

One deviation from the spec's layout: a single importable `unnat` package
instead of `packages/unnat.core/` and friends. Import paths are exactly as
specified (`unnat.core.ingest`, `unnat.chhaya.agmc`); directory names with dots
are not importable.

## Contracts

`unnat/core/types.py` is the interface between workstreams. Every stage is a
pure function `stage(input) -> output`, no globals, which is what makes the
ablation table cheap to generate — and it is why `ablate` can run inference once
and re-solve only the calibration for every variant.

## Conventions worth stating once

- **Sign.** The backbone returns higher values for surfaces closer to the
  sensor. From nadir, closer means higher elevation, so relative depth maps
  monotonically to height. There is no flip anywhere in the pipeline.
- **Metres.** `geo.gsd_metres` is the only place allowed to answer "how many
  metres is one pixel", and it converts degrees when the CRS is geographic.
- **Missing metadata is missing**, not defaulted. A scene without sun angles
  reports `has_sun = False` and drops to a lower calibration tier rather than
  inventing a number.
- **Shadow anchors are relative.** Each says "this roof stands h metres above
  the ground at its foot", carrying a reference pixel. A shadow measures a
  height, never an elevation; letting one enter as an elevation is how a good
  height anchor silently becomes a bad datum anchor.
- **A batch is a scheduling decision, not a numerical one.** Batched inference
  must produce the identical mosaic; there is a test for it.
- **Simulated inputs are labelled.** `--dem sim:` and `--backbone synthetic`
  both stamp their provenance into every artifact they touch.
