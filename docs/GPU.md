# Running UNNAT on a GPU box

Everything here works with no data to download: the harness generates a
synthetic town with a known DSM, correct CRS, sun angles and ray-marched
shadows, so every metric it prints is measured against real ground truth. Point
it at your own imagery when you have some.

---

## Quick start

```bash
git clone <repo> && cd unnat
bash scripts/setup_gpu.sh          # detects CUDA, builds .venv, verifies
bash scripts/harness.sh            # doctor, tests, bench, run, ablation
```

Results land in `out/harness/`: `bench.json`, `run_summary.json`,
`ablation.json` + `.md`, the artifact rasters under `run/`, and `harness.log`
with the whole transcript.

### Colab

Open `notebooks/unnat_gpu_harness.ipynb`, set the runtime to GPU, run all.

### Docker

```bash
docker build -t unnat:gpu .
docker run --gpus all --rm -it -v unnat-cache:/cache -v "$PWD:/work" unnat:gpu \
    bash scripts/harness.sh
```

The `/cache` volume holds Hugging Face weights, so restarting the container does
not re-download 1.3 GB of ViT-L.

---

## Regenerating the published results

The site publishes whatever is in `results/`. To replace the CPU numbers with
yours:

```bash
python -m unnat.cli study --out results \
    --backbone dav2-vitl --device cuda --batch 0 --size 2048 --seeds 7,21,33
git add results && git commit -m "results: GPU run" && git push
```

The Pages workflow redeploys on that push and the site renders your environment,
your timings and your metrics. Only the JSON and the web previews are tracked —
the rasters are gitignored and regenerate byte-identically from the seeds.

## Watching a run

Every stage that can take more than a few seconds reports where it is. Silence
during a demo is indistinguishable from a hang, and on a rented GPU it is
indistinguishable from money being wasted.

An interactive terminal gets one line, rewritten in place:

```
  [study 1/3] [depth 12/36] depth  ▕████████░░░░░░░░░░▏ 12/36 chip  4.10 chip/s  ETA 6s  VRAM 4.2/16.0 GB
```

A notebook, a CI log or a redirect to a file gets timestamped lines instead, at
most one every 20 s, because thousands of carriage returns in a log file are
unreadable. The mode is detected automatically; force it with
`--progress rich|plain|none` (or `UNNAT_PROGRESS=plain` in the environment).

The line carries the whole nesting at once — which scene, which stage, which
chip — because "12/36" alone does not tell anyone how much of the run is left.
VRAM is read from the driver, not from torch's allocator, so it includes
whatever else is resident on the card; that is the number that decides your
batch size.

## Presentation output

The study writes publication figures and LaTeX tables alongside the JSON:

```
results/figures/
  fig1_ablation.{png,pdf}            which components earn their place
  fig2_error_by_class.{png,pdf}      where the error lives
  fig3_sun_window.{png,pdf}          the shadow physics window
  fig4_lambda_sensitivity.{png,pdf}  sensitivity to the one free parameter
  fig5_reliability.{png,pdf}         does sigma predict the error
  fig6_qualitative.{png,pdf}         input / reference / prediction / error / sigma
  tables/table1_headline.tex         booktabs, ready to \input{}
  tables/table2_ablation.tex
  tables/table3_by_class.tex
```

PNG at 300 dpi for slides, PDF as vector for LaTeX. Every figure is rendered
from `study.json`, so it cannot drift from the number it illustrates.
Re-plot without re-running inference — captions and colours cost seconds, a GPU
hour does not:

```bash
python -m unnat.cli figures --study results/study.json
python -m unnat.cli study --no-figures ...   # skip them during a sweep
```

Two figures need a completed run on disk (`fig5`, `fig6`) because they read the
per-pixel σ and error rasters; without one they are skipped rather than faked
from the aggregate.

## The four commands

### `doctor` — is this machine ready

```bash
python -m unnat.cli doctor --load dav2-vits,dav2-vitl
```

Prints platform, torch, CUDA, GPU name, VRAM total and free, GDAL, and then
actually loads each named backbone and times it. Exits non-zero if something is
missing, so it works as a CI gate.

### `bench` — how fast, and what fits

```bash
python -m unnat.cli bench --image data/scene.tif \
    --backbones dav2-vits,dav2-vitb,dav2-vitl \
    --chips 518,1024 --batches 1,2,4,8 --device cuda --json out/bench.json
```

Reports wall time, seconds per chip, megapixels per second and **peak VRAM** for
each cell of the sweep. Warm-up is timed separately and excluded, so the first
pass's kernel autotuning does not pollute the average. An out-of-memory error is
recorded as a result for that cell rather than killing the sweep — that is how
you find the batch ceiling.

Batch size `0` means "pick from free VRAM" (`suggest_batch_size`).

### `run` — the full pipeline

```bash
python -m unnat.cli run data/scene.tif --out out/run \
    --backbone dav2-vitl --device cuda --batch 0 \
    --dem sim:data/scene_dtm.tif --ref data/scene_dsm.tif \
    --bootstrap 24 --json out/run_summary.json
```

Streams each stage as it executes, then prints the tier, the anchor census, the
validation table, the global-affine baseline next to it, and the per-class error
breakdown.

Artifacts written to `--out`: `dsm.tif`, `ndsm.tif`, `sigma.tif`, `sem.tif`,
`shadow.tif`, `relative_depth.tif`, `error.tif` (with `--ref`), `texture.jpg`,
quick-look PNGs, and `provenance.json`. All COGs, all openable in QGIS without
UNNAT installed.

`--dem` takes a real bare-earth GeoTIFF, or `sim:<terrain.tif>` to simulate a
Copernicus GLO-30 from a known terrain surface during development. Network DEM
fetching is deliberately not wired up: a run must never silently proceed with a
DEM it could not load.

### `ablate` — which parts earn their place

```bash
python -m unnat.cli ablate data/scene.tif --ref data/scene_dsm.tif \
    --dem sim:data/scene_dtm.tif --backbone dav2-vitl --device cuda --json out/ablation.json
```

Inference runs **once**; every variant then re-solves only the calibration.
That is not just a speed trick — every row sees exactly the same depth field, so
the difference between rows is the thing being ablated and nothing else.

| variant | what it removes |
|---|---|
| `dem_only` | the depth model entirely — just the public DEM resampled. **The floor.** If the method does not beat this row, the depth model is contributing nothing and every other row is measuring a DEM interpolator |
| `global_affine` | everything: one `a, b` for the whole tile (the published baseline) |
| `agmc_no_gate` | the semantic gate — DEM anchors taken on rooftops too |
| `agmc_no_shadow` | shadow-derived height anchors |
| `agmc_no_water` | the water flatness constraint |
| `agmc` | nothing |
| `agmc_bootstrap` | nothing, plus the bootstrap σ field, so ECE and coverage are reported |

Writes both JSON and a markdown table ready to paste into a slide.

---

## Tests

```bash
python -m pytest tests -q              # everything; GPU tests skip with a reason
python -m pytest tests -q -m gpu -v    # GPU-only
python -m pytest tests -q -m "not slow"  # no model weights downloaded
```

The GPU tests exist to prove the GPU path produces the **same** surface as the
CPU path, not that it is fast:

- `test_batching_does_not_change_the_mosaic` — a batch is a scheduling decision,
  not a numerical one (runs on CPU too).
- `test_gpu_and_cpu_agree_on_the_same_chip` — fp16 on CUDA vs fp32 on CPU,
  correlation > 0.995.
- `test_gpu_batching_matches_single_chip_inference` — batched vs single, r > 0.999.
- `test_suggested_batch_size_fits_in_memory` — the suggested batch does not OOM.
- `test_full_pipeline_runs_on_gpu` — end to end on the device the demo will use.

A quiet numerical drift in the GPU path would otherwise surface as an
unexplained metrics shift three stages later.

---

## Tuning knobs that actually matter

| knob | effect |
|---|---|
| `--chip 518` | Depth Anything's native size. Larger chips resize inside the processor, so 1024 costs more and adds no detail. Bench both. |
| `--batch 0` | Auto from free VRAM. Set explicitly to reproduce a timing. |
| `--dtype float16` | Default on CUDA. Roughly 2x, and it moves the surface by far less than the calibration residual — the GPU tests check this. |
| `--overlap 0.25` | Below ~0.15 the overlap band gets too small to harmonise chips against each other. |
| `--bootstrap 24` | 24 solves of a small sparse system. Seconds. `0` turns σ off. |
| `--stride 32` | AGMC lattice stride in pixels. 32 on a 4k tile is a 128×128 lattice, 32k unknowns. |
| `--lam 1.0` | AGMC smoothness weight. Results are flat over roughly 0.25–4, which is what a correctly scaled parameter should look like. Push it to 10+ and AGMC collapses back to a global affine fit. |

## Reference numbers

The published CPU study: three independent 1024 px scenes, `dav2-vits`, chip
512, simulated Copernicus GLO-30, on an 8-core CPU with no CUDA. Full report in
[`results/README.md`](../results/README.md). Use these to tell "the GPU is slow"
apart from "the build is broken" — the metrics should reproduce closely on any
device, and only the timings should change.

```
                       MAE          RMSE          r        1σ cov  edge F1
global affine     5.49 ± 0.85   7.80 ± 0.82   0.16 ± 0.15     -       -
AGMC (default)    3.30 ± 0.08   5.49 ± 0.14   0.71 ± 0.09    0.67    0.26
DEM alone (floor) 3.49 ± 0.05   5.37 ± 0.12   0.71 ± 0.09     -       -

per class   road 1.79 · bare ground 2.79 · vegetation 5.51
            water 7.93 · building 12.94   (MAE, metres)

timings     depth 26.6 s · anchors 3.2 s · calibration 0.4 s
            bootstrap σ (24) 7.8 s · assemble 0.6 s
            whole study, 3 scenes + 4 experiments: 450 s
```

Four things to read off that.

The metrics are measured against **synthetic scenes**, not satellite imagery.

**The floor row is the important one.** AGMC beats the DEM it was anchored to by
5% on MAE, loses by 2% on RMSE, and ties on correlation. On these scenes the
depth model contributes little and the anchor graph is doing the work; `edge F1`
of 0.26 is the signature of a surface with correct terrain and flattened
structures. Use that row to tell a real improvement from a DEM interpolator.

AGMC's spread across scenes (±0.08 m) is a tenth of the baseline's (±0.85 m), so
whatever else is true, the anchor graph is far more stable scene to scene.

The non-inference half of the pipeline costs about 12 s per Mpix, so on a GPU
the run time is essentially the inference time, which is what `bench` measures.

## Expected shape of the results

On a T4 or better, `dav2-vits` at chip 518 should clear several chips per second
and a 4096² tile should finish depth inference in well under a minute. Wall time
is dominated by inference; calibration, bootstrap σ and assembly are seconds.

If `bench` shows CPU-like timings, check `doctor` output for
`cuda_available: false` — that is almost always a torch wheel built for the
wrong CUDA version, and `scripts/setup_gpu.sh` picks the index for you.
