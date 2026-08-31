# TRAKSHA — architecture audit and design assessment

Written before any code changed, against commit `89b7063`. Every number here was
measured on this machine; nothing is quoted from the README without re-checking
it. The purpose is to say what the system *is*, what is actually wrong with it,
and what can and cannot be built on the hardware available.

---

## 1. What exists

### 1.1 Where a run begins

There are three entry points and they all converge on one function.

```
traksha.cli run <image>          ─┐
traksha.cli dataset <dir>        ─┼─►  traksha/api/pipeline.py :: run()
POST /api/jobs  (web upload)     ─┘         └─ JobStore._run() wraps it
```

`pipeline.run()` is 200 lines and is the whole method. It is a straight line of
ten `with clock.stage(...)` blocks. There is no orchestrator, no state machine
and no queue: `_Clock` times each block and fires a `StageEvent` at its start
and end.

### 1.2 The ten stages, and what each produces

| # | stage | module | output held in memory | written to disk |
|---|-------|--------|----------------------|-----------------|
| 1 | `ingest` | `core/ingest.py` | `Scene` (rgb, meta) | – |
| 2 | `depth` | `depth/infer.py` + `backbones/hf.py` | `DepthField.relative` | `relative_depth.tif` |
| 3 | `segmentation` | `semantics/segment.py` | `sem` — **(H,W) uint8, 5 classes** | `sem.tif` |
| 4 | `shadow` | `semantics/shadow.py` | `shadow` bool | `shadow.tif` |
| 5 | `anchors` | `chhaya/ladder.py`, `anchors.py` | `anchors`, DEM | – |
| 6 | `calibration` | **`chhaya/agmc.py`** | `CalibrationField` → surface | – |
| 7 | `uncertainty` | `chhaya/uncertainty.py` | `sigma` | `sigma.tif` |
| 8 | `assemble` | `dsm/assemble.py` | `ElevationSurface` (dsm, ndsm) | `dsm.tif`, `ndsm.tif` |
| 9 | `artifacts` | `dsm/cog.py` | – | the COGs + PNG previews |
| 10 | `validation` | `eval/metrics.py` | metrics dict | `error.tif` |

Phase 3/4 (tiles and mesh) run **outside** `run()`, in `mesh/build.py`,
triggered separately by `traksha.cli mesh` or by `JobStore._run` after the
pipeline returns.

### 1.3 Artifacts

One flat directory per run. There is no per-phase artifact tree, no schema
version on any artifact, and no manifest joining them — except
`provenance.json`, which is a flat dict of strings written once at the end, and
`tiles3d/tileset.json`, which is the only versioned contract in the system
(`traksha_tileset_version`).

### 1.4 How the frontend learns about progress

```
JobStore._run  ──emit()──►  job.events[]  (a plain list, in memory)
                                 │
              GET /api/jobs/{id}/events  ──►  SSE, framed as `event: stage`
              GET /api/jobs/{id}         ──►  Job.public()
                                 │
                          web/src/App.jsx :: useJob()
```

### 1.5 How the 3D object is generated

```
dsm.tif ──► mesh/adaptive.py :: plan()      per-block bilinear error
        ──► mesh/adaptive.py :: triangles() centre-fan stitching
        ──► mesh/obj.py      :: write_obj_adaptive()
        ──► mesh/tiles.py               24-bit RGB height tiles
        ──► web/src/renderer.js         one VBO per tile, WebGL
```

`adaptive_mesh()` takes a height *raster* and returns one vertex per retained
grid node:

```python
xyz = np.stack([cols * step, (h - 1 - rows) * step, z[rows, cols]], 1)
```

This is the single most important line in the audit. See §2.2.

---

## 2. What is actually wrong

### 2.1 The progress UI is not stuck — it is never connected

Two independent defects, both confirmed by running a real job and reading the
wire.

**(a) The SSE channel delivers nothing.** The server frames every message as a
*named* event:

```
event: stage
data: {"stage": "depth", "status": "running", "detail": "", "pct": 0.0, "t": 0.96}
```

The client listens with `es.onmessage`, which by specification fires **only for
frames with no `event:` field** (that is, the default type `message`). There is
no `addEventListener('stage', …)` anywhere in the app. So every frame the server
sends is discarded by the browser. Measured event names on the wire: `['stage']`.

**(b) The poll fallback carries no phase.** `Job.public()` returns
`id, status, created, finished, error, params, summary, notes, stages,
elapsed_s`. There is no current-phase field at all. Measured, polling every two
seconds through a complete run:

```json
{"status": "queued",  "stage": null, "detail": null, "pct": null}
{"status": "running", "stage": null, "detail": null, "pct": null}
{"status": "running", "stage": null, "detail": null, "pct": null}
```

`Progress` in `components.jsx` computes each row's state as
`done.has(s) ? 'done' : job.stage === s ? 'active' : 'todo'`. With (a) `done` is
always empty, and with (b) `job.stage` is always `undefined`, so **every row
renders `todo` from submission to completion**, and then the screen swaps to the
viewer. The bar is not stuck; it was never connected.

A third, smaller defect: the SSE handler does `done.add(d.stage)` without
checking `d.status`, so had the channel worked it would have marked a stage
complete the moment it *started*.

### 2.2 The "3D model" is a single connected sheet, and that is the whole problem

There is no point cloud, no volume, no instance and no building anywhere in the
geometry pipeline. `adaptive_mesh` emits one vertex per grid node and
triangulates the entire grid as one manifold sheet. The consequences are
structural, not tuning problems:

* **A building wall is a ramp.** A 20 m facade between a roof node and the
  adjacent ground node becomes a triangle spanning one GSD horizontally and 20 m
  vertically. It is welded to both.
* **Adjacent buildings are connected.** Two blocks with a 4 m alley between them
  share a continuous triangle strip that dips into the alley and back out.
  Nothing in the mesh knows they are two objects.
* **The footprint is not represented.** No edge in the mesh corresponds to a
  building outline; outlines fall wherever the adaptive block boundaries land.
* **Adaptive decimation is edge-blind.** `_bilinear_block_error` retains blocks
  by height curvature alone. It has no notion of a structural boundary, so it
  will coarsen across a roof edge whenever that block's bilinear residual is
  small.

This is why buildings read as bumps under a blanket rather than as raised
objects. It is a **topology** defect, not a depth-accuracy defect, and no depth
model and no calibration can fix it.

### 2.3 The only structural knowledge in the system is a colour heuristic

`semantics/segment.py` produces a 5-class raster from excess-green, excess-blue,
saturation, luminance and local texture. It has:

* **no instances** — `BUILDING` is one label over the whole image;
* **no confidence** — the return type is `uint8` and nothing else;
* a building rule that is `bright & textured` when no nDSM is available, which is
  the case on the first and only pass.

The README already documents this class as untrustworthy (§3.4), and the viewer
carries a permanent NOTE saying so. Every downstream consumer — the DEM anchor
gate, `extract_dtm`, `evaluate_by_class` — is reading it.

### 2.4 What is *not* wrong

Worth stating, because the plan asks for redesign in places where the existing
code is already correct:

* **Ground estimation already exists and is already local.** `dsm/assemble.py ::
  extract_dtm` keeps ground-classified pixels, carries them under buildings by a
  nearest-neighbour fill, smooths with a 30 m Gaussian, and clamps so terrain can
  never rise above measured ground. It does **not** assume a constant Z plane.
  What is missing is per-instance ground statistics, not ground estimation.
* **Chhaya/AGMC is sound and stays untouched.** `H(p) = a(p)·D(p) + b(p)` on a
  32 px lattice, IRLS + Huber. No change is required or planned.
* **The renderer is not the cause of the visual defect.** It was verified in a
  real browser: correct camera, correct normals-from-heightfield shading, no
  header overlap, no console errors, 2.09 M triangles drawn. It renders the mesh
  it is given, accurately. The mesh is the problem.

---

## 3. Measured baseline

Four swisstopo city scenes, `dav2-vitl`, from `results/dataset.json`:

| metric | value |
|---|---|
| MAE | **9.04 m** |
| RMSE | 11.46 m |
| Pearson r | 0.769 |
| edge F1 | **0.604** |
| δ < 1.25 | 0.174 |
| 1σ coverage | 0.426 |
| nDSM MAE | **5.41 m** (flat-ground floor 7.59 m; DEM-only floor 8.47 m) |
| mean true height | 14.38 m |
| mean predicted height | 5.25 m → **36.5 % of relief recovered** |

Stage timings, one 1024 px scene (seconds):

```
ingest 0.5 │ depth 165.9 │ segmentation 0.8 │ shadow 0.2 │ anchors 0.6
calibration 1.9 │ uncertainty 3.1 │ assemble 0.7 │ artifacts 5.2 │ validation 6.0
```

**Depth is 90 % of the run.** Any new stage has to be costed against that.

There is no building-separation metric, no instance metric and no mesh-quality
metric in the repository today. Those have to be built before any claim about
separation can be made.

---

## 4. What this machine can actually run

| | |
|---|---|
| Python | 3.13.5 |
| torch | 2.13.0**+cpu** — `cuda.is_available() == False`, 4 threads |
| available | numpy, scipy, rasterio, transformers 5.15.1 |
| **not installable** | **open3d — no cp313 wheels exist** (`No matching distribution found`) |
| installable | trimesh, scikit-image (both publish cp313 wheels) |

### 4.1 SAM 2 is available, and it is fast enough — measured

`transformers` 5.15.1 ships the official SAM 2 architecture as `Sam2Model`,
`Sam2Processor`, `Sam2VisionModel`. No git install, no `sam2` package, and it
loads through the same Hugging Face path the depth backbones already use.
Measured here with `facebook/sam2.1-hiera-tiny`:

```
parameters      31.4 M
load            23.6 s   (once per process)
image encode     3.1 s   at 1024 px
16 point prompts 0.6 s   after the encode
64 point prompts 10.6 s  after the encode
```

Two consequences for the design:

* **Automatic mask generation is not provided.** `transformers` exposes the
  promptable model only; there is no `SAM2AutomaticMaskGenerator`. The AMG loop —
  point grid, batched prompts, stability-score and predicted-IoU filtering, mask
  NMS, small-region removal — has to be implemented. That is a specified
  algorithm, not a research problem.
* **Grid density is the cost knob.** A 32×32 grid (1024 points) is roughly three
  minutes on this CPU and would about double total runtime. A 16×16 grid is
  roughly forty seconds. Grid size therefore belongs in config, and the benchmark
  has to include it.

### 4.2 Open3D, TSDF and Poisson: unavailable, and also the wrong tool

Open3D cannot be installed on Python 3.13. That matters less than it appears,
because both volumetric methods are poor fits for this data, and it is more
honest to say so before benchmarking than after:

* **TSDF fusion integrates *many* depth images into a volume.** This pipeline has
  exactly one nadir view. With a single view there is nothing to fuse: the TSDF
  would be a re-rasterisation of the input at the cost of a volume.
* **Poisson reconstruction wants oriented normals over a closed surface and
  returns a smooth watertight result.** On a 2.5 D height field it would round
  precisely the roof edges this work exists to keep sharp, and would hallucinate
  closure beneath the terrain.
* **The data is already organised.** A depth raster on a regular georeferenced
  grid has known connectivity. Poisson and TSDF exist to *recover* connectivity
  that unstructured clouds lack. Discarding it and paying to re-derive it is a
  downgrade.

The defensible method for this data is **instance-aware triangulation of the
organised grid**: cut the sheet along instance boundaries and emit explicit
vertical walls. That addresses §2.2 directly, keeps every vertex on a calibrated
value, and costs milliseconds. It will be benchmarked against the current sheet
and against a plane-regularised variant. `trimesh` covers topology validation and
glTF/GLB export; `scipy.ndimage` and `scikit-image` cover morphology, connected
components and robust plane fitting.

---

## 5. Design assessment

### 5.1 Where the plan is adopted as written

* Segmentation before depth, as an explicit stage with its own artifact. SAM 2 is
  class-agnostic and image-only, so it genuinely can run first.
* Instance-level building extraction with id, mask, bbox, area, confidence and
  boundary.
* Per-instance robust height statistics (median / MAD / percentile), not means.
* Explicit typed artifacts with schema, provenance and timestamps.
* A backend-authoritative phase state machine with weights, persisted to disk.
* Debug layers in the viewer, and a building-separation metric with a threshold.

### 5.2 Where the plan needs amending, and why

**"Depth estimation conditioned by structure" cannot mean a conditioned depth
model here.** Depth Anything V2 takes an image and nothing else. Conditioning it
on masks means fine-tuning, which needs a GPU, supervision and a training set;
this machine has four CPU threads and the study has four scenes. What is
achievable and measurable is *structure-aware post-processing* of the depth
field: edge-aware refinement keyed on instance boundaries, per-instance outlier
rejection, and instance-aware anchor harvesting. The pipeline order still changes
— segmentation must run first to inform all three — so the architectural change
the plan asks for is real. I will not claim a conditioned depth model.

**The "raised buildings" objective is won in triangulation, not in depth.** §2.2
shows the sheet topology is what welds buildings to the ground. Fixing depth
alone leaves the ramp. This reorders the value: the geometry work matters more
than the depth-refinement work, and I expect the ablation to show exactly that.

**Instance geometry must not alter the calibrated DSM.** Per the plan's §32, the
cut mesh is a *render-space* product derived from the calibrated surface.
`dsm.tif`, `ndsm.tif` and `sigma.tif` keep their current values and provenance.
Any per-instance regularisation (plane fitting, wall insertion) is written as a
separate artifact and labelled as such, never written back over the COGs.

**"Do not fabricate gaps" cuts both ways.** SAM 2 is class-agnostic: it will
happily return a mask for a shadow, a car park, a roof section or a tree. Using
raw SAM masks as building instances would fabricate structure. Instances must be
gated on height evidence from the calibrated nDSM before anything is extruded —
which means the *instance* stage runs first but the *building classification* of
those instances happens after calibration, when height exists. This is a
departure from the linear diagram in the plan and it is deliberate.

### 5.3 Revised pipeline

```
ingest
  ↓
segmentation ──────────────► SegmentationArtifact
  ├─ SAM 2 automatic masks       instance_map, boundary_map, confidence, metadata
  └─ colour semantics (kept)     the 5-class raster the anchor gate still needs
  ↓
depth  (unchanged model)
  ↓
depth refinement ──────────► DepthArtifact
  └─ edge-aware, keyed on instance boundaries
  ↓
shadow → anchors → CHHAYA/AGMC (untouched) → uncertainty → assemble
  ↓                                              CalibrationArtifact
instance classification ────► which instances are buildings, from nDSM evidence
  ↓
structural geometry ───────► GeometryArtifact  (render-space)
  └─ cut along instance boundaries, explicit walls, per-instance ground
  ↓
mesh validation ───────────► ValidationArtifact  (separation score, topology)
  ↓
tiles / OBJ / glTF → viewer
```

### 5.4 Ordering

Commit 1 is this document plus the baseline. Then, in the order the value lands:

1. **Phase state machine** — the reported defect, independent of everything else,
   and what makes every later stage observable while it runs.
2. **SAM 2 segmentation stage** — artifact, AMG, tests.
3. **Instance-aware geometry** — the cut mesh and walls. This is where the
   primary objective is won.
4. **Separation and mesh-quality metrics** — needed before claiming (3) worked.
5. **Structure-aware depth refinement** — measured against the baseline, kept
   only if it moves a metric.
6. **Viewer debug layers**, ablation, hardening.

Putting 1 ahead of 2 is a deliberate departure from the plan's commit list: the
progress system is the user-visible bug, it is cheap, and developing a
three-minute segmentation stage behind a progress bar that reports nothing would
make every later step harder.

---

## 6. Known limitations, stated up front

* A single nadir image contains no evidence of a facade. Walls are inferred from
  the footprint boundary and the roof-to-ground height difference; they are
  vertical by construction, not by observation.
* SAM 2 has no building class. Instance → building is decided by calibrated
  height and by the existing semantics, both of which have known error.
* The colour heuristic remains the semantic source. SAM 2 adds instances and
  boundaries, not classes.
* Recovered relief is 36.5 % of truth (§3). Better topology makes the surface
  structurally correct; it does not make short buildings tall. The scale problem
  is separate and is documented in README §5.
* There is no CUDA here. Every number in the benchmark is a four-thread CPU
  number.
