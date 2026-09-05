# Running TRAKSHA on Google Colab

Everything the delivered pipeline does runs on CPU. Colab is needed for exactly
one optional stage — `refine-mesh`, which textures **the mesh TRAKSHA already
measured** using TRELLIS.2, the frozen half of Sat2City v2. No training, no
fine-tuning, and no vertex moved. It is the stage threefiner failed to be.

Two things worth knowing before you start.

**The GPU is not on the critical path.** If the GPU cell fails, every measured
product — `dsm.tif`, `ndsm.tif`, `structural.obj`, the tileset — is already
written and correct. The stage skips with a reason recorded and the run stands.
Work §3 first and treat §4 as the extra.

**The CUDA source builds are back, and this time they buy something.**
threefiner also needed `nvdiffrast`, and what it returned for the trouble was a
64×64 single-colour texture per building (README §6.2e). TRELLIS.2 needs
`nvdiffrast`, `flash-attn`, `o-voxel` and friends too — but it is the frozen
appearance stack of Sat2City v2, it takes *our* mesh as input, and it asserts
the geometry it returns is the geometry it was given. Budget 15–25 minutes for
the build.

---

## 0. Pick a GPU runtime

`Runtime → Change runtime type`. For §4 you need **L4 or A100** - see the
table there for why a T4 or P100 will not do. For §1-§3, any runtime works,
including no GPU at all. Then:

```python
!nvidia-smi
```

If that prints no table you are on a CPU runtime. Everything except §4 still
works; the refiner will skip with the reason recorded and say so on screen.

---

## 1. Get TRAKSHA onto the machine

Whichever of these matches where your copy lives.

**From GitHub:**

```bash
!git clone https://github.com/<you>/traksha.git /content/traksha
%cd /content/traksha
```

**From Google Drive:**

```python
from google.colab import drive
drive.mount('/content/drive')
!cp -r "/content/drive/MyDrive/UNNAT" /content/traksha
%cd /content/traksha
```

**By upload** (zip the repo locally first, minus `.venv`, `node_modules`, `out`):

```python
from google.colab import files
up = files.upload()                      # choose traksha.zip
!unzip -q traksha.zip -d /content && mv /content/UNNAT /content/traksha
%cd /content/traksha
```

---

## 2. Install

```bash
!pip install -q -e ".[torch,api,mesh,terrain,geometry,refiner]"
```

That pulls, among others:

| package | what it is for |
|---|---|
| `bulldozer-dtm` | bare earth by drape-cloth filter — the 6.195 m → 0.811 m DTM improvement (§6.2a). Has manylinux wheels for CPython 3.10–3.13, so it is a wheel install on every Colab image |
| `scikit-image` | sub-pixel contour tracing and Douglas–Peucker for footprint regularisation (§6.2d) |
| `rasterio` | every raster read and write |

Confirm the optional pieces actually arrived — this prints what is missing
rather than failing later and quietly falling back:

```bash
!python -m traksha.cli doctor
```

```python
from traksha.dsm import dtm
print("bulldozer:", dtm.available(), dtm.version())
```

If `bulldozer: False`, the DTM silently falls back to the morphological filter —
which is the one that sits 6 m up on the rooftops. The provenance records which
branch ran, so check it rather than assuming.

---

## 3. Run the measured pipeline

This is the part that does not need the GPU. `--osm` fetches OpenStreetMap
footprints and roads for the scene's bounding box (§6.2a) and needs network,
which Colab has.

```bash
!python -m traksha.cli run traksha/data/fixture/zurich_rgb.tif \
    --out out/colab \
    --backbone dav2-vitl \
    --chip 1024 \
    --bootstrap 24 \
    --instances sam2-base \
    --osm \
    --dtm bulldozer \
    --dem sim:traksha/data/fixture/zurich_dtm.tif \
    --ref traksha/data/fixture/zurich_dsm.tif \
    --progress plain
```

On your own georeferenced image, swap the DEM for the real product — this is the
fetcher that replaced the simulated one:

```bash
!python -m traksha.cli run /content/my_scene.tif \
    --out out/mine \
    --backbone dav2-vitl --chip 1024 --instances sam2-base \
    --osm --dtm bulldozer \
    --dem copernicus --fetch-dem \
    --progress plain
```

`--dem copernicus --fetch-dem` pulls the GLO-30 tiles the scene touches from the
AWS Open Data mirror. No credentials. Without `--fetch-dem` a product name
resolves only from the local tile cache and a miss is an error, never a silent
skip.

Then build the tileset and the browser mesh — this is where the facade UV fix
runs (§6.2c):

```bash
!python -m traksha.cli mesh out/colab
```

```python
import json
s = json.load(open("out/colab/tiles3d/tileset.json"))["mesh"]["structural"]
print("roof modes :", s["roof_modes"])
print("facade UV  :", s["facade_uv"])
```

`sampling_road_before` → `sampling_road_after` is the measurement: the fraction
of wall vertices whose texel landed on a road pixel, before and after.

---

## 4. The GPU stage that actually refines: TRELLIS.2

This is the one to run. It takes **the mesh TRAKSHA already measured** and gives
it a real PBR texture — no training, no fine-tuning, and no vertex moved.

### Why this and not Sat2City

Sat2City v2 is the strongest published method here, and **nobody can run it**:
its project page says "Code Coming", and the v1 release repository contains
exactly one file, `README.md`, reading "Coming soon".

But its architecture is seven frozen TRELLIS.2 modules plus **one** fine-tuned
geometry flow. The frozen seven are downloadable (`microsoft/TRELLIS.2-4B`, MIT).
The one trained module is the part that invents a shape — which is the part you
do not want, because TRAKSHA has already measured the shape. So the whole
appearance half of Sat2City v2 runs with zero training. See README §6.2g.

> **Which GPU.** Two limits, and they are of different kinds — one has an
> official workaround, the other does not.
>
> *Attention* has an escape hatch. `flash-attn` builds only for compute
> capability 8.0+, but TRELLIS.2's README names the way round it: install
> `xformers` and set `ATTN_BACKEND=xformers`. It is written for V100 and up.
>
> *Memory* has none. "An NVIDIA GPU with at least 24GB of memory is necessary",
> one checkpoint (TRELLIS.2-4B, 4 B parameters), no smaller variant, and no
> documented inference sharding.
>
> | GPU | cap | VRAM | verdict |
> |---|---|---|---|
> | **A100** (Colab Pro+) | 8.0 | 40 GB | ✅ verified upstream |
> | **H100** | 9.0 | 80 GB | ✅ verified upstream |
> | **L4** (Colab Pro) | 8.9 | 24 GB | ✅ smallest card that meets the stated requirement |
> | **T4 ×2** (Kaggle) | 7.5 | 15 GB **each** | ⚠️ attention solvable; **9 GB short**, and the two do not pool |
> | **T4** (Colab free) | 7.5 | 15 GB | ⚠️ same, without the second card |
> | **V100** | 7.0 | 16 GB | ⚠️ attention solvable; 8 GB short |
> | **P100** (Kaggle default) | 6.0 | 16 GB | ❌ no documented backend at all |
>
> **"T4 ×2" is not a 30 GB card.** It is two 15 GB devices. Without model
> parallelism — which TRELLIS.2 does not document — this stage sees 15 GB. The
> second card buys *throughput*, not capacity: `refine` is independent per
> building by construction, so two cards can texture two buildings at once, and
> neither can hold half a model.
>
> **Time is not the constraint anywhere.** At 25–90 s per building, Kaggle's 30
> h/week is hundreds of scenes. What decides this is the 24 GB line.
>
> On a card below it, the levers are `--resolution 512` (which selects
> TRELLIS.2's lighter flow model, not just a smaller output), `low_vram` (on by
> default), and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Whether that
> is enough to clear 15 GB is **untested** — upstream verified only A100 and
> H100, and there is no GPU here to find out on.
>
> `preflight` reports all of this before anything is built:
>
> ```
> MISSING    a newer GPU - this is Pascal (P100) at 6.0. flash-attn needs 8.0+
>            and bfloat16 needs 8.0+, and unlike Volta and Turing this
>            generation has no documented route round either ...
> note       15.8 GB of VRAM against the 24 GB TRELLIS.2 states as necessary -
>            short by 8 GB ... a second card of the same size does not help
> note       2 GPUs visible, but this stage uses one. They do not pool ...
> ```
>
> **None of this affects §1–§3.** The measured pipeline is CPU-only: a P100
> runtime reconstructs the scene exactly as an A100 one does. A weak GPU costs
> you the appearance refinement; the run stands without it.

### Install

TRELLIS.2 builds several CUDA extensions from source. This is slow — budget
15–25 minutes — and it is the real cost of this stage.

```bash
!git clone --recurse-submodules https://github.com/microsoft/TRELLIS.2.git /content/TRELLIS.2
%cd /content/TRELLIS.2
!. ./setup.sh --new-env --basic --flash-attn --nvdiffrast --o-voxel
%cd /content/traksha
```

If `setup.sh` fights Colab's preinstalled torch, install the pieces directly and
skip the env creation:

```bash
%cd /content/TRELLIS.2
!pip install -q -e .
!pip install -q flash-attn --no-build-isolation
# On a card below compute capability 8.0 (V100, T4) flash-attn will not build.
# Use TRELLIS.2's documented fallback instead:
#   !pip install -q xformers
#   %env ATTN_BACKEND=xformers
!pip install -q ./o-voxel
%cd /content/traksha
```

Check what landed before spending GPU time:

```bash
!TRELLIS_ROOT=/content/TRELLIS.2 python -m traksha.cli refine-mesh out/colab --dry-run
```

That prints the missing pieces by name, then how many buildings it selected and
what they will cost. It touches no GPU.

### Run it

```bash
!TRELLIS_ROOT=/content/TRELLIS.2 python -m traksha.cli refine-mesh out/colab \
    --root /content/TRELLIS.2 \
    --limit 8 \
    --resolution 1024 \
    --texture-res 2048
```

On a 15 GB card, drop to the lighter flow model:

```bash
!python -m traksha.cli refine-mesh out/colab --root /content/TRELLIS.2 \
    --limit 4 --resolution 512 --texture-res 1024
```

The first run also pays for the ~4 B-parameter checkpoint coming down.

### Reading the result

Per building it prints the time, the **vertex drift**, and the file:

```
  refined    8/8
    building_1     41.2s  drift 0.0000 m  -> building_1.glb
    building_4     38.7s  drift 0.0000 m  -> building_4.glb
```

**Drift is the number that matters.** It is the largest distance between a
vertex we sent and the one that came back, after inverting TRELLIS.2's own
normalisation. It should be ~0: `preprocess_mesh` is a pure similarity transform
built with `process=False`, so it merges nothing. Anything over 10 mm and the
building is **refused** rather than kept — only colour is allowed to cross back.

That guard is exactly what threefiner could not offer: it ran `clean_mesh`,
which merges every pair of vertices within 1 % of the bounding-box diagonal, so
there was no index-wise comparison to make (README §6.2e).

```python
import json
r = json.load(open("out/colab/refined/trellis.json"))
print("trained:", r["trained"], "  geometry measured:", r["geometry_measured"])
print("refined:", r["refined"], "of", r["attempted"])
for b in r["buildings"][:5]:
    print(" ", b.get("id"), b.get("max_drift_m"), b.get("file") or b.get("skipped"))
```

### As a pipeline phase

The web service runs the same stage. Point it at the checkout once:

```bash
!TRELLIS_ROOT=/content/TRELLIS.2 python -m traksha.cli serve --host 0.0.0.0 --port 8000
```

Without a GPU or a checkout the phase is skipped up front with the reason
recorded, so the bar still reaches 100% and the reason is on screen.

---

## 5. Look at it

```python
from google.colab import files
!zip -qr /content/colab_run.zip out/colab
files.download("/content/colab_run.zip")
```

Or serve the viewer over a tunnel:

```bash
!npm --prefix web ci && npm --prefix web run build
```

```python
from google.colab.output import eval_js
print(eval_js("google.colab.kernel.proxyPort(8000)"))
```

```bash
!python -m traksha.cli viewer out/colab --port 8000
```

The viewer's side panel has three panels fed by this work: **Facades** (how many
walls were given a flat colour, and the road-sampling fraction before and
after), **Roofs** (platform / plane / measured counts), and **Generated scene**
(which model ran, and how far its surface sits from the calibrated one).

---

## Troubleshooting

| symptom | cause |
|---|---|
| `bulldozer: False` | the `terrain` extra did not install. The DTM falls back to the morphological filter, which sits ~6 m high on rooftops. Check `provenance.json` → `dtm.method` |
| `no cached OSM extract … network access was not requested` | you passed `--osm-cache-only`, or the scene has no CRS. OSM needs a georeferenced image |
| `tile … is not cached and network access was not requested` | add `--fetch-dem` |
| Sat2City will not run | it has published no code. See README §6.2g — its appearance half is what §4 runs |
| `a CUDA device` missing | CPU runtime. Switch to a GPU runtime, or accept the skip — the measured products are unaffected |
| `o_voxel` missing | TRELLIS.2's `pip install ./o-voxel` step did not run or did not build. It is the flexible dual grid the geometry encoder needs; there is no fallback |
| refine-mesh OOMs | 24 GB is what TRELLIS.2 asks for. Use `--resolution 512 --texture-res 1024 --limit 4`, or move to an A100 |
| a building is refused with `the returned surface sits up to … m from ours` | the pipeline remeshed instead of preserving the input. The building is dropped rather than kept — that guard is the point |
| the run is slow | `--backbone dav2-vits` and `--chip 512` cut depth time several-fold at some accuracy cost. Depth is ~80% of a run |
