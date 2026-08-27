#!/usr/bin/env bash
# One-shot harness. Everything you need to know about a machine and a build,
# written to a results directory you can commit or paste into a slide.
#
#   bash scripts/harness.sh                       # defaults: dav2-vits, 2048 px
#   BACKBONES=dav2-vits,dav2-vitl bash scripts/harness.sh
#   SIZE=4096 CHIPS=1024 BATCHES=1,2,4,8 bash scripts/harness.sh
#   IMAGE=data/real.tif REF=data/lidar_dsm.tif DEM=data/copernicus.tif bash scripts/harness.sh
#
# With no IMAGE it generates a synthetic town with known ground truth, so the
# harness produces real metrics on any machine with no data to download.
set -uo pipefail

cd "$(dirname "$0")/.."
PYBIN=${PYBIN:-.venv/bin/python}
[ -x "$PYBIN" ] || PYBIN=${PYBIN_FALLBACK:-python}
[ -x ".venv/Scripts/python.exe" ] && PYBIN=".venv/Scripts/python.exe"   # Windows venv

OUT=${OUT:-out/harness}
SIZE=${SIZE:-2048}
BACKBONES=${BACKBONES:-dav2-vits}
CHIPS=${CHIPS:-512,1024}
BATCHES=${BATCHES:-1,2,4}
DEVICE=${DEVICE:-auto}
DTYPE=${DTYPE:-auto}
BOOTSTRAP=${BOOTSTRAP:-24}
PRIMARY=${PRIMARY:-${BACKBONES%%,*}}

IMAGE=${IMAGE:-}
REF=${REF:-}
DEM=${DEM:-}

mkdir -p "$OUT"
LOG="$OUT/harness.log"
: > "$LOG"

say() { echo "" | tee -a "$LOG"; echo "=== $* ===" | tee -a "$LOG"; }
run() { echo "\$ $*" | tee -a "$LOG"; "$@" 2>&1 | tee -a "$LOG"; return "${PIPESTATUS[0]}"; }

say "1/6  doctor"
run "$PYBIN" -m unnat.cli doctor --load "$BACKBONES" --device "$DEVICE"

if [ -z "$IMAGE" ]; then
  say "2/6  synthetic scene (${SIZE}x${SIZE}, known ground truth)"
  run "$PYBIN" -m unnat.cli synth --out "$OUT/scene.tif" --size "$SIZE"
  IMAGE="$OUT/scene.tif"
  REF="$OUT/scene_dsm.tif"
  DEM="sim:$OUT/scene_dtm.tif"
else
  say "2/6  using supplied image"
  run "$PYBIN" -m unnat.cli info "$IMAGE"
fi

say "3/7  unit tests"
run "$PYBIN" -m pytest tests -q

say "4/7  throughput sweep"
run "$PYBIN" -m unnat.cli bench --image "$IMAGE" --backbones "$BACKBONES" \
    --chips "$CHIPS" --batches "$BATCHES" --device "$DEVICE" --dtype "$DTYPE" \
    --json "$OUT/bench.json"

say "5/7  full pipeline run"
CMD=("$PYBIN" -m unnat.cli run "$IMAGE" --out "$OUT/run" --backbone "$PRIMARY"
     --device "$DEVICE" --batch 0 --bootstrap "$BOOTSTRAP" --json "$OUT/run_summary.json")
[ -n "$DEM" ] && CMD+=(--dem "$DEM")
[ -n "$REF" ] && CMD+=(--ref "$REF")
run "${CMD[@]}"

if [ -n "$REF" ]; then
  say "6/7  ablation table"
  ABL=("$PYBIN" -m unnat.cli ablate "$IMAGE" --ref "$REF" --backbone "$PRIMARY"
       --device "$DEVICE" --batch 0 --bootstrap "$BOOTSTRAP" --json "$OUT/ablation.json")
  [ -n "$DEM" ] && ABL+=(--dem "$DEM")
  run "${ABL[@]}"
else
  say "6/7  ablation skipped (no REF supplied)"
fi

say "7/7  Phase 3 tileset + mesh"
run "$PYBIN" -m unnat.cli mesh "$OUT/run" --out "$OUT/tiles3d" --progress plain
if command -v node >/dev/null 2>&1; then
  run node scripts/check_app.js "$OUT/tiles3d"
else
  say "     viewer render check skipped (node not on PATH)"
fi

say "done"
echo "results in $OUT:" | tee -a "$LOG"
ls -la "$OUT" | tee -a "$LOG"
