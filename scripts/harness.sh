#!/usr/bin/env bash
# One-shot harness. Everything you need to know about a machine and a build,
# written to a results directory you can commit or paste into a slide.
#
#   bash scripts/harness.sh                       # defaults: dav2-vits, 2048 px
#   BACKBONES=dav2-vits,dav2-vitl bash scripts/harness.sh
#   SIZE=4096 CHIPS=1024 BATCHES=1,2,4,8 bash scripts/harness.sh
#   IMAGE=data/real.tif REF=data/lidar_dsm.tif DEM=data/copernicus.tif bash scripts/harness.sh
#
# With no IMAGE it writes out the bundled real sample scene - a lidar crop of
# central Zurich with a swissSURFACE3D DSM and a swissALTI3D DTM - so the
# harness produces real metrics on any machine with nothing to download.
set -uo pipefail

cd "$(dirname "$0")/.."

# ---- pick an interpreter that actually works -----------------------------------
# "Is the file executable" is not the question. A venv whose creation failed at
# ensurepip - the normal outcome on Debian images without python3-venv, Colab
# included - leaves behind a perfectly executable python that can import nothing.
# The old check passed it happily and every one of the eight stages below then
# failed with the same ModuleNotFoundError, which tells the reader nothing about
# the actual problem.
#
# So each candidate is tried by importing what the harness needs, and if none
# work we say so once, at the top, instead of eight times.
usable() {
  [ -x "$1" ] || command -v "$1" >/dev/null 2>&1 || return 1
  "$1" -c "import numpy, scipy, ayama" >/dev/null 2>&1
}

PYBIN=""
for cand in ${PYBIN_OVERRIDE:-} .venv/bin/python .venv/Scripts/python.exe python3 python; do
  [ -z "$cand" ] && continue
  if usable "$cand"; then PYBIN="$cand"; break; fi
done

if [ -z "$PYBIN" ]; then
  echo "harness: no interpreter here can 'import numpy, scipy, ayama'." >&2
  echo "" >&2
  for cand in .venv/bin/python .venv/Scripts/python.exe python3 python; do
    if [ -x "$cand" ] || command -v "$cand" >/dev/null 2>&1; then
      why=$("$cand" -c "import numpy, scipy, ayama" 2>&1 | tail -1)
      echo "  $cand -> $why" >&2
    fi
  done
  echo "" >&2
  echo "  Fix one of:" >&2
  echo "    python -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
  echo "    pip install -r requirements.txt    # into whatever python you are using" >&2
  echo "    PYBIN_OVERRIDE=/path/to/python bash scripts/harness.sh" >&2
  echo "" >&2
  echo "  A .venv that exists but imports nothing usually means 'python -m venv'" >&2
  echo "  failed at ensurepip: rm -rf .venv, then apt install python3-venv." >&2
  exit 1
fi

OUT=${OUT:-out/harness}
SIZE=${SIZE:-2048}
BACKBONES=${BACKBONES:-dav2-vits}
CHIPS=${CHIPS:-512,1024}
BATCHES=${BATCHES:-1,2,4}
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

echo "harness: using $PYBIN ($("$PYBIN" --version 2>&1))"

say "1/8  doctor"
run "$PYBIN" -m ayama.cli doctor --load "$BACKBONES"

if [ -z "$IMAGE" ]; then
  say "2/8  bundled sample scene (${SIZE}x${SIZE}, real lidar truth)"
  run "$PYBIN" -m ayama.cli sample --out "$OUT/scene.tif" --size "$SIZE"
  IMAGE="$OUT/scene.tif"
  REF="$OUT/scene_dsm.tif"
  DEM="sim:$OUT/scene_dtm.tif"
else
  say "2/8  using supplied image"
  run "$PYBIN" -m ayama.cli info "$IMAGE"
fi

say "3/8  unit tests"
run "$PYBIN" -m pytest tests -q

say "4/8  throughput sweep"
run "$PYBIN" -m ayama.cli bench --image "$IMAGE" --backbones "$BACKBONES" \
    --chips "$CHIPS" --batches "$BATCHES" \
    --json "$OUT/bench.json"

say "5/8  full pipeline run"
CMD=("$PYBIN" -m ayama.cli run "$IMAGE" --out "$OUT/run" --backbone "$PRIMARY"
     --batch 0 --bootstrap "$BOOTSTRAP" --json "$OUT/run_summary.json")
[ -n "$DEM" ] && CMD+=(--dem "$DEM")
[ -n "$REF" ] && CMD+=(--ref "$REF")
run "${CMD[@]}"

if [ -n "$REF" ]; then
  say "6/8  ablation table"
  ABL=("$PYBIN" -m ayama.cli ablate "$IMAGE" --ref "$REF" --backbone "$PRIMARY"
       --batch 0 --bootstrap "$BOOTSTRAP" --json "$OUT/ablation.json")
  [ -n "$DEM" ] && ABL+=(--dem "$DEM")
  run "${ABL[@]}"
else
  say "6/8  ablation skipped (no REF supplied)"
fi

say "7/8  Phase 3 tileset + mesh"
run "$PYBIN" -m ayama.cli mesh "$OUT/run" --out "$OUT/tiles3d" --progress plain
if command -v node >/dev/null 2>&1; then
  run node scripts/check_app.js "$OUT/tiles3d"
else
  say "     viewer render check skipped (node not on PATH)"
fi

say "8/8  Phase 3/4 delivery benchmark"
run "$PYBIN" -m ayama.cli delivery "$OUT/run" --out "$OUT/delivery"     --obj-strides 2,4

say "done"
echo "results in $OUT:" | tee -a "$LOG"
ls -la "$OUT" | tee -a "$LOG"
