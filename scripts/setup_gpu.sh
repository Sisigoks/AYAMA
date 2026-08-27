#!/usr/bin/env bash
# AYAMA GPU environment setup (Linux / WSL).
#
#   bash scripts/setup_gpu.sh              # detect CUDA, build .venv, verify
#   CUDA=cu121 bash scripts/setup_gpu.sh   # force a wheel index
#   PY=python3.11 bash scripts/setup_gpu.sh
#
# rasterio ships GDAL inside its manylinux wheel, so no system GDAL is needed.
set -euo pipefail

PY=${PY:-python3}
VENV=${VENV:-.venv}
cd "$(dirname "$0")/.."

# ---- pick a torch wheel index -------------------------------------------------
detect_cuda() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then echo "cpu"; return; fi
  local ver
  ver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1)
  [ -z "$ver" ] && { echo "cpu"; return; }
  # Driver -> newest CUDA runtime that driver supports, mapped to a wheel tag.
  if   [ "$ver" -ge 550 ]; then echo "cu124"
  elif [ "$ver" -ge 525 ]; then echo "cu121"
  elif [ "$ver" -ge 510 ]; then echo "cu118"
  else echo "cpu"; fi
}
CUDA=${CUDA:-$(detect_cuda)}

echo "== AYAMA setup =="
echo "   python : $($PY --version 2>&1)"
echo "   cuda   : $CUDA"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed 's/^/   gpu    : /'
else
  echo "   gpu    : none detected (installing CPU wheels)"
fi

# ---- venv ---------------------------------------------------------------------
[ -d "$VENV" ] || $PY -m venv "$VENV"
PIP="$VENV/bin/pip"
PYBIN="$VENV/bin/python"
$PIP install --quiet --upgrade pip wheel

echo "== core deps =="
$PIP install --quiet -r requirements.txt

echo "== torch ($CUDA) =="
if [ "$CUDA" = "cpu" ]; then
  $PIP install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cpu
else
  $PIP install --quiet torch torchvision --index-url "https://download.pytorch.org/whl/$CUDA"
fi
$PIP install --quiet transformers

echo "== verify =="
$PYBIN -m ayama.cli doctor

cat <<'EOF'

Next:
  .venv/bin/python -m ayama.cli synth --out data/sample.tif --size 2048
  bash scripts/harness.sh                      # full harness, writes to out/harness
  .venv/bin/python -m pytest tests -q -m gpu -v # GPU-only checks
EOF
