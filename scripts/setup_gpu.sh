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

# ---- interpreter ---------------------------------------------------------------
# A virtualenv is the right default on a machine you own and the wrong one on a
# managed notebook. Colab ships torch built against its own driver, preinstalled
# globally; a venv there hides it and you end up downloading a second torch that
# may not match the driver. Worse, Debian-based images often ship python3
# without ensurepip, so `python3 -m venv` half-succeeds: it creates the
# directory, fails on pip, and leaves an interpreter that imports nothing. That
# is what produced eight identical ModuleNotFoundError tracebacks.
#
# So: detect a managed environment and use it as-is; otherwise build a venv, and
# if that fails, say why and fall back rather than leaving a broken one behind.
is_managed() {
  [ -n "${AYAMA_NO_VENV:-}" ] && return 0
  [ -d /content ] && $PY -c "import google.colab" >/dev/null 2>&1 && return 0
  [ -n "${KAGGLE_KERNEL_RUN_TYPE:-}" ] && return 0
  return 1
}

USE_VENV=1
if is_managed; then
  USE_VENV=0
  echo "   env    : managed notebook detected - installing into the ambient python"
  echo "            (set AYAMA_NO_VENV= to force a virtualenv)"
fi

if [ "$USE_VENV" = "1" ]; then
  if [ ! -x "$VENV/bin/python" ] && [ ! -x "$VENV/Scripts/python.exe" ]; then
    if ! $PY -m venv "$VENV" 2>/tmp/ayama_venv.err; then
      echo "   env    : could not create a virtualenv:"
      sed 's/^/            /' /tmp/ayama_venv.err | tail -3
      echo "            falling back to the ambient python."
      echo "            (on Debian/Ubuntu: apt install python3-venv)"
      rm -rf "$VENV"                       # never leave a half-built venv
      USE_VENV=0
    fi
  fi
fi

if [ "$USE_VENV" = "1" ]; then
  PYBIN="$VENV/bin/python"
  [ -x "$VENV/Scripts/python.exe" ] && PYBIN="$VENV/Scripts/python.exe"
  PIP="$PYBIN -m pip"
else
  PYBIN="$PY"
  PIP="$PY -m pip"
fi
echo "   using  : $PYBIN"

$PIP install --quiet --upgrade pip wheel

echo "== core deps =="
$PIP install --quiet -r requirements.txt

echo "== torch ($CUDA) =="
# On a managed notebook torch is already present and already matched to the
# driver. Replacing it with a wheel picked from `nvidia-smi` is how a working
# CUDA setup gets broken, so it is left alone unless it is missing.
if [ "$USE_VENV" = "0" ] && $PYBIN -c "import torch" >/dev/null 2>&1; then
  $PYBIN - <<'PYEOF'
import torch
print(f"   keeping preinstalled torch {torch.__version__}"
      f"  (cuda available: {torch.cuda.is_available()})")
PYEOF
elif [ "$CUDA" = "cpu" ]; then
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
