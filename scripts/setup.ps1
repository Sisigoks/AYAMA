# AYAMA setup on Windows.
#
#   powershell -ExecutionPolicy Bypass -File scripts/setup.ps1
#
# Installs CPU wheels. AYAMA is CPU-only by design - see the note at the top of
# ayama/depth/backbones/hf.py for the measurement behind that decision.
param(
    [string]$Venv = ".venv"
)

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

Write-Host "== AYAMA setup =="
if (-not (Test-Path $Venv)) { python -m venv $Venv }
$py  = Join-Path $Venv "Scripts\python.exe"

& $py -m pip install --quiet --upgrade pip wheel
Write-Host "== core deps =="
& $py -m pip install --quiet -r requirements.txt

Write-Host "== torch (cpu) =="
& $py -m pip install --quiet torch torchvision --index-url "https://download.pytorch.org/whl/cpu"
& $py -m pip install --quiet transformers

Write-Host "== verify =="
& $py -m ayama.cli doctor

Write-Host ""
Write-Host "Next:"
Write-Host "  $py -m ayama.cli sample --out data/sample.tif --size 576"
Write-Host "  bash scripts/harness.sh"
Write-Host "  $py -m pytest tests -q"
