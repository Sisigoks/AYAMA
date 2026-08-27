# AYAMA setup on Windows.
#
#   powershell -ExecutionPolicy Bypass -File scripts/setup.ps1
#   powershell -ExecutionPolicy Bypass -File scripts/setup.ps1 -Cuda cu124
#
# Picks CPU wheels unless an NVIDIA GPU is detected or -Cuda is given.
param(
    [string]$Cuda = "",
    [string]$Venv = ".venv"
)

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

if (-not $Cuda) {
    $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($smi) {
        $driver = (& nvidia-smi --query-gpu=driver_version --format=csv,noheader | Select-Object -First 1)
        $major = [int]($driver -split '\.')[0]
        if     ($major -ge 550) { $Cuda = "cu124" }
        elseif ($major -ge 525) { $Cuda = "cu121" }
        elseif ($major -ge 510) { $Cuda = "cu118" }
        else                    { $Cuda = "cpu" }
        Write-Host "   gpu    : $(& nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | Select-Object -First 1)"
    } else {
        $Cuda = "cpu"
        Write-Host "   gpu    : none detected (installing CPU wheels)"
    }
}

Write-Host "== AYAMA setup =="
Write-Host "   cuda   : $Cuda"

if (-not (Test-Path $Venv)) { python -m venv $Venv }
$py  = Join-Path $Venv "Scripts\python.exe"

& $py -m pip install --quiet --upgrade pip wheel
Write-Host "== core deps =="
& $py -m pip install --quiet -r requirements.txt

Write-Host "== torch ($Cuda) =="
& $py -m pip install --quiet torch torchvision --index-url "https://download.pytorch.org/whl/$Cuda"
& $py -m pip install --quiet transformers

Write-Host "== verify =="
& $py -m ayama.cli doctor

Write-Host ""
Write-Host "Next:"
Write-Host "  $py -m ayama.cli synth --out data/sample.tif --size 2048"
Write-Host "  bash scripts/harness.sh"
Write-Host "  $py -m pytest tests -q"
