# Bootstrap a clean Python virtual environment on Windows.
# Usage: powershell -ExecutionPolicy Bypass -File scripts/setup_venv.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Venv = Join-Path $Root ".venv"
if (Test-Path $Venv) {
    Remove-Item -Recurse -Force $Venv
}

python -m venv $Venv
$Py = Join-Path $Venv "Scripts\python.exe"
$Pip = Join-Path $Venv "Scripts\pip.exe"

& $Py -m pip install --upgrade pip wheel setuptools

# Install PyTorch CPU wheels first for maximum Windows venv compatibility.
# For CUDA, set $Cuda = "cu124" and uncomment the index-url line below.
& $Pip install torch --index-url https://download.pytorch.org/whl/cpu
& $Pip install -r requirements.txt
& $Pip install -r requirements-dev.txt

# Register conda/base DLL directories when the venv inherits Anaconda.
$base = & $Py -c "import sys; print(sys.base_prefix)"
$libBin = Join-Path $base "Library\bin"
if (Test-Path $libBin) {
    $activate = Join-Path $Venv "Scripts\activate.ps1"
    $dllHook = @"

# Added by setup_venv.ps1 — helps PyTorch locate MSVC/CUDA DLLs on Windows
`$env:PATH = "$libBin;" + `$env:PATH
"@
    Add-Content -Path $activate -Value $dllHook
}

Write-Host "Verifying torch..."
& $Py -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
Write-Host "Running unit tests..."
$env:PYTHONPATH = "$Root;$Root\python"
& $Py -m pytest tests/ -m unit -q --tb=short
Write-Host "Setup complete."
