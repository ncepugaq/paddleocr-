[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$PARENT_DIR = Split-Path -Parent $SCRIPT_DIR
$CONDA_DIR = Join-Path $PARENT_DIR "miniconda3"
$ENV_NAME = "paddle-ocr"

function Write-Box($msg) {
    $line = "+" + ("-" * 50) + "+"
    Write-Host ""
    Write-Host $line -ForegroundColor Cyan
    Write-Host "| $msg" -ForegroundColor Cyan
    Write-Host $line -ForegroundColor Cyan
}

Write-Box "PaddleOCR GPU Launcher"

$condaExe = $null
try {
    $sysConda = Get-Command conda -ErrorAction SilentlyContinue
    if ($sysConda) { $condaExe = $sysConda.Source }
} catch {}

if (-not $condaExe) {
    $localConda = Join-Path $CONDA_DIR "Scripts\conda.exe"
    if (Test-Path $localConda) {
        $condaExe = $localConda
        $env:PATH = "$CONDA_DIR;$CONDA_DIR\Scripts;$CONDA_DIR\Library\bin;$env:PATH"
    }
}

if (-not $condaExe) {
    Write-Host ""
    Write-Host "  [ERROR] Conda not found!" -ForegroundColor Red
    Write-Host "  Please run the setup script first (install_paddleocr.bat)" -ForegroundColor Red
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

$envExists = & $condaExe env list 2>&1 | Select-String "^$ENV_NAME\s"
if (-not $envExists) {
    Write-Host ""
    Write-Host "  [ERROR] Environment '$ENV_NAME' not found!" -ForegroundColor Red
    Write-Host "  Please run the setup script first (install_paddleocr.bat)" -ForegroundColor Red
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

$envPython = Join-Path $CONDA_DIR "envs\$ENV_NAME\python.exe"

$PDX_CACHE = Join-Path $SCRIPT_DIR ".paddlex"
$env:PADDLE_PDX_CACHE_HOME = $PDX_CACHE
$env:HF_HOME = Join-Path $PDX_CACHE "huggingface"
$env:HF_HUB_CACHE = Join-Path $PDX_CACHE "huggingface\hub"

Write-Host ""
Write-Host "  Starting PaddleOCR GPU Web UI..." -ForegroundColor Green
Write-Host ""
Write-Host "  +----------------------------------------------+" -ForegroundColor Cyan
Write-Host "  | Browser will open automatically              |" -ForegroundColor Cyan
Write-Host "  | URL: http://localhost:7860                   |" -ForegroundColor Cyan
Write-Host "  |                                              |" -ForegroundColor Cyan
Write-Host "  | First run will download OCR models (~100MB)  |" -ForegroundColor Cyan
Write-Host "  | Please wait patiently...                     |" -ForegroundColor Cyan
Write-Host "  +----------------------------------------------+" -ForegroundColor Cyan
Write-Host ""

Push-Location $SCRIPT_DIR
& $envPython paddleocr_webui.py --port 7860
Pop-Location

Write-Host ""
Write-Host "  PaddleOCR closed." -ForegroundColor Yellow
Read-Host "Press Enter to exit"
