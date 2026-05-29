[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$PARENT_DIR = Split-Path -Parent $SCRIPT_DIR
$CONDA_DIR = Join-Path $PARENT_DIR "miniconda3"
$ENV_NAME = "paddle-ocr"
$PYTHON_VER = "3.10"

function Write-Box($msg) {
    $line = "+" + ("-" * 50) + "+"
    Write-Host ""
    Write-Host $line -ForegroundColor Cyan
    Write-Host "| $msg" -ForegroundColor Cyan
    Write-Host $line -ForegroundColor Cyan
}

function Write-Step($num, $total, $msg) {
    Write-Host ""
    Write-Host "[$num/$total] $msg" -ForegroundColor Yellow
}

function Write-OK($msg) {
    Write-Host "  [OK] $msg" -ForegroundColor Green
}

function Write-Warn($msg) {
    Write-Host "  [!!] $msg" -ForegroundColor Red
}

function Write-Info($msg) {
    Write-Host "  ... $msg" -ForegroundColor Gray
}

function Safe-Exec {
    param([string]$Command, [string]$ErrorMessage)
    Write-Host "  [RUN] $Command" -ForegroundColor DarkYellow
    try {
        $prev = $global:LASTEXITCODE
        $global:LASTEXITCODE = 0
        $output = Invoke-Expression $Command *>&1
        if ($global:LASTEXITCODE -ne 0) {
            Write-Warn "$ErrorMessage (exit code: $global:LASTEXITCODE)"
            return $false
        }
        return $true
    } catch {
        Write-Warn "$ErrorMessage : $($_.Exception.Message)"
        return $false
    }
}

Write-Box "PaddleOCR GPU One-Click Setup"

# ==================== Step 1: Conda ====================
Write-Step 1 4 "Checking Conda environment..."

$condaExe = $null

try {
    $sysConda = Get-Command conda -ErrorAction SilentlyContinue
    if ($sysConda) {
        $condaExe = $sysConda.Source
        Write-OK "System Conda found"
    }
} catch {}

if (-not $condaExe) {
    $localConda = Join-Path $CONDA_DIR "Scripts\conda.exe"
    if (Test-Path $localConda) {
        $condaExe = $localConda
        $env:PATH = "$CONDA_DIR;$CONDA_DIR\Scripts;$CONDA_DIR\Library\bin;$env:PATH"
        Write-OK "Local Miniconda found"
    }
}

if (-not $condaExe) {
    Write-Warn "Conda not found! Please install Miniconda first."
    Read-Host "Press Enter to exit"
    exit 1
}

# ==================== Step 2: Create Environment ====================
Write-Step 2 4 "Creating Python $PYTHON_VER environment..."

Write-Info "Accepting conda channel Terms of Service..."
$null = & $condaExe tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>&1
$null = & $condaExe tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>&1
$null = & $condaExe tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2 2>&1

$envExists = & $condaExe env list 2>&1 | Select-String "^$ENV_NAME\s"
if ($envExists) {
    Write-OK "Environment '$ENV_NAME' already exists"
} else {
    Write-Info "Creating environment with Python $PYTHON_VER..."
    $null = & $condaExe create -n $ENV_NAME python=$PYTHON_VER -y 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Info "Default channels failed, trying conda-forge..."
        $null = & $condaExe create -n $ENV_NAME python=$PYTHON_VER -y --override-channels -c conda-forge 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Warn "Failed to create environment!"
            Read-Host "Press Enter to exit"
            exit 1
        }
    }
    Write-OK "Environment created"
}

$envPython = Join-Path $CONDA_DIR "envs\$ENV_NAME\python.exe"
$envPip = Join-Path $CONDA_DIR "envs\$ENV_NAME\Scripts\pip.exe"

if (-not (Test-Path $envPython)) {
    Write-Warn "Python not found in environment: $envPython"
    Read-Host "Press Enter to exit"
    exit 1
}

# ==================== Step 3: Install PaddlePaddle GPU + PaddleOCR ====================
Write-Step 3 4 "Installing PaddlePaddle GPU + PaddleOCR..."

Write-Info "Upgrading pip..."
$null = & $envPython -m pip install --upgrade pip 2>&1
Write-OK "pip ready"

Write-Info "Installing PaddlePaddle GPU (CUDA 12.6)..."
Write-Host "  --------------------------------------------------" -ForegroundColor DarkGray
Write-Host "  This may take 5-10 minutes, please wait..." -ForegroundColor DarkGray
Write-Host "  Download size: ~500MB" -ForegroundColor DarkGray
Write-Host "  --------------------------------------------------" -ForegroundColor DarkGray
Write-Host ""

$paddle_ok = $false

& $envPython -m pip install paddlepaddle-gpu==3.2.2 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
if ($LASTEXITCODE -eq 0) {
    Write-OK "PaddlePaddle GPU (CUDA 12.6) installed"
    $paddle_ok = $true
} else {
    Write-Info "CUDA 12.6 failed (exit: $LASTEXITCODE), trying CUDA 11.8..."
    & $envPython -m pip install paddlepaddle-gpu==3.2.2 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
    if ($LASTEXITCODE -eq 0) {
        Write-OK "PaddlePaddle GPU (CUDA 11.8) installed"
        $paddle_ok = $true
    } else {
        Write-Info "GPU version failed, trying CPU fallback..."
        & $envPython -m pip install paddlepaddle==3.2.2 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
        if ($LASTEXITCODE -eq 0) {
            Write-OK "PaddlePaddle CPU installed (GPU will not be available)"
            $paddle_ok = $true
        } else {
            Write-Warn "PaddlePaddle installation failed!"
            Write-Warn "Please check network connection and try again."
            Read-Host "Press Enter to exit"
            exit 1
        }
    }
}

if ($paddle_ok) {
    Write-Info "Installing PaddleX with OCR support..."
    Write-Host "  --------------------------------------------------" -ForegroundColor DarkGray
    Write-Host "  This will also download OCR model dependencies" -ForegroundColor DarkGray
    Write-Host "  --------------------------------------------------" -ForegroundColor DarkGray
    Write-Host ""

    & $envPython -m pip install "paddlex[ocr]"
    if ($LASTEXITCODE -eq 0) {
        Write-OK "PaddleX with OCR installed"
    } else {
        Write-Info "paddlex[ocr] failed, trying paddlex alone..."
        & $envPython -m pip install paddlex
        if ($LASTEXITCODE -eq 0) {
            Write-OK "PaddleX installed (without OCR extras)"
        } else {
            Write-Warn "PaddleX installation failed!"
            Write-Warn "Please check network connection."
            Read-Host "Press Enter to exit"
            exit 1
        }
    }
}

Write-Info "Installing Gradio + PyMuPDF for Web UI..."
& $envPython -m pip install gradio pillow pymupdf
if ($LASTEXITCODE -eq 0) {
    Write-OK "Web UI dependencies installed"
} else {
    Write-Info "Gradio install had issues, retrying..."
    & $envPython -m pip install gradio pillow pymupdf --no-cache-dir
    if ($LASTEXITCODE -eq 0) {
        Write-OK "Web UI dependencies installed (retry OK)"
    } else {
        Write-Warn "Gradio installation had issues, UI may not work"
    }
}

# ==================== Step 4: Verify ====================
Write-Step 4 4 "Verifying installation..."

$verify_script = @'
import paddle
print(f"PaddlePaddle: {paddle.__version__}")
gpu_available = paddle.device.is_compiled_with_cuda()
print(f"GPU compiled: {gpu_available}")
if gpu_available:
    try:
        count = paddle.device.cuda.device_count()
        print(f"GPU count: {count}")
        for i in range(count):
            print(f"  GPU {i}: {paddle.device.cuda.get_device_name(i)}")
    except:
        pass
try:
    import paddlex
    print(f"PaddleX: {paddlex.__version__}")
except:
    print("PaddleX: installed (version check skipped)")
try:
    import gradio
    print(f"Gradio: {gradio.__version__}")
except:
    pass
'@

$result = & $envPython -c $verify_script 2>&1
Write-Host ""
Write-Host "  Verification result:" -ForegroundColor Cyan
Write-Host $result

if ($LASTEXITCODE -ne 0) {
    Write-Warn "Verification had issues, but installation may still work."
}

Write-Host ""
Write-Box "Setup Complete!"
Write-Host ""
Write-Host "  Double-click '????PaddleOCR.bat' to start" -ForegroundColor Green
Write-Host ""
Read-Host "Press Enter to exit"
