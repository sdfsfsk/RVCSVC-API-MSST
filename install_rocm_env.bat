@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

where uv >nul 2>&1
if errorlevel 1 (
    echo [ERROR] uv was not found in PATH. Install uv first: https://docs.astral.sh/uv/
    pause
    exit /b 1
)

if not exist "%~dp0runtime-rocm\Scripts\python.exe" (
    echo [1/3] Creating Python 3.12 environment...
    uv venv --python 3.12 "%~dp0runtime-rocm"
    if errorlevel 1 goto :failed
) else (
    echo [1/3] Existing runtime-rocm found; updating it in place.
)

echo [2/3] Installing AMD ROCm 7.2.1 and MSST dependencies...
uv pip install --python "%~dp0runtime-rocm\Scripts\python.exe" -r "%~dp0requirements-rocm.txt"
if errorlevel 1 goto :failed

echo [3/3] Verifying AMD GPU access...
set "MIOPEN_LOG_LEVEL=3"
"%~dp0runtime-rocm\Scripts\python.exe" -c "import torch; assert torch.cuda.is_available(); print(torch.__version__); print(torch.version.hip); print(torch.cuda.get_device_name(0))"
if errorlevel 1 goto :failed

echo [OK] Native AMD ROCm environment is ready.
pause
exit /b 0

:failed
echo [ERROR] Native ROCm environment setup failed.
pause
exit /b 1
