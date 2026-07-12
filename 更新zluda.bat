@echo off
chcp 65001 >nul
setlocal
title Legacy ZLUDA Updater
cd /d "%~dp0"

if exist "%~dp0runtime-rocm\Scripts\python.exe" (
    echo [INFO] Native AMD ROCm is installed. ZLUDA is not used and does not need updating.
    pause
    exit /b 0
)

if not exist "%~dp0env\python.exe" (
    echo [ERROR] Legacy env\python.exe was not found.
    echo [ERROR] Run install_rocm_env.bat to install the recommended native ROCm runtime.
    pause
    exit /b 1
)

echo [WARN] Updating the legacy ZLUDA fallback environment only.
set "upzluda=1"
"%~dp0env\python.exe" -c "import zluda.upzluda"
if errorlevel 1 (
    echo [ERROR] ZLUDA updater failed.
    pause
    exit /b 1
)

if exist "%~dp0env\zluda\upzluda\*.pyd" move /Y "%~dp0env\zluda\upzluda\*.pyd" "%~dp0env\zluda\"
pause
