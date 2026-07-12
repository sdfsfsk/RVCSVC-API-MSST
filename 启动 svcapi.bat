@echo off
chcp 65001 >nul
setlocal
title SVC API - MSST (Native AMD ROCm)
cd /d "%~dp0"

set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"
set "NATIVE_PYTHON=%~dp0runtime-rocm\Scripts\python.exe"
set "LEGACY_PYTHON=%~dp0env\python.exe"
set "GATEWAY_PORT=9999"

echo [INFO] Checking for an old SVC gateway listener on port %GATEWAY_PORT%...
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -Command "$port = %GATEWAY_PORT%; $listeners = @(Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue); foreach ($listener in $listeners) { $ownerId = [int]$listener.OwningProcess; $proc = Get-CimInstance Win32_Process -Filter ('ProcessId=' + $ownerId) -ErrorAction SilentlyContinue; if ($null -eq $proc) { continue }; if (($proc.Name -ine 'python.exe') -and ($proc.Name -ine 'pythonw.exe')) { Write-Host ('[ERROR] Port ' + $port + ' is occupied by unrelated process PID ' + $ownerId + ' (' + $proc.Name + ').'); exit 21 }; if ($proc.CommandLine -notlike '*app_svc.py*') { Write-Host ('[ERROR] Port ' + $port + ' is occupied by an unrelated Python process PID ' + $ownerId + '.'); exit 21 }; Write-Host ('[INFO] Stopping old SVC gateway PID ' + $ownerId + '...'); Stop-Process -Id $ownerId -Force -ErrorAction Stop }; $deadline = (Get-Date).AddSeconds(10); while ((Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue) -and ((Get-Date) -lt $deadline)) { Start-Sleep -Milliseconds 250 }; if (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue) { Write-Host ('[ERROR] Port ' + $port + ' was not released within 10 seconds.'); exit 22 }"
if errorlevel 1 (
    echo [ERROR] Unable to clear port %GATEWAY_PORT%. SVC gateway will not be started.
    pause
    exit /b 1
)

if exist "%NATIVE_PYTHON%" (
    set "RVC_NATIVE_ROCM=1"
    set "MIOPEN_LOG_LEVEL=3"
    echo [SVC] Starting MSST gateway with native AMD ROCm 7.2.1 + PyTorch 2.9.1...
    "%NATIVE_PYTHON%" "%~dp0launch.py" app_svc.py --is_nohalf
) else if exist "%LEGACY_PYTHON%" (
    set "RVC_NATIVE_ROCM="
    echo [WARN] Native ROCm runtime not found; falling back to legacy ZLUDA...
    "%LEGACY_PYTHON%" "%~dp0launch.py" app_svc.py --is_nohalf
) else (
    echo [ERROR] No Python runtime found.
    echo [ERROR] Run install_rocm_env.bat first.
    pause
    exit /b 2
)

set "SVC_EXIT_CODE=%ERRORLEVEL%"
echo.
echo [INFO] SVC gateway exited with code: %SVC_EXIT_CODE%
pause
exit /b %SVC_EXIT_CODE%
