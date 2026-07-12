@echo off
chcp 65001 >nul
setlocal
title RVCSVC-API-MSST Native AMD ROCm Setup
cd /d "%~dp0"

echo [INFO] install_env.bat now installs the native AMD ROCm runtime.
echo [INFO] Redirecting to install_rocm_env.bat...
call "%~dp0install_rocm_env.bat"
set "SETUP_EXIT_CODE=%ERRORLEVEL%"
exit /b %SETUP_EXIT_CODE%
