@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
if not defined GO2_WSL_DISTRO set "GO2_WSL_DISTRO=Ubuntu-24.04"
wsl.exe -d "%GO2_WSL_DISTRO%" -- env LD_LIBRARY_PATH=/opt/gl-libs:/usr/lib/wsl/lib LIBGL_DRIVERS_PATH=/opt/gl-libs/dri MUJOCO_GL=egl .venv/bin/python scripts/run_full_coverage.py %*
set "result=%errorlevel%"
if not "%result%"=="0" echo Run failed. Keep this window and the output folder.
pause
exit /b %result%
