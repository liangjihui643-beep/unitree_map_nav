@echo off
setlocal
cd /d "%~dp0"
if not defined GO2_WSL_DISTRO set "GO2_WSL_DISTRO=Ubuntu-24.04"
wsl.exe -d "%GO2_WSL_DISTRO%" -- env LD_LIBRARY_PATH=/opt/gl-libs:/usr/lib/wsl/lib LIBGL_DRIVERS_PATH=/opt/gl-libs/dri MUJOCO_GL=egl .venv/bin/python scripts/run_perception_demo.py %*
set "result=%errorlevel%"
if not "%result%"=="0" echo Demo failed with exit code %result%.
pause
exit /b %result%
