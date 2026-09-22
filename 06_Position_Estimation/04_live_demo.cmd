@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Run 01_setup.cmd first.
  pause
  exit /b 1
)
echo APPROXIMATE demo: may move the physical robot. This is NOT validated sag-free control.
echo Check actual ports, rig and clearance in config.approximate.json first.
.venv\Scripts\python.exe run_demo.py --live --config config.approximate.json
pause
