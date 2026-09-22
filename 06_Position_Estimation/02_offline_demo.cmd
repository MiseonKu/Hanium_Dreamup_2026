@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Run 01_setup.cmd first.
  pause
  exit /b 1
)
.venv\Scripts\python.exe run_demo.py --dry-run
pause
