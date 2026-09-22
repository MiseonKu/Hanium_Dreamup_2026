@echo off
cd /d "%~dp0"
py -3.11 -m venv .venv
if errorlevel 1 goto fail
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto fail
.venv\Scripts\python.exe -m unittest -v test_demo
if errorlevel 1 goto fail
echo Setup and offline tests passed.
pause
exit /b 0
:fail
echo Setup or tests failed. See errors above. No hardware commands were sent.
pause
exit /b 1
