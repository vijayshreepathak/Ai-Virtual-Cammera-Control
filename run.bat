@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    python -m venv .venv
    .venv\Scripts\pip install -r requirements.txt
)
.venv\Scripts\python.exe smoke_test.py
if errorlevel 1 (
    echo Smoke tests failed. Fix errors above before running the app.
    pause
    exit /b 1
)
echo.
echo Starting AI Virtual Camera Control...
.venv\Scripts\python.exe app.py
