@echo off
REM Windows universal launcher for AIOS/Pandora (NVIDIA/Intel/ASUS ROG)
where python
if %ERRORLEVEL% neq 0 (
    echo Python not found! Please install Python 3.8+.
    pause
    exit /b 1
)
python Launch_AI.py
pause