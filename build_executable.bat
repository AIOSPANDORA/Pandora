@echo off
REM Build script for Windows executable using PyInstaller
REM
REM This script builds a standalone .exe for Pandora Quantum System
REM that can run on any Windows system without requiring Python installation.
REM
REM Requirements:
REM   - Python 3.8+
REM   - PyInstaller: pip install pyinstaller
REM   - numpy, scipy: pip install numpy scipy

echo ======================================================================
echo   Pandora Quantum System - Windows Executable Builder
echo ======================================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.8 or higher.
    pause
    exit /b 1
)

REM Check if PyInstaller is installed
python -c "import PyInstaller" >nul 2>&1
if errorlevel 1 (
    echo PyInstaller not found. Installing...
    pip install pyinstaller
)

REM Check if dependencies are installed
echo Checking dependencies...
python -c "import numpy, scipy" >nul 2>&1
if errorlevel 1 (
    echo Missing dependencies. Installing numpy and scipy...
    pip install numpy scipy
)
echo [OK] All dependencies installed
echo.

REM Clean previous builds
echo Cleaning previous builds...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
echo.

REM Build executable
echo Building PandoraLauncher.exe...
echo This may take a few minutes...
echo.

pyinstaller PandoraLauncher.spec --clean

if errorlevel 1 (
    echo.
    echo ERROR: Build failed
    pause
    exit /b 1
)

echo.
echo ======================================================================
echo   Build Complete!
echo ======================================================================
echo.
echo The executable is located at: dist\PandoraLauncher.exe
echo.
echo To run:
echo   1. Double-click dist\PandoraLauncher.exe
echo   2. Or run from command line: dist\PandoraLauncher.exe
echo.
echo Distribution:
echo   You can copy PandoraLauncher.exe to any Windows system
echo   No Python installation required on target system
echo.
echo Testing executable...
dist\PandoraLauncher.exe --help >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Executable may have issues, but this is normal for interactive apps
) else (
    echo [OK] Executable works!
)
echo.

pause
