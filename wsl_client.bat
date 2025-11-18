@echo off
REM Pandora AIOS WSL Access Terminal Launcher for Windows
REM This batch file makes it easy to connect to the WSL terminal

SET HOST=localhost
SET PORT=9000
SET TOKEN=

echo ========================================
echo Pandora AIOS WSL Access Terminal
echo ========================================
echo.

REM Check if PowerShell is available
where pwsh >nul 2>nul
IF %ERRORLEVEL% EQU 0 (
    SET PS=pwsh
) ELSE (
    SET PS=powershell
)

REM Parse command line arguments
IF "%1"=="" GOTO interactive
IF "%1"=="--token" SET TOKEN=%2
IF "%1"=="--status" GOTO status
IF "%1"=="--help" GOTO help

:execute
echo Executing command on WSL...
%PS% -ExecutionPolicy Bypass -File wsl_client.ps1 -Host %HOST% -Port %PORT% -Token %TOKEN% -Command "%*"
GOTO end

:interactive
echo Starting interactive mode...
echo.
IF "%TOKEN%"=="" (
    echo ERROR: Authentication token required!
    echo Please set TOKEN in this script or use --token parameter
    echo.
    GOTO help
)
%PS% -ExecutionPolicy Bypass -File wsl_client.ps1 -Host %HOST% -Port %PORT% -Token %TOKEN% -Interactive
GOTO end

:status
echo Getting server status...
%PS% -ExecutionPolicy Bypass -File wsl_client.ps1 -Host %HOST% -Port %PORT% -Token %TOKEN% -Status
GOTO end

:help
echo.
echo Usage: wsl_client.bat [options] [command]
echo.
echo Options:
echo   --status              Get server status
echo   --token TOKEN         Set authentication token
echo   --help                Show this help
echo.
echo Examples:
echo   wsl_client.bat --token abc123
echo   wsl_client.bat --token abc123 --status
echo   wsl_client.bat --token abc123 ls -la /home
echo.
echo Note: Edit this file to set default HOST, PORT, and TOKEN
echo.

:end
pause
