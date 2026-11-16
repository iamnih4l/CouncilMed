@echo off
title CouncilMed Medical AI — Live Demo Server
cd /d "%~dp0"
echo ======================================================================
echo    Starting CouncilMed All-In-One Demo Server
echo ======================================================================
echo.

if exist "venv\Scripts\python.exe" (
    "venv\Scripts\python.exe" start_demo.py
) else (
    python start_demo.py
)

pause
