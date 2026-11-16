@echo off
title CouncilMed - Deploy Backend to Hugging Face Spaces
cd /d "%~dp0"
echo ========================================================
echo   CouncilMed: 24/7 Cloud Backend Deployment
echo ========================================================
echo.

if exist "venv\Scripts\python.exe" (
    venv\Scripts\python.exe scripts\deploy_hf_space.py
) else (
    python scripts\deploy_hf_space.py
)

echo.
pause
