@echo off
title CouncilMed - Instant Cloud Patch (app.py)
cd /d "%~dp0"
echo ========================================================
echo   CouncilMed: Instant Cloud Patch (app.py)
echo ========================================================
echo.

if exist "venv\Scripts\python.exe" (
    venv\Scripts\python.exe scripts\patch_space_app.py
) else (
    python scripts\patch_space_app.py
)

echo.
pause
