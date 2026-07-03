@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo === pipe_vision build (Windows) ===

if exist ".venv\Scripts\python.exe" (
    set "PY=.venv\Scripts\python.exe"
) else (
    set "PY=python"
)

where %PY% >nul 2>&1
if errorlevel 1 (
    echo Python не найден. Установите Python 3.8+ или создайте .venv
    exit /b 1
)

%PY% build_project.py %*
exit /b %ERRORLEVEL%
