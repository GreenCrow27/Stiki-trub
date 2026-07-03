# Сборка pipe_vision (Windows)
# Запуск:  .\build.ps1
#          .\build.ps1 --no-deps

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (Test-Path ".venv\Scripts\python.exe") {
    & ".venv\Scripts\python.exe" build_project.py @args
} else {
    python build_project.py @args
}

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
