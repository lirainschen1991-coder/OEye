param(
    [string]$Venv = ".venv",
    [int]$Port = 8620,
    [switch]$Headless
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $projectRoot

$venvPython = Join-Path $projectRoot "$Venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    throw "Virtual environment not found. Run scripts\install_environment.ps1 first."
}

$argsList = @("-m", "streamlit", "run", "app.py", "--server.port", "$Port")
if ($Headless) {
    $argsList += @("--server.headless", "true")
}

Write-Host "Starting OEye on http://localhost:$Port"
& $venvPython @argsList
