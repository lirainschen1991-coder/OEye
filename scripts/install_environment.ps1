param(
    [string]$Python = "python",
    [string]$Venv = ".venv",
    [ValidateSet("runtime", "standard", "dev")]
    [string]$Profile = "runtime",
    [string]$IndexUrl = "",
    [switch]$NoCache
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $projectRoot

if ($Profile -eq "runtime") {
    $requirements = "requirements-runtime.txt"
} elseif ($Profile -eq "standard") {
    $requirements = "requirements-standard.txt"
} else {
    $requirements = "requirements.txt"
}

if (-not (Test-Path $requirements)) {
    throw "Requirements file not found: $requirements"
}

Write-Host "Project root: $projectRoot"
Write-Host "Python command: $Python"
Write-Host "Dependency profile: $Profile ($requirements)"

& $Python --version
if (-not (Test-Path $Venv)) {
    Write-Host "Creating virtual environment: $Venv"
    & $Python -m venv $Venv
}

$venvPython = Join-Path $projectRoot "$Venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    throw "Virtual environment python not found: $venvPython"
}

& $venvPython -m pip install --upgrade pip setuptools wheel

$pipArgs = @("-m", "pip", "install", "-r", $requirements)
if ($IndexUrl.Trim().Length -gt 0) {
    $pipArgs += @("-i", $IndexUrl)
}
if ($NoCache) {
    $pipArgs += "--no-cache-dir"
}

Write-Host "Installing dependencies. This can take a long time for runtime profile..."
& $venvPython @pipArgs

Write-Host ""
Write-Host "Environment installation completed."
Write-Host "Run the app with:"
Write-Host "  powershell -ExecutionPolicy Bypass -File scripts\run_app.ps1"
