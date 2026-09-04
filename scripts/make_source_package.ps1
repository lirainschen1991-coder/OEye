param(
    [string]$OutputDir = "release",
    [string]$PackageName = "OEye_source_minimal"
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $projectRoot

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$releaseRoot = Join-Path $projectRoot $OutputDir
$staging = Join-Path $releaseRoot "$PackageName`_$timestamp"
$zipPath = "$staging.zip"

if (Test-Path $staging) {
    Remove-Item -LiteralPath $staging -Recurse -Force
}
New-Item -ItemType Directory -Path $staging | Out-Null

$items = @(
    "README.md",
    "LICENSE",
    ".gitignore",
    "app.py",
    "requirements.txt",
    "requirements-runtime.txt",
    "requirements-standard.txt",
    "requirements-bootstrap.txt",
    "pytest.ini",
    "OEye.jpeg",
    "install_runtime.bat",
    "run_app.bat",
    "src",
    "custom_models",
    "sample_data",
    "scripts",
    "packaging",
    "tests",
    "generate_timeseries_data.py",
    "generate_classification_data.py",
    "test_data_loader.py",
    "test_batch_loader.py",
    "test_batch_train.py",
    "test_batch_train_full.py",
    "test_functionality.py",
    "指南说明.md",
    "PACKAGING_SOURCE_DISTRIBUTION.md",
    "PACKAGING_PROTECTED_DELIVERY.md"
)

foreach ($item in $items) {
    $source = Join-Path $projectRoot $item
    if (Test-Path $source) {
        $target = Join-Path $staging $item
        $parent = Split-Path -Parent $target
        if (-not (Test-Path $parent)) {
            New-Item -ItemType Directory -Path $parent | Out-Null
        }
        Copy-Item -LiteralPath $source -Destination $target -Recurse -Force
    }
}

# Source folders may contain local bytecode caches from development or tests.
# Remove them only from this newly-created staging directory.
$generatedDirectories = Get-ChildItem -LiteralPath $staging -Recurse -Directory -Force | Where-Object {
    $_.Name -in @('__pycache__', '.pytest_cache', '.venv', 'venv', 'env', 'build', 'dist', 'saved_models', 'catboost_info')
}
foreach ($directory in ($generatedDirectories | Sort-Object FullName -Descending)) {
    Remove-Item -LiteralPath $directory.FullName -Recurse -Force
}
$generatedFiles = Get-ChildItem -LiteralPath $staging -Recurse -File -Force | Where-Object {
    $_.Extension.ToLowerInvariant() -in @('.pyc', '.pyo', '.pkl', '.h5', '.keras', '.log', '.tmp', '.out', '.dat')
}
foreach ($file in $generatedFiles) {
    Remove-Item -LiteralPath $file.FullName -Force
}

$unexpected = Get-ChildItem -LiteralPath $staging -Recurse -Force | Where-Object {
    ($_.PSIsContainer -and $_.Name -in @('__pycache__', '.pytest_cache', '.venv', 'venv', 'env', 'build', 'dist', 'saved_models', 'catboost_info')) -or
    (!$_.PSIsContainer -and $_.Extension.ToLowerInvariant() -in @('.pyc', '.pyo', '.pkl', '.h5', '.keras', '.log', '.tmp', '.out', '.dat'))
}
if ($unexpected) {
    $paths = ($unexpected | ForEach-Object { $_.FullName }) -join "`n"
    throw "Source package contains forbidden generated or large artifacts:`n$paths"
}

Compress-Archive -Path (Join-Path $staging "*") -DestinationPath $zipPath -Force
if (-not (Test-Path -LiteralPath $zipPath)) {
    throw "Source package was not created: $zipPath"
}
$sizeMb = [math]::Round((Get-Item $zipPath).Length / 1MB, 2)

Write-Host "Created source package: $zipPath"
Write-Host "Package size: $sizeMb MB"
Write-Host "Large raw data, .git, caches, saved_models, and generated artifacts are intentionally excluded."
