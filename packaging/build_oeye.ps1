param(
    [string]$Python = "python",
    [switch]$SkipPyInstaller,
    [switch]$SkipSmokeTest,
    [switch]$CheckOnly,
    [switch]$BuildInstaller,
    [string]$InnoCompiler = ""
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$BuildRoot = Join-Path $ProjectRoot "build"
$ProtectedRoot = Join-Path $BuildRoot "oeye_protected_sources"
$DistRoot = Join-Path $ProjectRoot "dist"
$DistApp = Join-Path $DistRoot "OEye"

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Assert-Command {
    param([string]$CommandName)
    if (-not (Get-Command $CommandName -ErrorAction SilentlyContinue)) {
        throw "Missing command: $CommandName"
    }
}

function Assert-PythonImport {
    param(
        [string]$ImportName,
        [string]$PackageName
    )
    $code = "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('$ImportName') else 1)"
    & $Python -c $code
    if ($LASTEXITCODE -ne 0) {
        throw "Missing Python package: $PackageName (import $ImportName failed)"
    }
}

function Assert-UnderProject {
    param([string]$PathToCheck)
    $resolved = [System.IO.Path]::GetFullPath($PathToCheck)
    $project = [System.IO.Path]::GetFullPath($ProjectRoot)
    if (-not $resolved.StartsWith($project, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to operate outside project root: $resolved"
    }
}

function Reset-Directory {
    param([string]$PathToReset)
    Assert-UnderProject $PathToReset
    if (Test-Path $PathToReset) {
        Remove-Item -LiteralPath $PathToReset -Recurse -Force
    }
    New-Item -ItemType Directory -Path $PathToReset | Out-Null
}

function Copy-ProtectedTree {
    Reset-Directory $ProtectedRoot

    Copy-Item -LiteralPath (Join-Path $ProjectRoot "app.py") -Destination $ProtectedRoot
    Copy-Item -LiteralPath (Join-Path $ProjectRoot "src") -Destination (Join-Path $ProtectedRoot "src") -Recurse

    if (Test-Path (Join-Path $ProjectRoot "custom_models")) {
        Copy-Item -LiteralPath (Join-Path $ProjectRoot "custom_models") -Destination (Join-Path $ProtectedRoot "custom_models") -Recurse
    }

    Push-Location $ProtectedRoot
    try {
        & $Python -m compileall -b -q app.py src custom_models
        if ($LASTEXITCODE -ne 0) {
            throw "compileall failed with exit code $LASTEXITCODE"
        }
    }
    finally {
        Pop-Location
    }

    Get-ChildItem -LiteralPath $ProtectedRoot -Recurse -Filter "*.py" | Remove-Item -Force
}

function Copy-LightSampleData {
    $source = Join-Path $ProjectRoot "sample_data"
    $target = Join-Path $DistApp "sample_data"
    if (-not (Test-Path $source) -or -not (Test-Path $DistApp)) {
        return
    }

    if (Test-Path $target) {
        Remove-Item -LiteralPath $target -Recurse -Force
    }
    New-Item -ItemType Directory -Path $target | Out-Null

    Get-ChildItem -LiteralPath $source -File | Where-Object {
        $_.Extension -in ".csv", ".json", ".md", ".txt" -and $_.Length -lt 5MB
    } | ForEach-Object {
        Copy-Item -LiteralPath $_.FullName -Destination $target
    }
}

function Test-DeliveryContents {
    param([string]$RootToCheck)

    if (-not (Test-Path $RootToCheck)) {
        throw "Delivery folder not found: $RootToCheck"
    }

    $forbidden = @(
        "app.py",
        "src\*.py",
        "src\**\*.py",
        "custom_models\*.py",
        "custom_models\**\*.py"
    )

    foreach ($pattern in $forbidden) {
        $matches = Get-ChildItem -LiteralPath $RootToCheck -Recurse -File -ErrorAction SilentlyContinue |
            Where-Object { $_.FullName -like ("*" + $pattern) }
        if ($matches) {
            $first = $matches | Select-Object -First 1
            throw "Plain business source leaked into dist: $($first.FullName)"
        }
    }

    $largeData = Get-ChildItem -LiteralPath $RootToCheck -Recurse -File -ErrorAction SilentlyContinue |
        Where-Object {
            $_.Name -in "Forces.dat", "RotorDyn.dat", "PatformDyn.dat", "PatformDyn.txt", "OutputSummary.dat", "浮动平台运动.out" -or
            ($_.Extension -eq ".out" -and $_.Length -gt 1MB) -or
            ($_.Extension -eq ".dat" -and $_.Length -gt 5MB -and $_.FullName -notmatch "\\_internal\\") -or
            $_.FullName -match "\\saved_models\\" -or
            $_.FullName -match "\\\.venv\\" -or
            $_.FullName -match "\\pip-cache\\"
        }
    if ($largeData) {
        $first = $largeData | Select-Object -First 1
        throw "Forbidden data/model/cache artifact found: $($first.FullName)"
    }
}

function Get-FolderSizeMb {
    param([string]$PathToMeasure)
    $bytes = (Get-ChildItem -LiteralPath $PathToMeasure -Recurse -File | Measure-Object -Property Length -Sum).Sum
    return [Math]::Round(($bytes / 1MB), 1)
}

function Stop-OEye {
    $procs = @(Get-Process OEye -ErrorAction SilentlyContinue)
    if ($procs.Count -gt 0) {
        Write-Host "Stopping existing OEye process(es): $($procs.Id -join ', ')"
        $procs | Stop-Process -Force
        Start-Sleep -Seconds 2
    }
}

function Wait-Health {
    param(
        [int]$Port,
        [int]$TimeoutSeconds = 75
    )
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    $healthUrl = "http://127.0.0.1:$Port/_stcore/health"
    $lastError = $null

    while ((Get-Date) -lt $deadline) {
        try {
            $response = Invoke-WebRequest -UseBasicParsing -Uri $healthUrl -TimeoutSec 3
            if ($response.StatusCode -eq 200) {
                return
            }
        } catch {
            $lastError = $_.Exception.Message
        }
        Start-Sleep -Milliseconds 700
    }

    throw "Health check failed for $healthUrl. Last error: $lastError"
}

function Test-PackagedDependencyUi {
    param([string]$ExePath)

    if (-not (Test-Path -LiteralPath $ExePath)) {
        throw "Executable not found: $ExePath"
    }

    Write-Step "Checking frozen hidden imports"
    $importProc = Start-Process -FilePath $ExePath -ArgumentList @("--check-import", "streamlit.runtime.scriptrunner.magic_funcs") -Wait -PassThru -WindowStyle Hidden
    if ($importProc.ExitCode -ne 0) {
        throw "Frozen import check failed for streamlit.runtime.scriptrunner.magic_funcs"
    }

    Write-Step "Smoke testing dependency UI"
    $port = 26340
    $proc = Start-Process -FilePath $ExePath -ArgumentList @("--serve-dependency", "--port", "$port") -PassThru -WindowStyle Hidden
    try {
        Wait-Health -Port $port
        $page = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:$port/" -TimeoutSec 20
        $badNeedles = @(
            "ModuleNotFoundError",
            "Traceback",
            "No module named",
            "ImportError",
            "AttributeError",
            "Streamlit server is not responding",
            "配置加载失败"
        )
        $hits = @($badNeedles | Where-Object { $page.Content -like "*$_*" })
        if ($hits.Count -gt 0) {
            throw "Smoke test found error text: $($hits -join ', ')"
        }
        Write-Host "Dependency UI smoke test passed on port $port."
    } finally {
        $current = Get-Process -Id $proc.Id -ErrorAction SilentlyContinue
        if ($current) {
            $current | Stop-Process -Force
        }
        Start-Sleep -Seconds 2
    }
}

Set-Location $ProjectRoot

Write-Host "== OEye protected delivery build =="
Write-Host "Project: $ProjectRoot"
Write-Host "Python : $Python"

Write-Step "Checking tools and bootstrap packages"
Assert-Command $Python
Assert-PythonImport "PyInstaller" "pyinstaller"
Assert-PythonImport "streamlit" "streamlit"
Assert-PythonImport "webview" "pywebview"
Assert-PythonImport "packaging" "packaging"

if ($CheckOnly) {
    Write-Host "Environment check passed. No build was started because -CheckOnly was set."
    return
}

Write-Step "Stopping old OEye processes"
Stop-OEye

Write-Step "Preparing protected bytecode"
Copy-ProtectedTree
Write-Host "Protected bytecode prepared: $ProtectedRoot"

if (-not $SkipPyInstaller) {
    Write-Step "Building dist with PyInstaller"
    & $Python -m PyInstaller --clean --noconfirm (Join-Path $PSScriptRoot "OEye.spec")
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller failed with exit code $LASTEXITCODE"
    }
    Copy-LightSampleData
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot "CUSTOMER_README.txt") -Destination (Join-Path $DistApp "CUSTOMER_README.txt") -Force
    Test-DeliveryContents $DistApp

    $sizeMb = Get-FolderSizeMb $DistApp
    Write-Host "Dist folder: $DistApp"
    Write-Host "Dist size  : $sizeMb MB"

    $exe = Join-Path $DistApp "OEye.exe"
    if (-not (Test-Path $exe)) {
        throw "OEye.exe not found after build."
    }

    if (-not $SkipSmokeTest) {
        Test-PackagedDependencyUi -ExePath $exe
    }
}
else {
    Test-DeliveryContents $ProtectedRoot
    Write-Host "Skipped PyInstaller. Protected source check passed."
}

if ($BuildInstaller) {
    if (-not $InnoCompiler) {
        $candidate = "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe"
        if (Test-Path $candidate) {
            $InnoCompiler = $candidate
        }
    }
    if (-not $InnoCompiler -or -not (Test-Path $InnoCompiler)) {
        throw "Inno Setup compiler not found. Pass -InnoCompiler path."
    }
    $iss = Join-Path $PSScriptRoot "OEye.iss"
    if (-not (Test-Path $iss)) {
        throw "Inno Setup script not found: $iss"
    }
    & $InnoCompiler $iss
}
