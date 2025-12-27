# Requires: Windows PowerShell 5.1
# Purpose: Bootstrap a portable Python runtime inside the repo and install dependencies.

$ErrorActionPreference = "Stop"

function Write-Info($msg) {
    Write-Host "[INFO] $msg" -ForegroundColor Cyan
}
function Write-Warn($msg) {
    Write-Host "[WARN] $msg" -ForegroundColor Yellow
}
function Write-Err($msg) {
    Write-Host "[ERROR] $msg" -ForegroundColor Red
}

# Resolve paths
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$repoRoot = Split-Path -Parent $scriptDir
$runtimeDir = Join-Path $repoRoot "python_runtime"

# Choose Python version (embeddable)
$pyVersion = "3.11.9"
$arch = if ([Environment]::Is64BitOperatingSystem) { "amd64" } else { "win32" }
$embedZipUrl = "https://www.python.org/ftp/python/$pyVersion/python-$pyVersion-embed-$arch.zip"
$zipFile = Join-Path $runtimeDir "python-embed.zip"

# Bootstrap runtime dir
if (!(Test-Path $runtimeDir)) {
    Write-Info "Creating runtime directory: $runtimeDir"
    New-Item -ItemType Directory -Path $runtimeDir | Out-Null
}

# Download embeddable Python
Write-Info "Downloading embeddable Python $pyVersion ($arch) ..."
try {
    Invoke-WebRequest -Uri $embedZipUrl -OutFile $zipFile -UseBasicParsing
} catch {
    Write-Err "Failed to download Python from $embedZipUrl"
    throw
}

# Extract
Write-Info "Extracting Python runtime ..."
Expand-Archive -Path $zipFile -DestinationPath $runtimeDir -Force
Remove-Item $zipFile -Force

# Enable site-packages by ensuring import site in *_pth file
$pthFile = Get-ChildItem -Path $runtimeDir -Filter "python*._pth" | Select-Object -First 1
if ($null -eq $pthFile) {
    Write-Warn "No _pth file found; continuing"
} else {
    Write-Info "Configuring $($pthFile.Name) to enable site-packages"
    $content = Get-Content $pthFile.FullName
    if ($content -notcontains "import site") {
        Add-Content -Path $pthFile.FullName -Value "import site"
    } else {
        # Ensure not commented out
        $updated = $content | ForEach-Object { $_ -replace "^#\s*import site","import site" }
        Set-Content -Path $pthFile.FullName -Value $updated
    }
}

# Download get-pip.py
$getPipUrl = "https://bootstrap.pypa.io/get-pip.py"
$getPipPath = Join-Path $runtimeDir "get-pip.py"
Write-Info "Downloading get-pip.py ..."
Invoke-WebRequest -Uri $getPipUrl -OutFile $getPipPath -UseBasicParsing

# Install pip into the embedded runtime
$pythonExe = Join-Path $runtimeDir "python.exe"
if (!(Test-Path $pythonExe)) { $pythonExe = Join-Path $runtimeDir "python311.dll" }
if (!(Test-Path $pythonExe)) {
    Write-Err "python.exe not found in $runtimeDir"
    throw "Python runtime not extracted correctly"
}

Write-Info "Installing pip ..."
& $pythonExe $getPipPath
Remove-Item $getPipPath -Force

# Install project requirements
$requirementsPath = Join-Path $scriptDir "requirements.txt"
if (!(Test-Path $requirementsPath)) {
    Write-Warn "requirements.txt not found at $requirementsPath; skipping dependency install"
} else {
    Write-Info "Installing Python dependencies from requirements.txt ..."
    & $pythonExe -m pip install -r $requirementsPath --no-cache-dir
}

Write-Info "Portable Python runtime is ready at: $runtimeDir"
