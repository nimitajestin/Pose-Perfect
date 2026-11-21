# Pose Perfect - one-shot setup + start script (Windows / PowerShell)
# Usage: .\start.ps1            start everything (sets up on first run)
#        .\start.ps1 -Setup     only run setup, do not start the servers
#        .\start.ps1 -Clean     delete .venv + node_modules, then set up and start

param(
    [switch]$Setup,
    [switch]$Clean,
    [int]$ApiPort = 8000,
    [int]$WebPort = 5173
)

$ErrorActionPreference = 'Stop'

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$VenvDir     = Join-Path $ProjectRoot '.venv'
$FrontendDir = Join-Path $ProjectRoot 'Frontend'

function Write-Info { param($Message) Write-Host "==> $Message" -ForegroundColor Cyan }
function Write-Fail { param($Message) Write-Host "ERROR: $Message" -ForegroundColor Red; exit 1 }

if ($Clean) {
    Write-Info 'Removing .venv and Frontend/node_modules...'
    Remove-Item -Recurse -Force $VenvDir -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force (Join-Path $FrontendDir 'node_modules') -ErrorAction SilentlyContinue
}

# ---------------------------------------------------------------- prerequisites
Write-Info 'Checking prerequisites'

$PythonBin = $null
foreach ($candidate in @('python', 'python3', 'py')) {
    $cmd = Get-Command $candidate -ErrorAction SilentlyContinue
    if (-not $cmd) { continue }
    # `py` needs an explicit version selector; the others run directly.
    $probeArgs = if ($candidate -eq 'py') { @('-3', '-c', 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)') }
                 else { @('-c', 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)') }
    & $cmd.Source @probeArgs 2>$null
    if ($LASTEXITCODE -eq 0) {
        $PythonBin = $cmd.Source
        $PythonPrefix = if ($candidate -eq 'py') { @('-3') } else { @() }
        break
    }
}
if (-not $PythonBin) { Write-Fail 'Python 3.9+ not found. Install it from https://www.python.org/downloads/ and tick "Add python.exe to PATH".' }

if (-not (Get-Command node -ErrorAction SilentlyContinue)) { Write-Fail 'Node.js not found. Install it from https://nodejs.org/ (v18 or newer).' }
if (-not (Get-Command npm  -ErrorAction SilentlyContinue)) { Write-Fail 'npm not found. It ships with Node.js - reinstall from https://nodejs.org/' }

Write-Host "    Python: $(& $PythonBin @PythonPrefix --version) ($PythonBin)"
Write-Host "    Node:   $(node --version)"

# ---------------------------------------------------------------- backend setup
if (-not (Test-Path $VenvDir)) {
    Write-Info 'Creating Python virtual environment (.venv)'
    & $PythonBin @PythonPrefix -m venv $VenvDir
}

$VenvPy = Join-Path $VenvDir 'Scripts\python.exe'
if (-not (Test-Path $VenvPy)) { Write-Fail "Virtual environment looks broken (no $VenvPy). Re-run with -Clean." }

# Install requirements only when they changed since the last successful install.
$ReqStamp = Join-Path $VenvDir '.requirements.sha'
$ReqHash  = (Get-FileHash 'requirements.txt' -Algorithm SHA256).Hash
if ((-not (Test-Path $ReqStamp)) -or ((Get-Content $ReqStamp -Raw).Trim() -ne $ReqHash)) {
    Write-Info 'Installing Python dependencies (first run takes a few minutes)'
    & $VenvPy -m pip install --upgrade pip --quiet
    & $VenvPy -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { Write-Fail 'pip install failed. Scroll up for the reason.' }
    Set-Content -Path $ReqStamp -Value $ReqHash
} else {
    Write-Info 'Python dependencies already installed'
}

# ---------------------------------------------------------------- model weights
if (-not (Test-Path (Join-Path $ProjectRoot 'yolo11m-pose.pt'))) {
    Write-Info 'Downloading YOLO11 pose model'
    & $VenvPy -c "from ultralytics import YOLO; YOLO('yolo11m-pose.pt')"
}

# ---------------------------------------------------------------- frontend setup
if (-not (Test-Path (Join-Path $FrontendDir 'node_modules'))) {
    Write-Info 'Installing frontend dependencies'
    Push-Location $FrontendDir
    npm install --no-audit --no-fund
    $npmExit = $LASTEXITCODE
    Pop-Location
    if ($npmExit -ne 0) { Write-Fail 'npm install failed. Scroll up for the reason.' }
} else {
    Write-Info 'Frontend dependencies already installed'
}

if ($Setup) {
    Write-Info 'Setup complete. Run .\start.ps1 to launch the app.'
    exit 0
}

# ---------------------------------------------------------------- start servers
function Test-PortBusy { param([int]$Port)
    $null -ne (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
}
if (Test-PortBusy $ApiPort) { Write-Fail "Port $ApiPort is already in use. Stop that process or run: .\start.ps1 -ApiPort 8001" }
if (Test-PortBusy $WebPort) { Write-Fail "Port $WebPort is already in use. Stop that process or run: .\start.ps1 -WebPort 5174" }

$processes = @()

Write-Info "Starting API server on http://localhost:$ApiPort"
# Run from the project root: the API resolves trained_models\ and the model weights relative to the CWD.
$processes += Start-Process -FilePath $VenvPy `
    -ArgumentList '-m', 'uvicorn', 'api.api_server:app', '--host', '0.0.0.0', '--port', "$ApiPort" `
    -WorkingDirectory $ProjectRoot -NoNewWindow -PassThru

Write-Info 'Waiting for the API to become ready (loading pose models)'
$apiReady = $false
foreach ($i in 1..120) {
    try {
        Invoke-WebRequest -Uri "http://localhost:$ApiPort/" -UseBasicParsing -TimeoutSec 2 | Out-Null
        $apiReady = $true
        Write-Host '    API is ready.'
        break
    } catch {
        if ($processes[0].HasExited) { Write-Fail 'API server exited during startup. Scroll up for the Python traceback.' }
        Start-Sleep -Seconds 1
    }
}
if (-not $apiReady) { Write-Host '    API did not respond in time; starting the frontend anyway.' -ForegroundColor Yellow }

Write-Info "Starting frontend on http://localhost:$WebPort"
# Start-Process needs npm.cmd explicitly - Get-Command npm can resolve to npm.ps1,
# which Start-Process would try to open rather than run.
$npmCmdInfo = Get-Command npm.cmd -ErrorAction SilentlyContinue
if (-not $npmCmdInfo) { $npmCmdInfo = Get-Command npm }
$npmCmd = $npmCmdInfo.Source
$processes += Start-Process -FilePath $npmCmd `
    -ArgumentList 'run', 'dev', '--', '--port', "$WebPort", '--strictPort' `
    -WorkingDirectory $FrontendDir -NoNewWindow -PassThru

Write-Host @"

------------------------------------------------------------
  Pose Perfect is running
    App : http://localhost:$WebPort
    API : http://localhost:$ApiPort  (docs at /docs)

  Press Ctrl+C to stop both servers.
------------------------------------------------------------

"@

# Stop both servers on Ctrl+C or when either one dies.
try {
    while ($true) {
        Start-Sleep -Seconds 1
        if ($processes | Where-Object { $_.HasExited }) { break }
    }
} finally {
    Write-Info 'Shutting down'
    foreach ($proc in $processes) {
        if (-not $proc.HasExited) {
            # /T kills the whole tree: `npm run dev` starts vite as a child, so
            # stopping npm alone would leave the dev server holding the port.
            taskkill /PID $proc.Id /T /F 2>&1 | Out-Null
        }
    }
}
