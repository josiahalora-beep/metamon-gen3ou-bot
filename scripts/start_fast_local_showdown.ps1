param(
    [int]$SimulatorProcesses = 4,
    [switch]$UpdateFastFork
)

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$ServerDir = Join-Path $RepoRoot 'server\pokemon-showdown-fast'
$ForkUrl = 'https://github.com/hsahovic/Pokemon-Showdown.git'

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw 'git is required to install the accelerated Showdown fork.'
}
if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
    throw 'Node.js is required to run the accelerated Showdown server.'
}

if (-not (Test-Path $ServerDir)) {
    Write-Host "Cloning hsahovic/Pokemon-Showdown -> $ServerDir"
    git clone $ForkUrl $ServerDir
}
elseif ($UpdateFastFork) {
    Push-Location $ServerDir
    try {
        git pull --ff-only
    }
    finally {
        Pop-Location
    }
}

Push-Location $ServerDir
try {
    if (-not (Test-Path 'node_modules')) {
        Write-Host 'Installing Showdown dependencies...'
        if (Test-Path 'package-lock.json') {
            npm ci
        }
        else {
            npm install
        }
    }

    if (-not (Test-Path 'config\config.js')) {
        if (-not (Test-Path 'config\config-example.js')) {
            throw 'Showdown config template not found.'
        }
        Copy-Item 'config\config-example.js' 'config\config.js'
    }

    $ConfigPath = Join-Path $ServerDir 'config\config.js'
    $Config = Get-Content $ConfigPath -Raw
    $Setting = "exports.simulatorprocesses = $SimulatorProcesses;"
    if ($Config -match 'exports\.simulatorprocesses\s*=\s*\d+\s*;') {
        $Config = [regex]::Replace($Config, 'exports\.simulatorprocesses\s*=\s*\d+\s*;', $Setting)
    }
    else {
        $Config += "`r`n$Setting`r`n"
    }
    Set-Content -Path $ConfigPath -Value $Config -Encoding UTF8

    Write-Host "Starting accelerated local Showdown with $SimulatorProcesses simulator processes."
    Write-Host 'Server mode: --no-security (battle throttling/rate limits disabled for local research).'
    node pokemon-showdown start --no-security
}
finally {
    Pop-Location
}
