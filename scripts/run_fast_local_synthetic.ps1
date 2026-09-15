param(
    [int]$Battles = 100,
    [int]$SimulatorProcesses = 4,
    [string]$Username = 'LocalSynthetic',
    [string]$TeamDir = 'public_gen3ou_teams',
    [string]$LogDir = 'battle_data\local_fast',
    [switch]$UpdateFastFork,
    [switch]$DecisionDebug
)

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$StartScript = Join-Path $RepoRoot 'scripts\start_fast_local_showdown.ps1'

$ServerJob = Start-Process -FilePath 'powershell' -ArgumentList @(
    '-NoProfile',
    '-ExecutionPolicy', 'Bypass',
    '-File', $StartScript,
    '-SimulatorProcesses', $SimulatorProcesses
) -WorkingDirectory $RepoRoot -PassThru -WindowStyle Hidden

try {
    Write-Host "Starting local accelerated Showdown (PID $($ServerJob.Id))..."
    $ready = $false
    for ($i = 0; $i -lt 60; $i++) {
        Start-Sleep -Milliseconds 500
        if (Test-NetConnection 127.0.0.1 -Port 8000 -InformationLevel Quiet) {
            $ready = $true
            break
        }
        if ($ServerJob.HasExited) {
            throw "Fast Showdown server exited early with code $($ServerJob.ExitCode)."
        }
    }
    if (-not $ready) {
        throw 'Fast Showdown server did not open port 8000 within 30 seconds.'
    }

    $Args = @(
        'scripts\play_local_fast_synthetic.py',
        '--username', $Username,
        '--password', '',
        '--team-dir', $TeamDir,
        '--battles', $Battles,
        '--enable-battle-ai',
        '--log-dir', $LogDir,
        '--analysis'
    )
    if ($DecisionDebug) {
        $Args += '--decision-debug'
    }

    Write-Host "Running $Battles local Gen 3 OU battles against the accelerated server..."
    python @Args
    if ($LASTEXITCODE -ne 0) {
        throw "Local SyntheticRLV2 runner exited with code $LASTEXITCODE."
    }
}
finally {
    if (-not $ServerJob.HasExited) {
        Write-Host 'Stopping accelerated Showdown server...'
        & taskkill.exe /PID $ServerJob.Id /T /F | Out-Null
    }
}
