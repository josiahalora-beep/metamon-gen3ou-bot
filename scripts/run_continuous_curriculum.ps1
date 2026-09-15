param(
    [int]$BattlesPerIteration = 100,
    [int]$BattleWorkers = 2,
    [int]$Iterations = 0,
    [int]$SimulatorProcesses = 4,
    [int]$StartingCheckpoint = 48,
    [string]$TeamDir = 'public_gen3ou_teams',
    [string]$RootDir = 'battle_data\curriculum',
    [string]$SaveDir = 'cache\selfplay_finetunes',
    [int]$Epochs = 2,
    [int]$StepsPerEpoch = 250,
    [int]$BatchSizePerGpu = 2,
    [int]$GradAccum = 1,
    [int]$DloaderWorkers = 0,
    [switch]$DecisionDebug
)

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

if ($BattleWorkers -lt 1) { throw 'BattleWorkers must be at least 1.' }
if ($SimulatorProcesses -lt $BattleWorkers) {
    Write-Warning "SimulatorProcesses ($SimulatorProcesses) is below BattleWorkers ($BattleWorkers). Consider using at least one simulator process per worker."
}

New-Item -ItemType Directory -Force $RootDir | Out-Null
New-Item -ItemType Directory -Force $SaveDir | Out-Null

$previousRunName = ''
$previousCheckpoint = 0
$completed = 0
$iteration = 1

while (($Iterations -eq 0) -or ($iteration -le $Iterations)) {
    $runName = ('synthetic_rl_v2_curriculum_{0:D4}' -f $iteration)
    $iterationRoot = Join-Path $RootDir ('iter_{0:D4}' -f $iteration)
    $trajectoryDir = Join-Path $iterationRoot 'trajectories'
    $logDir = Join-Path $iterationRoot 'logs'
    New-Item -ItemType Directory -Force $trajectoryDir | Out-Null
    New-Item -ItemType Directory -Force $logDir | Out-Null

    $phase = ($iteration - 1) % 3
    if ($phase -eq 0) { $temperature = 0.85 }
    elseif ($phase -eq 1) { $temperature = 1.10 }
    else { $temperature = 1.35 }

    Write-Host ''
    Write-Host ('=' * 72)
    Write-Host "CURRICULUM ITERATION $iteration"
    Write-Host "Training run: $runName"
    Write-Host "Battles: $BattlesPerIteration | Workers: $BattleWorkers | Simulators: $SimulatorProcesses | Temperature: $temperature | Teams: $TeamDir"
    if ($previousRunName -eq '') {
        Write-Host "Opponent/model source: SyntheticRLV2 checkpoint $StartingCheckpoint"
    } else {
        Write-Host "Opponent/model source: $previousRunName checkpoint $previousCheckpoint"
    }
    Write-Host ('=' * 72)

    $battleArgs = @(
        '-Battles', $BattlesPerIteration,
        '-BattleWorkers', $BattleWorkers,
        '-SimulatorProcesses', $SimulatorProcesses,
        '-TeamDir', $TeamDir,
        '-LogDir', $logDir,
        '-TrajectoryDir', $trajectoryDir,
        '-Temperature', $temperature
    )
    if ($previousRunName -eq '') {
        $battleArgs += @('-Checkpoint', $StartingCheckpoint)
    } else {
        $battleArgs += @('-LocalRunDir', $SaveDir, '-LocalRunName', $previousRunName, '-LocalCheckpoint', $previousCheckpoint)
    }
    if ($DecisionDebug) { $battleArgs += '-DecisionDebug' }

    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $RepoRoot 'scripts\run_fast_local_synthetic.ps1') @battleArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Self-play collection failed in iteration $iteration. No training promotion performed."
    }

    $trainArgs = @(
        '-SelfPlayRoot', $trajectoryDir,
        '-SaveDir', $SaveDir,
        '-RunName', $runName,
        '-Epochs', $Epochs,
        '-StepsPerEpoch', $StepsPerEpoch,
        '-BatchSizePerGpu', $BatchSizePerGpu,
        '-GradAccum', $GradAccum,
        '-DloaderWorkers', $DloaderWorkers
    )
    if ($previousRunName -eq '') {
        $trainArgs += @('-Checkpoint', $StartingCheckpoint)
    } else {
        $trainArgs += @('-PrevRunDir', $SaveDir, '-PrevRunName', $previousRunName, '-PrevCheckpoint', $previousCheckpoint)
    }

    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $RepoRoot 'scripts\train_from_selfplay.ps1') @trainArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Training failed in iteration $iteration. Previous model remains the active model."
    }

    $newCheckpoint = $Epochs
    $checkpointPath = Join-Path $SaveDir "$runName\ckpts\policy_weights\policy_epoch_$newCheckpoint.pt"
    if (-not (Test-Path $checkpointPath)) {
        throw "Training reported success but checkpoint was not found: $checkpointPath"
    }

    $previousRunName = $runName
    $previousCheckpoint = $newCheckpoint
    $completed++
    Write-Host "ITERATION $iteration COMPLETE: $checkpointPath" -ForegroundColor Green
    Write-Host "Experience retained at: $trajectoryDir"

    $iteration++
}

Write-Host "Continuous curriculum stopped after $completed iterations." -ForegroundColor Green
