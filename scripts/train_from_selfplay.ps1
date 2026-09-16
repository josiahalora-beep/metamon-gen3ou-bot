param(
    [Alias("Input")]
    [string]$SelfPlayRoot = ".\battle_data\local_fast\trajectories",
    [int]$Checkpoint = 48,
    [string]$SaveDir = ".\cache\selfplay_finetunes",
    [string]$RunName = "synthetic_rl_v2_gen3_selfplay",
    [int]$Epochs = 2,
    [int]$StepsPerEpoch = 250,
    [int]$BatchSizePerGpu = 2,
    [int]$GradAccum = 1,
    [int]$DloaderWorkers = 0,
    [string]$PrevRunDir = '',
    [string]$PrevRunName = '',
    [int]$PrevCheckpoint = 0
)

$ErrorActionPreference = "Stop"

# Resolve paths from the repository root rather than the caller's current
# directory. This keeps the training pipeline reproducible when invoked from
# another working directory or by an orchestration script.
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

function Resolve-RepoPath {
    param([Parameter(Mandatory = $true)][string]$PathValue)

    if ([System.IO.Path]::IsPathRooted($PathValue)) {
        return [System.IO.Path]::GetFullPath($PathValue)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $RepoRoot $PathValue))
}

$SelfPlayRoot = Resolve-RepoPath $SelfPlayRoot
$SaveDir = Resolve-RepoPath $SaveDir

if (-not (Test-Path -LiteralPath $SelfPlayRoot -PathType Container)) {
    throw "Self-play trajectory root does not exist: $SelfPlayRoot"
}

# Self-play is partitioned by worker. AMAGO's custom_replays entry expects a
# single replay pile per role, so combine every worker's gen3ou trajectories
# into a deterministic staging directory without modifying the originals.
$workerDirs = @(
    Get-ChildItem -LiteralPath $SelfPlayRoot -Directory -ErrorAction Stop |
        Where-Object { $_.Name -match '^worker_[0-9]+$' } |
        Sort-Object Name
)

if ($workerDirs.Count -eq 0) {
    throw "No worker_* directories found under $SelfPlayRoot. Expected worker_NN\acceptor\gen3ou and worker_NN\challenger\gen3ou."
}

$stagingRoot = Join-Path $SelfPlayRoot "_training_replays"
$acceptorRoot = Join-Path $stagingRoot "acceptor"
$challengerRoot = Join-Path $stagingRoot "challenger"
$acceptorGen3 = Join-Path $acceptorRoot "gen3ou"
$challengerGen3 = Join-Path $challengerRoot "gen3ou"

# Rebuild the staging pile on every invocation so stale trajectories cannot
# leak into a later training run. The source worker directories are untouched.
if (Test-Path -LiteralPath $stagingRoot) {
    Remove-Item -LiteralPath $stagingRoot -Recurse -Force
}
New-Item -ItemType Directory -Force $acceptorGen3 | Out-Null
New-Item -ItemType Directory -Force $challengerGen3 | Out-Null

$acceptorCount = 0
$challengerCount = 0

foreach ($worker in $workerDirs) {
    $workerAcceptor = Join-Path $worker.FullName "acceptor\gen3ou"
    $workerChallenger = Join-Path $worker.FullName "challenger\gen3ou"

    if (-not (Test-Path -LiteralPath $workerAcceptor -PathType Container)) {
        throw "Missing acceptor AMAGO trajectories for $($worker.Name): $workerAcceptor"
    }
    if (-not (Test-Path -LiteralPath $workerChallenger -PathType Container)) {
        throw "Missing challenger AMAGO trajectories for $($worker.Name): $workerChallenger"
    }

    $acceptorFiles = @(Get-ChildItem -LiteralPath $workerAcceptor -File -Recurse | Sort-Object FullName)
    $challengerFiles = @(Get-ChildItem -LiteralPath $workerChallenger -File -Recurse | Sort-Object FullName)

    foreach ($file in $acceptorFiles) {
        $destination = Join-Path $acceptorGen3 $file.Name
        if (Test-Path -LiteralPath $destination) {
            throw "Duplicate acceptor trajectory filename while aggregating workers: $($file.Name)"
        }
        Copy-Item -LiteralPath $file.FullName -Destination $destination
        $acceptorCount++
    }

    foreach ($file in $challengerFiles) {
        $destination = Join-Path $challengerGen3 $file.Name
        if (Test-Path -LiteralPath $destination) {
            throw "Duplicate challenger trajectory filename while aggregating workers: $($file.Name)"
        }
        Copy-Item -LiteralPath $file.FullName -Destination $destination
        $challengerCount++
    }
}

if ($acceptorCount -eq 0) {
    throw "No acceptor AMAGO trajectory files found under worker_*\acceptor\gen3ou in $SelfPlayRoot"
}
if ($challengerCount -eq 0) {
    throw "No challenger AMAGO trajectory files found under worker_*\challenger\gen3ou in $SelfPlayRoot"
}

$yamlDir = Join-Path $RepoRoot "battle_data\selfplay_training"
New-Item -ItemType Directory -Force $yamlDir | Out-Null
$yamlPath = Join-Path $yamlDir ("$RunName.yaml")

$acceptorYaml = $acceptorRoot.Replace('\','/')
$challengerYaml = $challengerRoot.Replace('\','/')

if ($PrevRunDir -ne '') {
    if ($PrevRunName -eq '' -or $PrevCheckpoint -le 0) {
        throw 'PrevRunDir requires PrevRunName and PrevCheckpoint.'
    }

    $PrevRunDir = Resolve-RepoPath $PrevRunDir
    $prevDatasetPath = Join-Path $PrevRunDir "$PrevRunName\dataset_config.yaml"
    $prevDataset = (Resolve-Path $prevDatasetPath -ErrorAction Stop).Path.Replace('\','/')
    $prevBlock = "prev_dataset: `"$prevDataset`"`nprev_weight: 0.75"
} else {
    $prevBlock = "prev_dataset: self_play_dset.yaml`nprev_weight: 0.75"
}

@"
# Automatically generated iterative Gen 3 OU curriculum dataset.
# Worker-partitioned self-play is aggregated into the staging replay piles.
# Keep the pretrained/previous distribution while emphasizing fresh self-play.
replay_weight: 0.05
$prevBlock
custom_replays:
  - dir: "$acceptorYaml"
    weight: 0.10
  - dir: "$challengerYaml"
    weight: 0.10
formats:
  - gen3ou
anneal_epochs: $Epochs
"@ | Set-Content -Encoding UTF8 $yamlPath

Write-Host "Self-play trajectory root: $SelfPlayRoot"
Write-Host "Workers discovered: $($workerDirs.Count)"
Write-Host "Aggregated acceptor trajectories: $acceptorCount"
Write-Host "Aggregated challenger trajectories: $challengerCount"
Write-Host "Training replay staging root: $stagingRoot"
Write-Host "Dataset config: $yamlPath"

$cmd = @(
    '-m', 'metamon.rl.finetune',
    '--run_name', $RunName,
    '--save_dir', $SaveDir,
    '--base_model', 'SyntheticRLV2',
    '--dataset_config', $yamlPath,
    '--train_gin_config', 'finetune.gin',
    '--epochs', $Epochs,
    '--steps_per_epoch', $StepsPerEpoch,
    '--batch_size_per_gpu', $BatchSizePerGpu,
    '--grad_accum', $GradAccum,
    '--dloader_workers', $DloaderWorkers,
    '--eval_gens', '3'
)

if ($PrevRunDir -ne '') {
    $cmd += @('--prev_run_dir', $PrevRunDir, '--prev_run_name', $PrevRunName, '--prev_checkpoint', $PrevCheckpoint)
} else {
    $cmd += @('--base_checkpoint', $Checkpoint)
}

Write-Host "Starting SyntheticRLV2 finetuning: $RunName"
Push-Location $RepoRoot
try {
    python @cmd
    if ($LASTEXITCODE -ne 0) {
        throw "SyntheticRLV2 self-play finetuning failed with exit code $LASTEXITCODE."
    }
} finally {
    Pop-Location
}

Write-Host "Finetuning completed." -ForegroundColor Green
Write-Host "New checkpoints: $SaveDir\$RunName\ckpts\policy_weights\"
Write-Host "Dataset config:  $SaveDir\$RunName\dataset_config.yaml"
