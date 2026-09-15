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

$SelfPlayRoot = (Resolve-Path $SelfPlayRoot -ErrorAction Stop).Path
$acceptorRoot = Join-Path $SelfPlayRoot "acceptor"
$challengerRoot = Join-Path $SelfPlayRoot "challenger"
if (-not (Test-Path (Join-Path $acceptorRoot "gen3ou"))) {
    throw "Missing acceptor AMAGO trajectories at $acceptorRoot\gen3ou."
}
if (-not (Test-Path (Join-Path $challengerRoot "gen3ou"))) {
    throw "Missing challenger AMAGO trajectories at $challengerRoot\gen3ou."
}

$yamlDir = Join-Path $PWD "battle_data\selfplay_training"
New-Item -ItemType Directory -Force $yamlDir | Out-Null
$yamlPath = Join-Path $yamlDir ("$RunName.yaml")

$acceptorYaml = $acceptorRoot.Replace('\','/')
$challengerYaml = $challengerRoot.Replace('\','/')

if ($PrevRunDir -ne '') {
    if ($PrevRunName -eq '' -or $PrevCheckpoint -le 0) {
        throw 'PrevRunDir requires PrevRunName and PrevCheckpoint.'
    }
    $prevDataset = (Resolve-Path (Join-Path $PrevRunDir "$PrevRunName\dataset_config.yaml") -ErrorAction Stop).Path.Replace('\','/')
    $prevBlock = "prev_dataset: `"$prevDataset`"`nprev_weight: 0.75"
    $weightBlock = ""
} else {
    $prevBlock = "prev_dataset: self_play_dset.yaml`nprev_weight: 0.75"
    $weightBlock = ""
}

@"
# Automatically generated iterative Gen 3 OU curriculum dataset.
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
python @cmd
if ($LASTEXITCODE -ne 0) {
    throw "SyntheticRLV2 self-play finetuning failed with exit code $LASTEXITCODE."
}

Write-Host "Finetuning completed." -ForegroundColor Green
Write-Host "New checkpoints: $SaveDir\$RunName\ckpts\policy_weights\"
Write-Host "Dataset config:  $SaveDir\$RunName\dataset_config.yaml"
