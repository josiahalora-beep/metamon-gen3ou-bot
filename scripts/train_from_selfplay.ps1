param(
    [string]$Input = ".\battle_data\local_fast",
    [int]$Checkpoint = 48,
    [string]$Output = ".\battle_data\selfplay_training\selfplay.jsonl"
)

$ErrorActionPreference = "Stop"

Write-Host "Exporting local self-play trajectories..."
python .\scripts\export_selfplay_training.py --input $Input --output $Output
if ($LASTEXITCODE -ne 0) {
    throw "Self-play export failed with exit code $LASTEXITCODE."
}

Write-Host ""
Write-Host "Training adapter is not enabled yet." -ForegroundColor Yellow
Write-Host "The exported dataset is ready at: $Output"
Write-Host "Checkpoint: $Checkpoint"
Write-Host ""
Write-Host "This script intentionally stops before AMAGO training until the exact" 
Write-Host "Metamon/SyntheticRLV2 trajectory adapter is wired and validated."
Write-Host "No checkpoint is modified by this command."
