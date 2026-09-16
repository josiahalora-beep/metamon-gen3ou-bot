param(
    [int]$MinRating = 1400,
    [switch]$SkipPublicSelfPlay
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $RepoRoot
try {
    Write-Host "Preparing Gen 3 OU expert bootstrap data..." -ForegroundColor Cyan

    python -c "from metamon.data.download import download_parsed_replays; print(download_parsed_replays('gen3ou', force_download=False))"
    if ($LASTEXITCODE -ne 0) { throw "Failed to prepare parsed Gen 3 OU human replays." }

    python -c "from metamon.data.download import download_teams; print(download_teams('gen3ou', 'hl_05_26', force_download=False))"
    if ($LASTEXITCODE -ne 0) { throw "Failed to prepare the high-ladder Gen 3 OU team set." }

    if (-not $SkipPublicSelfPlay) {
        python -c "from metamon.data.download import download_self_play_data; print(download_self_play_data('pac-base', 'gen3ou', force_download=False))"
        if ($LASTEXITCODE -ne 0) { throw "Failed to prepare pac-base Gen 3 OU self-play data." }

        python -c "from metamon.data.download import download_self_play_data; print(download_self_play_data('pac-exploratory', 'gen3ou', force_download=False))"
        if ($LASTEXITCODE -ne 0) { throw "Failed to prepare pac-exploratory Gen 3 OU self-play data." }
    }

    $cacheDir = python -c "from metamon.config import METAMON_CACHE_DIR; print(METAMON_CACHE_DIR)"
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($cacheDir)) {
        throw "METAMON_CACHE_DIR is not configured."
    }
    $cacheDir = $cacheDir.Trim()

    $source = Join-Path $cacheDir "parsed-replays"
    $destination = Join-Path $RepoRoot "battle_data\expert_human"

    Write-Host "Building hard-linked >= $MinRating Gen 3 OU human subset..." -ForegroundColor Cyan
    python .\scripts\build_elite_human_replays.py `
        --source $source `
        --destination $destination `
        --min-rating $MinRating `
        --format gen3ou
    if ($LASTEXITCODE -ne 0) { throw "Failed to build the elite human replay subset." }

    Write-Host "" 
    Write-Host "Expert bootstrap data is ready." -ForegroundColor Green
    Write-Host "Human subset: $destination\gen3ou"
    Write-Host "High-ladder teams: $cacheDir\teams\hl_05_26\gen3ou"
    if (-not $SkipPublicSelfPlay) {
        Write-Host "Public self-play: $cacheDir\self-play\pac-base\gen3ou.tar"
        Write-Host "Public self-play: $cacheDir\self-play\pac-exploratory\gen3ou.tar"
    }
    Write-Host "Training config: battle_data\selfplay_training\gen3ou_expert_bootstrap.yaml"
} finally {
    Pop-Location
}
