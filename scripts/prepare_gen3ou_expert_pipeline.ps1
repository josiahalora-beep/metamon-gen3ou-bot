param(
    [int]$MinRating = 1400,
    [int]$EliteRating = 1500,
    [int]$SuperEliteRating = 1600,
    [switch]$SkipPublicSelfPlay
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $RepoRoot
try {
    Write-Host "=== Gen 3 OU Expert Dataset Pipeline ===" -ForegroundColor Cyan
    Write-Host "Base human threshold: $MinRating"
    Write-Host "Elite threshold:       $EliteRating"
    Write-Host "Super-elite threshold: $SuperEliteRating"

    python -c "from metamon.data.download import download_parsed_replays; print(download_parsed_replays('gen3ou', force_download=False))"
    if ($LASTEXITCODE -ne 0) { throw "Parsed Gen 3 OU replay preparation failed." }

    python -c "from metamon.data.download import download_teams; print(download_teams('gen3ou', 'hl_05_26', force_download=False))"
    if ($LASTEXITCODE -ne 0) { throw "High-ladder Gen 3 OU team preparation failed." }

    if (-not $SkipPublicSelfPlay) {
        python -c "from metamon.data.download import download_self_play_data; print(download_self_play_data('pac-base', 'gen3ou', force_download=False))"
        if ($LASTEXITCODE -ne 0) { throw "pac-base Gen 3 OU self-play preparation failed." }
        python -c "from metamon.data.download import download_self_play_data; print(download_self_play_data('pac-exploratory', 'gen3ou', force_download=False))"
        if ($LASTEXITCODE -ne 0) { throw "pac-exploratory Gen 3 OU self-play preparation failed." }
    }

    $cacheDir = (python -c "from metamon.config import METAMON_CACHE_DIR; print(METAMON_CACHE_DIR)").Trim()
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($cacheDir)) { throw "METAMON_CACHE_DIR is not configured." }
    $source = Join-Path $cacheDir "parsed-replays"

    $tiers = @(
        @{ Name = "high_rated_1400"; Min = $MinRating },
        @{ Name = "elite_1500"; Min = $EliteRating },
        @{ Name = "super_elite_1600"; Min = $SuperEliteRating }
    )

    foreach ($tier in $tiers) {
        $destination = Join-Path $RepoRoot ("battle_data\expert_human\" + $tier.Name)
        Write-Host "Building $($tier.Name) ..." -ForegroundColor Cyan
        python .\scripts\build_elite_human_replays.py --source $source --destination $destination --min-rating $($tier.Min) --format gen3ou
        if ($LASTEXITCODE -ne 0) { throw "Failed to build $($tier.Name)." }
    }

    $manifest = [ordered]@{
        schema_version = 1
        format = "gen3ou"
        source = $source
        tiers = @(
            [ordered]@{ name = "high_rated_1400"; min_rating = $MinRating; path = "battle_data/expert_human/high_rated_1400/gen3ou" },
            [ordered]@{ name = "elite_1500"; min_rating = $EliteRating; path = "battle_data/expert_human/elite_1500/gen3ou" },
            [ordered]@{ name = "super_elite_1600"; min_rating = $SuperEliteRating; path = "battle_data/expert_human/super_elite_1600/gen3ou" }
        )
        team_set = "$cacheDir\teams\hl_05_26\gen3ou"
        public_self_play = if ($SkipPublicSelfPlay) { @() } else { @("pac-base", "pac-exploratory") }
    }
    $manifest | ConvertTo-Json -Depth 6 | Set-Content -Path .\battle_data\expert_human\manifest.json -Encoding utf8

    Write-Host ""
    Write-Host "EXPERT DATASET READY" -ForegroundColor Green
    Write-Host "Manifest: battle_data\expert_human\manifest.json"
    Write-Host "Training config: battle_data\selfplay_training\gen3ou_expert_bootstrap.yaml"
    Write-Host ""
    Write-Host "Recommended first training pass:" -ForegroundColor Yellow
    Write-Host ".\scripts\train_from_selfplay.ps1 -SelfPlayRoot '.\battle_data\curriculum\iter_0001\trajectories' -Checkpoint 48 -SaveDir '.\cache\selfplay_finetunes' -RunName 'synthetic_rl_v2_curriculum_0001' -Epochs 2 -StepsPerEpoch 250 -BatchSizePerGpu 2 -GradAccum 1 -DloaderWorkers 0"
} finally {
    Pop-Location
}
