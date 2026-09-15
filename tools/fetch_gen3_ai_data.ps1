$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
$dataDir = Join-Path $root 'battle_ai\data'
New-Item -ItemType Directory -Force -Path $dataDir | Out-Null

$smogonUrl = 'https://data.pkmn.cc/sets/gen3ou.json'
$showdownLearnsetsUrl = 'https://raw.githubusercontent.com/smogon/pokemon-showdown/master/data/mods/gen3rs/learnsets.ts'

$smogonOut = Join-Path $dataDir 'gen3ou.json'
$learnsetsOut = Join-Path $dataDir 'gen3rs_learnsets.ts'

Write-Host 'Downloading curated Smogon Gen 3 OU set corpus...'
Invoke-WebRequest -Uri $smogonUrl -OutFile $smogonOut -UseBasicParsing

Write-Host 'Downloading Pokémon Showdown Gen 3 RS learnsets...'
Invoke-WebRequest -Uri $showdownLearnsetsUrl -OutFile $learnsetsOut -UseBasicParsing

$smogonSize = (Get-Item $smogonOut).Length
$learnsetsSize = (Get-Item $learnsetsOut).Length

if ($smogonSize -lt 10000) { throw "Smogon dataset download looks invalid: $smogonSize bytes" }
if ($learnsetsSize -lt 10000) { throw "Showdown learnset download looks invalid: $learnsetsSize bytes" }

Write-Host "Gen 3 AI data ready:"
Write-Host "  $smogonOut ($smogonSize bytes)"
Write-Host "  $learnsetsOut ($learnsetsSize bytes)"
