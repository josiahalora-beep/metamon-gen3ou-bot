# Accelerated local Gen 3 OU research

This project now has an isolated local-battle path so the live Showdown ladder path remains unchanged.

## Why this matters

The `hsahovic/Pokemon-Showdown` fork is specifically modified for RL workloads: its README describes removal of battle delays/throttling and between-game rate limiting, plus faster local matchmaking. The upstream Pokémon Showdown project also exposes `BattleStream` for direct simulator use, and `poke-env` recommends a local `--no-security` server for training/development.

The current implementation uses the accelerated fork over the normal local WebSocket path. It does **not** render client animations; animation rendering is a client concern and does not need to be present for server-side RL simulation.

## One-command run

From the repository root in PowerShell:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_fast_local_synthetic.ps1 `
  -Battles 100 `
  -SimulatorProcesses 4
```

The script:

1. Clones `hsahovic/Pokemon-Showdown` into `server\pokemon-showdown-fast` when needed.
2. Installs its Node dependencies if needed.
3. Configures `exports.simulatorprocesses` to the requested process count.
4. Starts Showdown with `--no-security`.
5. Points the existing SyntheticRLV2 runner at `127.0.0.1:8000`.
6. Runs the existing Gen 3 OU battle AI, logger, and analysis path.
7. Stops the local server when the run finishes.

## Important current limitation

The production runner deliberately uses one environment and one battle at a time because the current 200M-parameter SyntheticRLV2 setup is CPU-only on the present machine. The `SimulatorProcesses` setting therefore provides server capacity but should not be interpreted as a 4x end-to-end speedup yet.

The next major throughput step would be an experimental local `BattleStream` backend. Upstream Showdown documents `BattleStream` as a direct simulator interface, and a 2026 `poke-env` issue specifically discusses it as a way to avoid WebSocket/server reset and matchmaking overhead for local training. That is a separate transport path from the live ladder code and is the appropriate follow-on for very large self-play generation.

## Research boundary

Use this accelerated path for local evaluation, regression testing, self-play data collection, and future fine-tuning. Keep the live ladder runner on its existing public configuration.
