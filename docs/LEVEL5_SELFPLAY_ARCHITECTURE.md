# Level 5 Gen 3 OU Self-Play Architecture

Status: design contract for `feature/anti-throw-safety`.

This document is the source-of-truth design for the next-generation local Gen 3 OU training loop. Do not replace the current working self-play/training pipeline with guessed AMAGO arguments or guessed checkpoint filenames.

## Objective

Build a champion-gated historical league training loop around the existing SyntheticRLV2 + AMAGO pipeline:

```text
Immutable Checkpoint 48
        |
        v
Current Champion <---- promotion gate <---- Candidate
        |                                      ^
        |                                      |
        +--> historical opponent pool          |
        |                                      |
        +--> diverse self-play --> trajectories --> AMAGO fine-tune
                                               |
                                               v
                                      held-out evaluation
```

## Non-negotiable rules

1. Checkpoint 48 is an immutable baseline/anchor and must remain available forever.
2. Training teams and evaluation teams are strictly separated.
3. A candidate never automatically becomes champion merely because training completed.
4. Promotion requires the evaluation gates to pass.
5. Historical champions are retained and sampled as opponents; the pool must have a configurable maximum size.
6. Current, historical, and anchor trajectory data are separate tiers.
7. The active champion is represented by `champion_manifest.json`, not a bare checkpoint number.
8. AMAGO's actual run/checkpoint API must be used. Do not invent an `-OutputFile` interface.
9. Any checkpoint path/identifier written to the manifest must be verified to exist and load before promotion.
10. Local Showdown usernames must be ASCII-safe and unique per worker/side, as implemented in the current local curriculum runner.

## Actual AMAGO integration contract

The existing `scripts/train_from_selfplay.ps1` accepts:

- `-SelfPlayRoot`
- `-Checkpoint`
- `-SaveDir`
- `-RunName`
- `-Epochs`
- `-StepsPerEpoch`
- `-BatchSizePerGpu`
- `-GradAccum`
- `-DloaderWorkers`
- `-PrevRunDir`
- `-PrevRunName`
- `-PrevCheckpoint`

For the initial candidate, use `-Checkpoint` with the numeric SyntheticRLV2 base checkpoint. For subsequent candidates, resume from the prior AMAGO run with `-PrevRunDir`, `-PrevRunName`, and `-PrevCheckpoint`.

The current training wrapper writes checkpoints under:

```text
<SaveDir>/<RunName>/ckpts/policy_weights/policy_epoch_<N>.pt
```

where `<N>` is the training epoch. The wrapper also writes `dataset_config.yaml` inside the run directory.

## Champion manifest

The canonical pointer is `champion_manifest.json`. It must contain enough information to reconstruct the exact AMAGO source without guessing.

Initial anchor example:

```json
{
  "schema_version": 1,
  "iteration": 0,
  "champion_id": "checkpoint_48",
  "source_type": "base_checkpoint",
  "base_checkpoint": 48,
  "run_dir": null,
  "run_name": null,
  "checkpoint": null,
  "checkpoint_path": null,
  "overall_win_rate": 1.0,
  "win_rate_vs_anchor": 1.0,
  "win_rate_vs_champion": 1.0,
  "evaluation_battles": 0
}
```

A promoted finetuned candidate should instead record its real run directory/name/checkpoint and its verified local checkpoint path, for example:

```json
{
  "schema_version": 1,
  "iteration": 1,
  "champion_id": "cand_iter_0001",
  "source_type": "finetuned_run",
  "base_checkpoint": null,
  "run_dir": "./cache/selfplay_finetunes",
  "run_name": "cand_iter_0001",
  "checkpoint": 2,
  "checkpoint_path": "./cache/selfplay_finetunes/cand_iter_0001/ckpts/policy_weights/policy_epoch_2.pt",
  "overall_win_rate": 0.58,
  "win_rate_vs_anchor": 0.50,
  "win_rate_vs_champion": 0.55,
  "evaluation_battles": 100
}
```

The manifest is metadata only. Model binaries remain local and are intentionally ignored by Git.

## Anchor handling

Checkpoint 48 is normally a local downloaded/generated model and is ignored by Git (`*.pt`, `*.pth`, `*.ckpt`, and `*.safetensors`). Therefore the repository cannot itself guarantee where the binary exists on a user's machine.

The Level 5 bootstrap must:

1. Search known local Metamon cache/run/checkpoint locations for the SyntheticRLV2 checkpoint 48.
2. Validate the discovered file before using it.
3. Copy it to the configured immutable local archive location as `anchor_48.pt` or an equivalent verified path.
4. Record the discovered source and archived path in the manifest/metadata.
5. Fail loudly if checkpoint 48 cannot be found instead of inventing a path.

## Trajectory tiers

The rolling replay layout is:

```text
battle_data/local_fast/trajectories/
    current/
        acceptor/gen3ou/
        challenger/gen3ou/
    history/
        ...
    anchor/
        ...
```

Fresh experience belongs in `current`. Previous champion/historical experience belongs in `history`. Anchor/baseline experience belongs in `anchor`.

The training dataset should expose a controlled mixture rather than deleting the previous experience. The initial target mixture is:

- 60% current self-play
- 30% historical self-play
- 10% anchor/base experience

These percentages are configuration targets, not permission to bypass the real AMAGO dataset format. The implementation must translate them into the actual Metamon dataset configuration supported by the installed version.

## Experience generation

`scripts/run_curriculum_battles.py` is responsible for:

1. Reading and validating `champion_manifest.json`.
2. Resolving the active champion through the real AMAGO run/checkpoint interface.
3. Building the historical opponent pool from verified checkpoints.
4. Sampling opponents according to a configurable policy.
5. Using the existing local accelerated Showdown/self-play machinery rather than reimplementing the battle environment.
6. Writing fresh AMAGO trajectories under `trajectories/current`.
7. Preserving historical and anchor trajectory tiers.
8. Using unique ASCII-safe usernames per worker/side.
9. Returning nonzero on any worker or trajectory failure.

## Evaluation gauntlet

`scripts/evaluate_and_promote.py` is responsible for evaluation only. Evaluation must use the held-out team file and must never feed those teams into the training trajectory pool.

Default gates:

| Gate | Requirement |
|---|---:|
| Overall progress | >= 55% |
| Anchor protection | >= 45% vs Checkpoint 48 |
| Champion matchup | >= 52% vs current champion |
| Evaluation size | >= 100 total battles |

The evaluator should expose configurable counts rather than hard-code the 40/30/30 split. Suggested defaults:

- 40 vs current champion
- 30 vs anchor
- 30 vs sampled historical opponents

The evaluator must print a machine-readable summary containing total wins/losses, per-opponent records, and each gate result.

Exit codes:

- `0`: every promotion gate passed; candidate may be promoted.
- `1`: evaluation completed but one or more gates failed; retain the current champion.
- `2+`: evaluator failed, checkpoint could not load, teams were invalid, or the battle infrastructure crashed. The curriculum loop must stop rather than silently reject a broken run.

## Promotion transaction

Promotion must be atomic from the orchestrator's perspective:

1. Candidate training completes.
2. Candidate checkpoint is verified.
3. Evaluation completes.
4. All gates pass.
5. Candidate is copied into the historical archive.
6. Champion manifest is updated atomically.
7. Only then is the candidate considered the new champion.

If any step before 6 fails, the previous champion remains active.

## Historical pool

The historical pool should retain:

- immutable Checkpoint 48
- recent champions
- optionally a sparse sample of older champions

The pool must have configurable retention, for example `MaxHistoricalOpponents = 12`, to prevent evaluation and rollout cost from growing without bound during an infinite curriculum.

Sampling should be diverse rather than always choosing the newest model.

## Hardware starting point

For the current i7-950 / 18 GB RAM / GTX 980 Ti machine, the initial benchmark configuration is:

```text
BattleWorkers       = 2
SimulatorProcesses  = 4
```

Do not jump directly to four Python battle workers. Benchmark 2, 3, and 4 workers after the handshake/runtime is stable and select based on measured battles/minute and memory/CPU pressure.

GPU training remains a single AMAGO process at a time.

## Implementation order

1. Locate and validate the actual local SyntheticRLV2 checkpoint 48.
2. Bootstrap `champion_manifest.json`.
3. Implement `run_curriculum_battles.py` on top of the working local self-play runner.
4. Implement `evaluate_and_promote.py` with the gates above.
5. Add replay-tier dataset configuration using the actual AMAGO/Metamon schema.
6. Replace/upgrade the current continuous runner only after both integration scripts pass standalone smoke tests.
7. Run a 20-battle / 2-worker / 4-simulator benchmark.
8. Run a 100-battle candidate/evaluation cycle.
9. Only then enable an infinite curriculum loop.

## Current known-good components

The following existing components are the foundation and should be reused:

- `scripts/play_local_selfplay_synthetic.py`
- `scripts/play_local_selfplay_synthetic_v2.py`
- `scripts/play_local_curriculum.py`
- `scripts/run_fast_local_synthetic.ps1`
- `scripts/train_from_selfplay.ps1`

The current local curriculum runner already contains the ASCII-safe Showdown username fix needed for parallel challenge handshakes.
