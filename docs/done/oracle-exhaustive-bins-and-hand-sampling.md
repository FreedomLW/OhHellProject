# Completed: exhaustive play-phase oracle with hand sampling

Date: 2026-03-17
Owner: Codex agent

## What was implemented
- `OracleDatasetGenerator` now supports:
  - `hand_samples_per_seed`: number of different hand deals sampled per seed,
  - `play_phase_only`: skip oracle labels in bidding and focus on card-play decisions.
- Added exhaustive target-player play search to maximize final tricks won (bins) for each candidate card.
- Kept rollout-based evaluation for non play-only mode.
- Updated curriculum pipeline and `scripts/train_oracle_teacher.py` to accept and pass new options:
  - `--hand-samples-per-seed`
  - `--include-bid-phase`
- Updated README and tests for new behavior.

## Validation
- `PYTHONPATH=. pytest`
- `PYTHONPATH=. python scripts/train_oracle_teacher.py --help`

## Note
- The exhaustive branching is applied to target-player play decisions; opponent turns are still resolved by configured opponent strategy to keep computation tractable.
