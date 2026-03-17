# Completed: one-round bruteforce/oracle dataset generator (WP2)

Date: 2026-03-17

## What was implemented
- Added `rlohhell/analysis/oracle_dataset.py` with one-round dataset generation.
- Implemented bounded oracle labeling via per-action Monte-Carlo rollouts.
- Recorded required metadata per sample: `round_size`, `phase`, `seat`, `seed`, `opponent_profile`.
- Added JSONL export helper for reproducible dataset artifacts.
- Added tests for deterministic generation under fixed seeds and stable legal labels.

## Validation
- `pytest tests/analysis/test_oracle_dataset.py`

## Notes
- Oracle currently uses rollout-based brute-force evaluation over current legal actions and is bounded by `rollouts_per_action`.
- This completes the WP2 dataset generator portion from `docs/planning/bruteforce-bc-realization-plan.md`.
