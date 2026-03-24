# Completed work log

## 2026-03-17: Oracle curriculum evaluation pipeline

- Implemented iterative oracle bootstrap loop in `rlohhell/analysis/curriculum_eval.py`
  (sample → oracle label → train imitation → add to pool → repeat)
- Added `scripts/evaluate_oracle_curriculum.py` CLI
- Added table-based `ImitationStrategy` trainer for oracle labels
- Validation: `pytest tests/analysis/`

## 2026-03-17: Exhaustive play-phase oracle + hand sampling

- Added `hand_samples_per_seed` and `play_phase_only` options to `OracleDatasetGenerator`
- Exhaustive target-player card search maximizing tricks/bins won
- Wired into `scripts/train_oracle_teacher.py` via `--hand-samples-per-seed` and
  `--include-bid-phase`
- Validation: `pytest`

## 2026-03-17: Oracle teacher CLI simplification

- Default to single seed (`--seed 0 --num-seeds 1`)
- `--seeds` available as optional override for advanced use

## 2026-03-17: Oracle teacher script + parameter search

- Added `find_optimal_oracle_params` and `OracleParamSearchResult` in
  `rlohhell/analysis/curriculum_eval.py`
- Added `scripts/train_oracle_teacher.py` with fixed-run and `--search` modes
- Validation: `pytest tests/analysis/test_oracle_param_search.py tests/scripts/test_train_oracle_teacher.py`

## 2026-03-17: One-round oracle dataset generator (BC plan WP2)

- Added `rlohhell/analysis/oracle_dataset.py` with one-round dataset generation
- Bounded oracle labeling via per-action Monte-Carlo rollouts
- JSONL export with metadata: `round_size`, `phase`, `seat`, `seed`, `opponent_profile`
- Validation: `pytest tests/analysis/test_oracle_dataset.py`

## 2026-03-17: One-round BC strategy research + planning

- Established isolated one-round training as the core approach
- Full game retained for user-facing play/evaluation only
- See `docs/research/bruteforce-eval-bc-feasibility.md`

## 2026-03-17: Explainable MLP strategy refactor

- Replaced ad-hoc placeholders with explicit `SimpleNamespace` defaults
- Added closest-legal-bid selection helper
- Hardened bidding against empty legal action lists

## 2026-03-17: Harness packaging fix

- Reworked packaging to PEP 621 compliant `pyproject.toml`
- Fixed editable install crash from mixed `setup.py`/`pyproject.toml` metadata

## 2026-03-17: Harness adoption

- Added `AGENTS.md`, `RULES.md`, `ARCHITECTURE.md`
- Established `docs/` lifecycle directories (planning, wip, done, research)
