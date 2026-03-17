# Completed: oracle teacher script with parameter search

Date: 2026-03-17
Owner: Codex agent

## What was implemented
- Added `find_optimal_oracle_params` and `OracleParamSearchResult` in `rlohhell/analysis/curriculum_eval.py`.
- Added executable script `scripts/train_oracle_teacher.py` that supports:
  - fixed oracle-teaching runs,
  - parameter search mode (`--search`) over candidate iterations and rollouts,
  - optional JSON output file.
- Exported the new search helper via `rlohhell/analysis/__init__.py`.
- Added tests for parameter search and script CLI behavior.

## Validation
- `pytest tests/analysis/test_oracle_param_search.py tests/scripts/test_train_oracle_teacher.py`
- `python scripts/train_oracle_teacher.py --help`

## Notes
- Search objective is final-stage `against_teacher_pool.avg_payoff` with deterministic tie-breakers on win rate and then lower compute budget.
