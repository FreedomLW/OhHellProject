# Completed: iterative oracle bootstrapping pipeline

Date: 2026-03-17
Owner: Codex agent

## What was implemented
- Reworked `rlohhell/analysis/curriculum_eval.py` into an explicit iterative loop:
  1. sample one-round scenarios from fixed seeds/hand sizes,
  2. collect oracle actions versus current teacher pool,
  3. train student imitation policy on oracle actions,
  4. add trained student into teacher pool,
  5. repeat for N iterations.
- Added a simple table-based `ImitationStrategy` trainer (`train_imitation`) for oracle labels.
- Added support in `OracleDatasetGenerator` for custom `opponent_factory`, so oracle collection can run against predefined opponents and then newly taught strategies.
- Updated CLI `scripts/evaluate_oracle_curriculum.py` to execute the iterative oracle->teach cycles.

## Why
This matches the requested curriculum:
- start vs predefined opponents,
- teach algorithm from oracle,
- collect new oracle vs taught algorithm,
- teach next algorithm,
- iterate.

## Validation
- `PYTHONPATH=. pytest`
- `PYTHONPATH=. python scripts/evaluate_oracle_curriculum.py --help`

## Notes
- Current teaching implementation is a deterministic lookup-table imitation baseline to keep the loop reproducible and testable.
- It can be replaced later with a full neural BC trainer while keeping the same cycle structure.
