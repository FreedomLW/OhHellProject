# Completed: simplify oracle teacher seed configuration

Date: 2026-03-17
Owner: Codex agent

## What changed
- Simplified `scripts/train_oracle_teacher.py` to default to a single seed (`--seed 0 --num-seeds 1`) instead of requiring multiple seed lists.
- Added optional `--seeds` only as an override when explicit scenario control is needed.
- Reduced default search grid to lighter settings (`iterations: 2,3` and `rollouts: 2,4`) for faster usage.
- Updated README example to use the simplified flags.

## Why
- The user requested simplification and asked why many seeds are necessary.
- A single-seed default is easier to run and understand, while still allowing multiple seeds for stability checks when needed.

## Validation
- `PYTHONPATH=. pytest`
- `PYTHONPATH=. python scripts/train_oracle_teacher.py --help`
