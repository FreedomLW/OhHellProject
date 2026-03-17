# Completed: revised research + planning for one-round BC strategy

Date: 2026-03-17

## What was completed
- Reworked research conclusions to remove full-game training focus.
- Defined the core training approach as isolated **one-round** tasks with random round size.
- Updated implementation planning to reflect one-round sampler, one-round oracle labeling, one-round evaluation, and iterative relabeling.
- Kept full-game usage only as user-facing/evaluation boundary.

## Artifacts
- `docs/research/bruteforce-eval-bc-feasibility.md`
- `docs/planning/bruteforce-bc-realization-plan.md`

## Notes
- This revision aligns with the requirement: rounds are independent for training.
- Full game remains available, but not as the main optimization/data-collection target.
