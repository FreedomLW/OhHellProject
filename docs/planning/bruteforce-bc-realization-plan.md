# Planning: realization of one-round oracle + behavior cloning pipeline

Date: 2026-03-17
Status: Planned
Owner suggestion: RL/Simulation contributor

## Goal
Build a training pipeline where learning is performed only on **isolated one-round episodes** with random round size. Full game remains only for user play/evaluation.

## Scope
- In scope: one-round scenario generation, oracle labeling, BC training, one-round evaluation, iterative relabeling.
- Out of scope: full-game oracle solving and full-game training objective.

## Work packages

### WP1 — One-round environment/sampler
- Add one-round runner utility (e.g., `rlohhell/analysis/one_round_runner.py`).
- Sample `round_size` randomly (initially from small range, then 1..9).
- Support configurable opponents and fixed seeds.
- Ensure each sample is independent (no carry-over state between rounds).

**Acceptance criteria**
- Deterministic one-round replay from seed.
- Tests verify no cross-round leakage.

### WP2 — One-round oracle dataset generator
- Add module: `rlohhell/analysis/oracle_dataset.py`.
- For each one-round state:
  - enumerate legal actions,
  - run bounded search/rollout,
  - choose best action + confidence margin.
- Store dataset with required metadata: `round_size`, `phase`, `seat`, `seed`, `opponent_profile`.

**Acceptance criteria**
- Reproducible datasets from fixed seed lists.
- Unit tests for stable labels on tiny fixtures.

### WP3 — Behavior cloning trainer
- Add script: `scripts/train_behavior_clone.py`.
- Train masked policy on one-round dataset.
- Log per-phase metrics (`bid` and `play`) and per-round-size metrics.

**Acceptance criteria**
- End-to-end train/validate on sample dataset.
- Report masked accuracy and per-round-size accuracy.

### WP4 — One-round evaluation protocol
- Add evaluation command/report for isolated rounds only:
  - win/trick metrics by `round_size`,
  - bid MAE,
  - robustness across opponent sets,
  - paired fixed-seed comparison.

**Acceptance criteria**
- Single command outputs structured report (CSV/JSON/NPZ).

### WP5 — Iterative relabeling loop (DAgger-style)
- Run current BC policy on one-round tasks.
- Collect hard states.
- Relabel with oracle and retrain.
- Repeat for N cycles.

**Acceptance criteria**
- At least 2 cycles with stable/improved one-round benchmark metrics.

## Milestone sequence
1. WP1 sampler + tests.
2. WP2 oracle data generation on small `round_size`.
3. WP3 BC baseline model.
4. WP4 one-round benchmark report.
5. WP5 iterative relabeling.
6. Optional expansion to all `round_size` in 1..9.

## Deliverables
- One-round runner + oracle dataset tool.
- BC training script and checkpoints.
- One-round evaluation artifacts.
- Iteration logs for relabeling cycles.

## Usage boundary
- Training stack uses one-round mode only.
- Full-game mode is retained for user-facing matches and final integrated comparisons.
