# Research: one-round training strategy (random hand size) + full-game usage boundary

Date: 2026-03-17

## Decision after review
Your correction is valid: **do not base training on full-game optimization**.  
For training, use **isolated single-round tasks** with random number of cards.  
Keep **full-game mode only for user-facing play/evaluation**.

## 1) Current repository capabilities relevant to this direction

### What helps one-round training already
- `OhHellGame` already supports round state progression, legal actions, bidding/play transitions, and scoring, so we can sample and stop at single-round boundaries.
- `OhHellEnv`/`OhHellEnv2` already provide action masks, which is exactly what BC needs to avoid illegal actions.
- `OpponentPool` already allows plugging policies with different opponents and seat assignments.

### What should not be the training target
- Full multi-round game optimization should not be the core data-generation objective due to combinatorial complexity and mixed objectives across rounds.

## 2) Recommended training formulation (strictly one round)

### Round sampler
For each generated sample:
1. Draw random seed.
2. Draw random round size `k` (e.g. `k ∈ [1, 9]`, or a curriculum subset first).
3. Build a one-round scenario with fixed opponents and seat.
4. Run oracle search for bidding and play inside this round only.
5. Save demonstration transitions.

### Oracle target
- Target is **best action for this round context**, not global multi-round outcome.
- Reward target inside the oracle can be:
  - tricks won (for trick-max curriculum), or
  - round score consistency with bid, depending on phase.

### Data format
Collect for each decision point:
- observation,
- legal action mask,
- chosen action,
- action value margin,
- metadata: `round_size`, phase (`bid`/`play`), seat, opponent profile, seed.

## 3) Evaluation algorithms to prioritize

1. **Round-only benchmark suite** (primary): fixed-seed paired comparison on isolated rounds with random `k`.
2. **Per-round-size breakdown**: performance curves for each `k` from 1 to 9.
3. **Bid calibration**: MAE between bid and realized tricks in one-round episodes.
4. **Play quality**: trick-win delta vs baseline opponents conditioned by `k` and seat.
5. **Cross-opponent robustness**: random/greedy/heuristic/param/evo opponents under the same one-round seeds.

## 4) Iterative BC loop (one-round only)

1. Generate one-round oracle data.
2. Train BC model with masked policy head.
3. Run BC in one-round matches; log failure states.
4. Relabel those states with oracle (DAgger-style).
5. Retrain and repeat.

This keeps the learning signal clean and aligned with your requirement that rounds are isolated from each other.

## 5) Boundary between training and user gameplay

- **Training:** isolated one-round environments with random card count.
- **User-facing gameplay / final demonstration:** full game remains available as an integration mode.

So the architecture should support both, but optimization and dataset collection should be focused on one-round scenarios.
