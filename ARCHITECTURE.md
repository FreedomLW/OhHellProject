# ARCHITECTURE.md

## High-level architecture

OhHellProject is organized into six layers:

### 1. Game domain layer (`rlohhell/games/ohhell/`)

Implements the Oh Hell card game logic using a 36-card deck (see `RULES.md`
for full game constants and scoring).

| Module | Responsibility |
|--------|----------------|
| `game.py` | `OhHellGame` — orchestrates rounds, players, scoring, round sequence |
| `round.py` | `OhHellRound` — single-round progression, trick management, legal action validation |
| `player.py` | `OhHellPlayer` — player state: hand, bids, tricks won |
| `dealer.py` | `OhHellDealer` — 36-card deck management, dealing |
| `judger.py` | `OhHellJudger` — trick winner determination, round scoring |
| `strategies.py` | Heuristic strategies: `RandomStrategy`, `GreedyStrategy`, `ConservativeStrategy` |
| `utils.py` | Trump suit constant, `determine_winner()`, action space definitions |
| `console.py` | `ConsoleOhHellMatch` — text UI for human play or bot replay |
| `base.py` | `Card` class with suit/rank enums |

### 2. Environment layer (`rlohhell/envs/`)

Wraps game logic into RL-friendly APIs. Two environments exist:

**`OhHellEnv`** (rlcard-style, legacy):
- Observation: **111-dim** float vector
  - Indices 0–50: played cards (one-hot via `card2index`)
  - Indices 51–103: hand cards (shifted +51)
  - Index 104: trump card index
  - Index 105: agent's tricks won
  - Index 106: agent's bid
  - Indices 107–110: all players' tricks won

**`OhHellEnv2`** (Gymnasium, primary):
- Observation: **`108 + 2 + 3×N + N + 1`** dims (127 for N=4 players)
  - 36 hand + 36 current trick + 36 previously played = 108 card slots
  - Trump index (normalized) + hand fullness ratio = 2
  - Bids + tricks won + hand sizes per player = 3×N
  - Current player one-hot = N
  - Round counter ratio = 1
- Action space: **`Discrete(11)`** — 9 card slots + 2 Joker modes (low/high)
- Action masking via `mask_from_state()` for MaskablePPO

### 3. Policy and opponents layer (`rlohhell/policies/`, `rlohhell/heuristics/`, `rlohhell/utils/opponents.py`)

| Component | Purpose |
|-----------|---------|
| `MaskableLstmPolicy` | LSTM-based policy for SB3 with action masking |
| `ExplainableMLPStrategy` | MLP strategy with explicit defaults and bid snapping |
| `ParamVector` / `ParametricHeuristicOpponent` | 30-parameter heuristic bot (HCP values, bonuses, trump weights) |
| `OpponentPool` | Manages opponent sampling for self-play table composition |
| `ModelOpponent` / `ThetaOpponentWrapper` | Adapters for frozen SB3 models and evolutionary bots |

### 4. Training and evaluation layer (`scripts/`, `rlohhell/evo/`)

**Self-play RL** (`scripts/train_maskable_self_play.py`):
- MaskablePPO with vectorized environments (8–32 parallel)
- `SharedPolicyOpponent` for self-play, opponent pool with model snapshots
- Linear entropy annealing, periodic checkpoints, TensorBoard logging

**Evolutionary optimization** (`rlohhell/evo/`):
- `train_evo.py` — CEM or CMA-ES over `ParamVector` parameters
- Common random numbers (CRN) for fair cross-generation comparisons
- `HallOfFame` archiving to prevent regression
- `eval_evo.py` — evaluation of trained theta against baseline + HoF opponents
- Metrics: fitness, points_per_round, win_rate, bid distributions

**Oracle teacher** (`scripts/train_oracle_teacher.py`):
- Fixed oracle-teaching runs or parameter search mode (`--search`)
- Supports play-phase-only labeling with exhaustive card search
- Configurable hand samples per seed and round sizes

### 5. Analysis / oracle layer (`rlohhell/analysis/`)

| Module | Purpose |
|--------|---------|
| `oracle_dataset.py` | `OracleDatasetGenerator` — one-round scenario sampling, exhaustive play-phase search, rollout-based labeling, JSONL export |
| `curriculum_eval.py` | Iterative oracle bootstrap loop: sample → oracle label → train imitation → add to pool → repeat. Also `find_optimal_oracle_params` for grid search |

The oracle targets **isolated single-round tasks** with random round sizes.
Full-game mode is retained only for user-facing play and final evaluation.

### 6. Validation layer (`tests/`)

Tests cover game rules, environment behavior, action masking, evolutionary
optimizers, oracle dataset generation, curriculum evaluation, and training
scripts.

Run with: `pytest`

## Runtime interaction flow (RL episode)

1. `Env.reset()` creates a new `OhHellGame` and initializes round sequence.
2. Game sets up players, dealer, and deals cards for the first round.
3. Agent and opponents act according to legal actions (enforced by action mask).
4. `Game.step(action)` processes transitions — trick/round closure handled internally.
5. `OhHellJudger` computes trick winners and round scores.
6. Env converts game state to observation vector + action mask.
7. Reward = change in agent's score after each completed round.

## Key invariants

- Legal actions enforce bidding constraints and suit-following rules at `round.get_legal_actions()`.
- Round and trick winners update turn order deterministically.
- Scoring matches the formula in `RULES.md` (see `judger.judge_game()`).
- Observation encoding and action masks stay aligned with action decoding.
- Action mask length is always 11 regardless of actual hand size.

## Documentation synchronization

When changing game logic, env interfaces, or scoring:
- update `RULES.md` for rules
- update this file for design impact
- record completed work in `docs/done/CHANGELOG.md`
