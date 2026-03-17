# ARCHITECTURE.md

## High-level architecture

OhHellProject is organized into five main layers:

1. **Game domain layer** (`rlohhell/games/ohhell`)
   - Implements cards, dealing, round progression, legal actions, trick resolution, and scoring.
   - Core classes: `OhHellGame`, `OhHellRound`, `OhHellDealer`, `OhHellJudger`, `OhHellPlayer`.

2. **Environment layer** (`rlohhell/envs`)
   - Wraps game logic into RL-friendly APIs.
   - Includes gym-style environment with action masks for MaskablePPO and adapters for observation/action encoding.

3. **Policy and opponents layer** (`rlohhell/policies`, `rlohhell/heuristics`, `rlohhell/utils/opponents.py`)
   - Contains trainable policies and heuristic opponents.
   - Provides opponent pooling/selection to diversify self-play.

4. **Training and evaluation layer** (`scripts/`, `rlohhell/evo`)
   - RL self-play training script(s).
   - Evolutionary optimization workflows and Hall-of-Fame support.

5. **Validation layer** (`tests/`)
   - Tests for game rules, environment behavior, evo optimizers, and scripts.

## Runtime interaction flow (RL episode)
1. Env reset creates a new `OhHellGame`.
2. Game initializes players, dealer, round sequence, and state.
3. Agent/opponents act according to legal actions.
4. Game transitions are processed by `step`, including trick and round closure.
5. Judger computes trick winners and round scores.
6. Env converts resulting state to observation + action mask.

## Key invariants
- Legal actions must enforce bidding and follow-suit constraints.
- Round and trick winners must update turn order deterministically.
- Scoring must match implemented rules in judger/tests.
- Observation encoding and action masks must stay aligned with action decoding.

## Documentation synchronization rule
When changing game logic, env interfaces, or scoring:
- update `RULES.md` for rules
- update `ARCHITECTURE.md` for design impact
- record plan in `docs/planning/` and completed result in `docs/done/`
