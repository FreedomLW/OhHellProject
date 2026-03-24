# CLAUDE.md

## Project

Oh Hell RL research — training the strongest agent for a 4-player 36-card trick-taking game.

## Quick start

```bash
cd OhHellProject
source .venv/bin/activate        # or: .venv/bin/python
pytest                            # run tests
.venv/bin/python scripts/run_experiment.py experiments/configs/bc_small.json
```

## Git rules

- **Never push to `main` directly.** All work happens on feature or experiment branches.
- **Autoresearch branches**: use `autoresearch/<tag>` naming (e.g. `autoresearch/mar24`).
- **Commit only on improvement.** If an experiment doesn't beat the current best, `git reset --hard HEAD` instead of committing.
- **Do not commit** `results.tsv`, `run.log`, or anything in `experiments/*/` (outputs are gitignored).
- **Do commit** config files (`experiments/configs/*.json`), tracking files (`IDEAS.md`, `RESULTS.md`, `THOUGHTS.md`), and model checkpoints that set new records (`models/*.zip`).
- **Commit messages** must include the metric: e.g. `exp07: RL chain v5, score=3.15 vs heuristic (keep)`.
- **Never amend published commits.** Create new commits instead.
- **Never force push** to shared branches.
- **Run `pytest` before committing code changes** (not needed for config-only changes).

## Read-only files (do NOT modify)

- `rlohhell/games/` — game rules, card mechanics, scoring
- `rlohhell/envs/` — environment observation/action space
- `scripts/evaluate_model.py` — evaluation harness (ground truth metric)

## Files you CAN modify

- `experiments/configs/*.json` — experiment configurations (primary lever)
- `experiments/IDEAS.md`, `RESULTS.md`, `THOUGHTS.md` — tracking
- `scripts/train_maskable_self_play.py` — RL training (add features like reward shaping)
- `scripts/train_bc.py` — BC training
- `scripts/generate_bc_data.py` — data generation
- `scripts/run_experiment.py` — experiment orchestration
- `rlohhell/policies/` — neural network architectures

## Key metric

**`avg_per_round_score` vs heuristic** (higher = better). Focus on score, NOT win rate.

Current best: **rl_chain_v3** with score ~59.7 vs baselines.

## Autoresearch mode

Read `experiments/program.md` for the full autonomous experiment loop instructions.
The loop: pick idea -> write config -> commit -> run 10-min experiment -> eval -> keep/revert -> repeat forever.

## Style

- No emojis in code or commit messages.
- No trailing summaries after completing a task.
- Don't add comments, docstrings, or type annotations to code you didn't change.
- Use `.venv/bin/python` (not `python` or `uv run`) to run scripts.
