# AGENTS.md

## Project goal

Train the strongest possible agent for **Oh Hell** ("Студенческий покер") —
a 4-player 36-card trick-taking game with bidding.

## Autonomous mode

For fully autonomous experiment iteration, read and follow `experiments/program.md`.
That file contains the complete autoresearch loop: setup, config format, experiment loop,
logging, git workflow, and all known results/findings.

## Key metric

**`avg_per_round_score` vs heuristic** (higher = better). Focus on score, NOT win rate.

## Key commands

| Task | Command |
|------|---------|
| Run experiment | `.venv/bin/python scripts/run_experiment.py <config.json>` |
| Generate BC data | `.venv/bin/python scripts/generate_bc_data.py --help` |
| Train BC | `.venv/bin/python scripts/train_bc.py --help` |
| RL self-play | `.venv/bin/python scripts/train_maskable_self_play.py --help` |
| Evaluate | `.venv/bin/python scripts/evaluate_model.py --model X --episodes 200 --opponents random,greedy,conservative,heuristic` |
| Tests | `pytest` |

## Baseline models

| Model | Score vs baselines (1000g) | Notes |
|-------|---------------------------|-------|
| `models/bc_hybrid_stage3.zip` | 49.5 | BC baseline |
| `models/rl_finetune_v1.zip` | 53.2 | Single RL fine-tune |
| `models/rl_chain_v3.zip` | 59.7 | Current best (3x RL chain) |

## Tracking files

| File | Purpose |
|------|---------|
| `experiments/program.md` | Full autoresearch agent instructions |
| `experiments/IDEAS.md` | What to try next |
| `experiments/RESULTS.md` | All experiment scores |
| `experiments/THOUGHTS.md` | Observations and analysis |
| `results.tsv` | Machine-readable experiment log (gitignored) |

## Rules

See `CLAUDE.md` for git rules, read-only boundaries, and style guidelines.
