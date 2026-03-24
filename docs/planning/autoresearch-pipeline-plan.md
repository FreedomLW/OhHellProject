# Plan: Autoresearch-style experiment pipeline for Oh Hell

Date: 2026-03-22
Status: Planned

## Goal

Build an automated experimentation pipeline (inspired by
[Karpathy's autoresearch](https://github.com/karpathy/autoresearch)) that lets
an AI agent autonomously run 10-minute training experiments, log results via git
commits, and iterate on ideas to find the best Oh Hell playing agent.

## First priority: create the strongest baseline bot

Before scaling up experiments, we need a solid **first bot** that proves the
pipeline works end-to-end. The approach:

1. **Generate BC data** from `BruteforceStrategy` (perfect-info oracle) on small
   round sizes (1-5) where exhaustive search is fast.
2. **Train a MaskablePPO** policy via supervised cross-entropy (behavior cloning).
3. **Evaluate** against all heuristic opponents to establish baseline metrics.
4. **Iterate**: tune seeds/epochs/lr/round_sizes to maximize win_rate vs heuristic.

This gives us a neural model that imitates the oracle's play with partial
information. Everything after this (RL fine-tuning, DAgger, curriculum) builds
on this baseline.

## Pipeline architecture

### Core loop (autoresearch-style)

```
1. Pick idea from IDEAS.md
2. Create/modify experiment config
3. Run: python scripts/run_experiment.py <config>   (10 min hard kill)
4. Read result.json
5. If improved → git commit with metrics in message
6. If worse → git reset, log as "discard"
7. Append result to RESULTS.md
8. Repeat
```

Git commits ARE the experiment log. Each commit message contains the key metrics
so `git log --oneline` gives a summary of all improvements.

### Experiment runner — `scripts/run_experiment.py`

Enforces 10-minute wall-clock budget via subprocess timeouts.

Phases (composable, declared in config JSON):

| Phase | Subprocess | Produces |
|-------|-----------|----------|
| `data_gen` | `scripts/generate_bc_data.py` | `bc_data.npz` |
| `bc_train` | `scripts/train_bc.py` | `bc_model.zip` |
| `rl_train` | `scripts/train_maskable_self_play.py` | `final_model.zip` |
| `evaluate` | `scripts/evaluate_model.py` | `eval_metrics.json` |

Shared state flows between phases: data_gen → bc_train → rl_train → evaluate.

Output per experiment:
```
experiments/<id>/
  config.json       # input params
  result.json       # metrics + timing + status
  log.txt           # stdout/stderr from all phases
  model/            # trained checkpoint(s)
```

### Evaluation harness — `scripts/evaluate_model.py`

Plays N full games against each opponent:

| Opponent | Episodes | Expected speed |
|----------|----------|---------------|
| Random | 50 | ~10s |
| Greedy | 50 | ~10s |
| Conservative | 50 | ~10s |
| Heuristic | 50 | ~10s |
| Bruteforce | 10 | ~60s (optional) |
| Self-play | 50 | ~15s |

Output: JSON with `win_rate`, `avg_score`, `avg_per_round_score` per opponent.

### Config presets — `experiments/configs/`

| Config | Description | Budget split |
|--------|-------------|-------------|
| `bc_small.json` | 50 seeds, rounds 1-3, 20 epochs | 15% gen / 15% train / 70% eval |
| `bc_medium.json` | 200 seeds, rounds 1-6, 50 epochs | 40% gen / 30% train / 30% eval |
| `bc_large.json` | 500 seeds, rounds 1-5, 80 epochs | 45% gen / 30% train / 25% eval |
| `rl_finetune.json` | RL 200K steps from checkpoint | 75% rl / 25% eval |
| `bc_then_rl.json` | 100 seeds BC + 150K RL steps | 20/15/45/20% |

### Tracking files

- `experiments/program.md` — agent loop instructions (like autoresearch program.md)
- `experiments/IDEAS.md` — hypothesis tracker
- `experiments/RESULTS.md` — results table

## Files to create

| File | Purpose |
|------|---------|
| `scripts/run_experiment.py` | Experiment runner (~200 lines) |
| `scripts/evaluate_model.py` | Evaluation harness (~120 lines) |
| `experiments/program.md` | Agent instructions |
| `experiments/IDEAS.md` | Ideas tracker |
| `experiments/RESULTS.md` | Results table |
| `experiments/configs/bc_small.json` | Quick config |
| `experiments/configs/bc_medium.json` | Medium config |
| `experiments/configs/bc_large.json` | Max BC config |
| `experiments/configs/rl_finetune.json` | RL config |
| `experiments/configs/bc_then_rl.json` | Combined config |
| `tests/scripts/test_run_experiment.py` | Runner smoke test |
| `tests/scripts/test_evaluate_model.py` | Eval smoke test |

## Files to modify

| File | Change |
|------|--------|
| `.gitignore` | Add experiment output dirs (keep configs + tracking files) |

## Verification

```bash
# Quick end-to-end test
python scripts/run_experiment.py experiments/configs/bc_small.json
cat experiments/*/result.json | python -m json.tool

# Standalone evaluation
python scripts/evaluate_model.py --model experiments/*/model/bc_model.zip --episodes 10

# Tests
pytest tests/scripts/ -v
```

## Key reference files

- `scripts/generate_bc_data.py` — BC data generation CLI
- `scripts/train_bc.py` — BC training CLI
- `scripts/train_maskable_self_play.py` — RL training with `--resume-from`
- `rlohhell/heuristics/bruteforce_agent.py` — perfect-info oracle
- `rlohhell/utils/opponents.py` — opponent pool and adapters
- `rlohhell/envs/ohhell.py:160-398` — `OhHellEnv2` environment
