# Oh Hell Autoresearch — Agent Instructions

This is an experiment to have the LLM autonomously research the strongest Oh Hell card-game agent.

## Setup

To set up a new experiment run, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar24`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files**: Read these files for full context:
   - `experiments/IDEAS.md` — idea backlog.
   - `experiments/RESULTS.md` — all previous experiment scores.
   - `experiments/THOUGHTS.md` — analysis and observations.
   - `scripts/run_experiment.py` — the experiment runner (read-only). Orchestrates phases.
   - `scripts/evaluate_model.py` — the evaluation harness (read-only). Ground-truth metric.
   - `scripts/generate_bc_data.py` — BC data generation (read-only).
   - `scripts/train_bc.py` — behaviour cloning trainer (read-only).
   - `scripts/train_maskable_self_play.py` — RL self-play trainer (read-only).
4. **Verify environment**: Check that the virtual environment exists and the package is installed:
   ```
   .venv/bin/python -c "import rlohhell; print('OK')"
   ```
5. **Initialize results.tsv**: Create `results.tsv` in the repo root with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs with a **fixed wall-clock budget of 10 minutes** (600 seconds), enforced by `scripts/run_experiment.py`. You launch experiments by writing a JSON config and running:

```
.venv/bin/python scripts/run_experiment.py experiments/configs/<config>.json
```

**What you CAN do:**
- Create or modify JSON configs in `experiments/configs/` — this is your primary lever. Everything is fair game: pipeline structure, budget allocation, hyperparameters, opponent mix.
- Modify training scripts in `scripts/` if you want to add new capabilities (e.g. per-trick reward shaping, new network architectures, curriculum learning).
- Add new strategies or modify `rlohhell/policies/` for network architecture changes.

**What you CANNOT do:**
- Modify game rules: `rlohhell/games/` is read-only (card mechanics, scoring, trick resolution).
- Modify the environment observation/action space: `rlohhell/envs/` is read-only.
- Modify the evaluation harness: `scripts/evaluate_model.py` is read-only — it is the ground truth.
- Install new packages or add dependencies beyond what's in `pyproject.toml`.

**The goal is simple: get the highest score vs baselines.** Since the time budget is fixed at 10 minutes, you don't need to worry about training time — it's always 10 minutes. Everything is fair game within the constraints above.

**The first run**: Your very first run should establish the baseline by running the current best model (`models/rl_chain_v3.zip`) through evaluation only. This gives you a reference score on this machine.

## Config format

Experiment configs are JSON files with ordered phases:

```json
{
  "description": "Short description of what this experiment tries",
  "total_budget_seconds": 600,
  "phases": [
    {
      "name": "data_gen",
      "budget_fraction": 0.20,
      "params": {
        "num_seeds": 2000,
        "round_sizes": "1,2,3,4,5"
      }
    },
    {
      "name": "bc_train",
      "budget_fraction": 0.15,
      "params": {
        "epochs": 30,
        "batch_size": 256,
        "lr": 0.0003,
        "device": "cpu"
      }
    },
    {
      "name": "rl_train",
      "budget_fraction": 0.45,
      "params": {
        "total_timesteps": 500000,
        "n_envs": 16,
        "ent_coef": 0.001,
        "final_ent_coef": 0.0003,
        "checkpoint_freq": 50000,
        "eval_freq": 50000
      }
    },
    {
      "name": "evaluate",
      "budget_fraction": 0.20,
      "params": {
        "episodes_per_opponent": 50,
        "opponents": ["random", "greedy", "conservative", "heuristic"],
        "self_play_episodes": 0
      }
    }
  ]
}
```

Available phases: `data_gen`, `bc_train`, `rl_train`, `evaluate`.
Phases run sequentially. Each phase's model output feeds the next phase automatically.

### Phase parameters

**data_gen** (generate oracle BC dataset):
- `num_seeds`: number of game scenarios (higher = more data, but slower)
- `round_sizes`: comma-separated card counts (e.g. "1,2,3,4,5"). Rounds 7-9 are exponentially slow — avoid unless budget allows.
- `target_seat`: player position (default 0)
- `opponent_model`: path to SB3 checkpoint for opponent pool (optional)

**bc_train** (behaviour cloning):
- `epochs`: training passes (diminishing returns past ~40)
- `batch_size`: default 256
- `lr`: learning rate (default 3e-4)
- `device`: "cpu" or "cuda" (CPU is fast enough for <100K samples)
- `net_arch`: comma-separated layer sizes, e.g. "256,256" (default: SB3 default)

**rl_train** (MaskablePPO self-play):
- `total_timesteps`: main training budget (100K–1M typical)
- `n_envs`: parallel environments (8–32, fewer = faster per-step but fewer samples)
- `ent_coef`: initial entropy coefficient (controls exploration)
- `final_ent_coef`: end entropy (enables linear annealing). Critical: very low values (0.0003–0.00006) work best.
- `checkpoint_freq`: save frequency in timesteps
- `eval_freq`: tournament evaluation frequency
- `resume_from`: path to pre-trained checkpoint (auto-filled from prior phase)
- `use_bruteforce_opponent`: add oracle to opponent pool (slow but informative)

**evaluate** (vs opponent portfolio):
- `episodes_per_opponent`: games per opponent type (50 for quick check, 200+ for reliable signal)
- `opponents`: list of opponent names: random, greedy, conservative, heuristic, bruteforce
- `self_play_episodes`: number of self-play games (0 to skip)

## Output format

After the experiment finishes, results are saved to `experiments/<id>/result.json`. The key metrics are in `experiments/<id>/eval_metrics.json`:

```json
{
  "vs_random": {"win_rate": 0.96, "avg_score": 89.2, "avg_per_round_score": 4.46},
  "vs_greedy": {"win_rate": 0.78, "avg_score": 72.1, "avg_per_round_score": 3.61},
  "vs_heuristic": {"win_rate": 0.62, "avg_score": 61.3, "avg_per_round_score": 3.07}
}
```

Extract the key metric from the result:

```
cat experiments/<id>/eval_metrics.json | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'vs_heuristic score: {d.get(\"vs_heuristic\",{}).get(\"avg_per_round_score\",\"N/A\")}')"
```

## Logging results

When an experiment is done, log it to `results.tsv` in the repo root (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 6 columns:

```
commit	score_vs_heur	wr_vs_heur	status	config	description
```

1. git commit hash (short, 7 chars)
2. avg_per_round_score vs heuristic (e.g. 3.07) — use 0.000 for crashes
3. win_rate vs heuristic (e.g. 0.620) — use 0.000 for crashes
4. status: `keep`, `discard`, or `crash`
5. config filename (e.g. `bc_rl_chain.json`)
6. short text description of what this experiment tried

Example:

```
commit	score_vs_heur	wr_vs_heur	status	config	description
a1b2c3d	2.975	0.591	keep	baseline	baseline: rl_chain_v3 eval only
b2c3d4e	3.120	0.630	keep	bc_rl_v5.json	RL chain v5: ent 0.00015 from v3
c3d4e5f	2.800	0.550	discard	large_net.json	net_arch=[256,256] — worse
d4e5f6g	0.000	0.000	crash	reward_shape.json	per-trick reward OOM
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar24`).

LOOP FOREVER:

1. Look at the current state: read `results.tsv`, review what's been tried, check which ideas remain in `IDEAS.md`.
2. Pick or invent an experiment idea. Write a JSON config in `experiments/configs/`.
3. `git add` the config and `git commit` with a message describing the hypothesis.
4. Run the experiment:
   ```
   .venv/bin/python scripts/run_experiment.py experiments/configs/<config>.json > run.log 2>&1
   ```
   Redirect everything — do NOT let output flood your context.
5. Read out the results:
   ```
   cat experiments/<latest_id>/eval_metrics.json
   ```
   If the file doesn't exist, the run crashed. Run `tail -n 50 run.log` to read the traceback.
6. Record the results in `results.tsv` (NOTE: do not commit results.tsv — leave it untracked by git).
7. If avg_per_round_score vs heuristic **improved** over the best known score, you "advance" the branch:
   - `git add` the experiment outputs and model
   - `git commit` with metrics in the message
   - Update `experiments/RESULTS.md` with the new result
8. If score is equal or worse:
   - `git reset --hard HEAD` to revert the config commit
   - Move on to the next idea
9. Update `IDEAS.md`: mark ideas as done `[x]` or abandoned `[-]`.
10. **Repeat. NEVER STOP.**

**Timeout**: Each experiment takes ~10 minutes (enforced by the runner). If a run exceeds 15 minutes, kill it and treat it as a crash.

**Crashes**: If a run crashes (OOM, bug, timeout), use your judgment: if it's a typo or config error, fix and re-run. If the idea is fundamentally broken, log it as `crash` and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — re-read THOUGHTS.md, try combining previous near-misses, try more radical changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes ~10 minutes, you can run ~6/hour, for a total of ~50 over the duration of an average human sleep. The user then wakes up to experimental results, all completed by you while they slept!

## Current best and known results

| Model | Score vs baselines | Key approach |
|-------|-------------------|-------------|
| bc_hybrid_stage3 | 49.5 | Baseline BC (heuristic bids + oracle play) |
| rl_finetune_v1 | 53.2 | Single-pass RL fine-tune from BC |
| **rl_chain_v3** | **59.7** | **Current best** — 3 RL chain passes with decreasing entropy |

### Key findings (so you don't repeat mistakes)

1. **Oracle bids are poison** — use hybrid labels (heuristic bids + oracle play).
2. **Self-play BC iterations are unreliable** at scale — focus on RL fine-tuning.
3. **RL chaining with decreasing entropy works**: 52 -> 55 -> 56 -> 60 score progression.
4. **Very low entropy is critical**: RHC-AI used 0.00006, our best uses 0.0003.
5. **Mixed opponent BC data didn't help** (score 52.1 vs 52.0 heuristic-only).
6. **More BC seeds > more epochs** (diminishing returns past ~40 epochs).
7. **Round sizes 7-9 are exponentially slow** for bruteforce data gen.
8. **RL chain plateaued at v3** — v4 (ent=0.0002) got 59.1, no improvement.

### Promising unexplored ideas

- Per-trick reward shaping (richer learning signal every card, not just round end)
- Larger network (net_arch=[256,256])
- Smarter bid teacher (position-aware, suit void counting)
- DAgger pipeline (play -> collect failures -> relabel -> retrain)
- RL with baseline opponents in pool (bc3 + rl_v1 + rl_v3)
- Much longer RL training (1M+ steps) from current best

## Tips

- BC training on CPU is very fast (~seconds for 10K samples, ~1 min for 100K).
- RL self-play eats most of the budget. Use it for fine-tuning, not from scratch.
- If RL is killed by timeout, the runner finds the latest checkpoint automatically.
- The runner streams output to console AND to `experiments/<id>/log.txt`.
- Model checkpoints from previous experiments can be reused via `resume_from` in rl_train params.
- Focus on **score** (avg_per_round_score), not win rate. Score is more isolated from opponent strength.
