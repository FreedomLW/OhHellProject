# Reinforcement Learning Project on Oh Hell

![Laptop and cards](img.jpg)

This repository contains a reinforcement-learning environment and agents for the card game **Oh Hell**. The current version focuses on a single 10-card round with support for self-play training using Stable Baselines3 and a text-mode console interface for quick matches.

The repository has been trimmed to exclude large training artifacts and build outputs so that only the source code, tests, and lightweight examples remain.

## Installation
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e .
```

## Playing or observing in the console
You can play a quick text-mode match or watch bots battle by using the `ConsoleOhHellMatch` helper:

```bash
python - <<'PY'
from rlohhell.games.ohhell.console import ConsoleOhHellMatch
from rlohhell.games.ohhell import load_model_strategy

# Example 1: take seat 0 and play against three random bots
ConsoleOhHellMatch(num_players=4, human_player=0).run()

# Example 2: watch four bots with a fixed seed for reproducible logs
ConsoleOhHellMatch(num_players=4, human_player=None, seed=123).run()

# Example 3: load a saved MaskablePPO checkpoint and take seat 0 vs the model
bot = load_model_strategy("runs/some_model/checkpoints/maskable_ppo_100000_steps.zip")
ConsoleOhHellMatch(num_players=4, human_player=0, strategies={1: bot, 2: bot, 3: bot}).run()
PY
```

During bidding you will be prompted for your bid; during play you will see your hand, the legal cards you can choose, and who wins each trick. If you set `record_history=True` (default) you can also inspect the in-memory log after the match.

## Training
Use the self-play helper in `scripts/train_maskable_self_play.py` to train a MaskablePPO agent or resume from a saved checkpoint:

```bash
python scripts/train_maskable_self_play.py --help  # view options
python scripts/train_maskable_self_play.py --total-timesteps 1_000_000 --num-envs 8
python scripts/train_maskable_self_play.py --resume-from runs/maskable_ppo/checkpoints/maskable_ppo_100000_steps.zip
```

Checkpoints and TensorBoard logs will be written under `runs/` (ignored by git). Saved models can be loaded with `load_model_strategy` for console play or evaluation.

For supervised imitation of the heuristic opponent with the custom `StudentPokerPolicy`, use the dedicated pipeline:

```bash
python scripts/train_student_poker_policy.py --episodes 250 --epochs 15 --save-path runs/student_poker_policy.pt
```

## Running tests
```bash
pytest
```

## Credits
Throughout this project we used a variety of references and tools:

- The structure of the game environment - [RLCARD](https://github.com/datamllab/rlcard)
- The idea to train the agent against itself (NFSP) - [NFSP](https://arxiv.org/abs/1603.01121)
- A template project using PPO on the game Big2 - [Big2PPO](https://github.com/henrycharlesworth/big2_PPOalgorithm)
- For an implementation of the PPO algorithm in PyTorch - [SB3](https://github.com/DLR-RM/stable-baselines3)
