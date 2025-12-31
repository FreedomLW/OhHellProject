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
Use the self-play helper in `scripts/train_maskable_self_play.py` to train a MaskablePPO agent:

```bash
python scripts/train_maskable_self_play.py --help  # view options
python scripts/train_maskable_self_play.py --total-timesteps 1_000_000 --num-envs 8
```

Checkpoints and TensorBoard logs will be written under `runs/` (ignored by git). Saved models can be loaded with `load_model_strategy` for console play or evaluation.

## Эволюционное обучение параметризованного бота
Запустите эволюционный поиск по весам параметризованного бота:

```bash
pip install -e .[evo]
rlohhell-train-evo --algo cem --generations 50 --pop-size 64 --batch-seeds 512 --jobs 8 --log-dir runs/evo
rlohhell-eval-evo --theta-json runs/evo/best_theta.json --episodes 512 --jobs 8 --log-dir runs/evo_eval
```

- **CRN-сиды:** и `rlohhell-train-evo`, и `rlohhell-eval-evo` используют фиксированные наборы сидов (`fixed_seeds`) для батчей оценок. Это реализует common random numbers (CRN), чтобы сравнения между поколениями были честными: каждый кандидат играeт на тех же сценариях, а `--seed` сдвигает базовое значение.
- **Hall-of-Fame:** во время обучения лучшие особи архивируются в `HallOfFame` и могут подмешиваться в турнир следующего поколения через `mix_opponents`, обеспечивая давление на текущую популяцию и следя за регрессией. Архив сохраняется в `runs/evo/hof.json` рядом с `best_theta.json`.
- **Метрики:** при обучении логируются fitness, `points_per_round`, `win_rate`, а также доля ставок 0 и 1 (см. `metrics.csv` и TensorBoard в `runs/evo`). Команда `rlohhell-eval-evo` дополнительно сохраняет `eval_metrics.csv` с распределением ставок и средними баллами против базовых и Hall-of-Fame оппонентов.
- **Интеграция θ в RL-самоигру:** полученный `best_theta.json` можно добавить в пул оппонентов перед запуском `scripts/train_maskable_self_play.py`:

  ```python
  from rlohhell.heuristics.param_bot import ParamVector
  from rlohhell.utils.opponents import OpponentPool, default_opponents

  theta = ParamVector(**json.load(open("runs/evo/best_theta.json")))
  opponent_pool = OpponentPool(default_opponents(), seed=0)
  opponent_pool.add_theta_opponent(theta, name="evo_theta")
  ```

  После этого пул будет раздавать evo-бота на столы вместе с базовыми стратегиями и снапшотами модели, усиливая разнообразие соперников в процессе RL-самоигры.

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
