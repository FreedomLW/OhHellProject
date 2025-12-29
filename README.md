# Reinforcement Learning Project on OhHell

![Laptop and cards](img.jpg)

A project to create a reinforcement learning agent to play the card game OH Hell. There are various versions of the game but they all require a lot of skill to master. The ultimate aim of the project is to search for the best possible strategy when playing this game.

The version of the game used in this code is a simple one-off 10 card game.

This is a link to a website where you can play a Oh Hell game that plays 10 cards then 9 cards all the way down to 1 card and then back up to 10 cards - [Oh Hell Website](https://cardgames.io/ohhell/)

## Playing or observing in the console
You can play a quick text-mode match or watch bots battle by using the `ConsoleOhHellMatch` helper:

```bash
python - <<'PY'
from rlohhell.games.ohhell.console import ConsoleOhHellMatch

# Example 1: take seat 0 and play against three random bots
ConsoleOhHellMatch(num_players=4, human_player=0).run()

# Example 2: watch four bots with a fixed seed for reproducible logs
ConsoleOhHellMatch(num_players=4, human_player=None, seed=123).run()
PY
```

During bidding you will be prompted for your bid; during play you will see your hand, the legal cards you can choose, and who wins each trick. If you set `record_history=True` (default) you can also inspect the in-memory log after the match.

# Credits
Throughout this project I used a lot of different resources and techniques to create the game environment and neural network

- The structure of the game environment - [RLCARD](https://github.com/datamllab/rlcard)
- The idea to train the agent against itself (NFSP) - [NFSP](https://arxiv.org/abs/1603.01121) 
- A template project using PPO on the game Big2 - [Big2PPO](https://github.com/henrycharlesworth/big2_PPOalgorithm)
- For an implementation of the PPO algorithm in PyTorch - [SB3](https://github.com/DLR-RM/stable-baselines3)

