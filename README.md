# Reinforcement Learning Project on OhHell

![Laptop and cards](img.jpg)

A project to create a reinforcement learning agent to play the card game OH Hell. There are various versions of the game but they all require a lot of skill to master. The ultimate aim of the project is to search for the best possible strategy when playing this game.

The version of the game used in this code is a simple one-off 10 card game.

This is a link to a website where you can play a Oh Hell game that plays 10 cards then 9 cards all the way down to 1 card and then back up to 10 cards - [Oh Hell Website](https://cardgames.io/ohhell/)

# Credits
Throughout this project I used a lot of different resources and techniques to create the game environment and neural network

- The structure of the game environment - [RLCARD](https://github.com/datamllab/rlcard)
- The idea to train the agent against itself (NFSP) - [NFSP](https://arxiv.org/abs/1603.01121)
- A template project using PPO on the game Big2 - [Big2PPO](https://github.com/henrycharlesworth/big2_PPOalgorithm)
- For an implementation of the PPO algorithm in PyTorch - [SB3](https://github.com/DLR-RM/stable-baselines3)

## Playing or watching a console match

The package includes a simple console runner that lets you either **play a seat yourself** or **watch bots battle** while their bids and tricks stream to stdout.

### Sit a human at the table
Run this from the repository root to control seat 0 and let bots play the other seats:

```bash
python - <<'PY'
from rlohhell.games.ohhell.console import ConsoleOhHellMatch

match = ConsoleOhHellMatch(num_players=4, human_player=0)
match.run()
PY
```

You will be prompted for a bid and for each card you play; the other seats default to the built-in `RandomStrategy` unless you provide different strategies via the `strategies` argument.

### Watch a replay between bots
If you just want to observe, leave `human_player=None` and the round will log every bid and trick:

```bash
python - <<'PY'
from rlohhell.games.ohhell.console import ConsoleOhHellMatch
from rlohhell.games.ohhell.strategies import ConservativeStrategy, GreedyStrategy

# Seat 0 plays Conservative, seat 1 plays Greedy, seats 2 and 3 default to Random.
match = ConsoleOhHellMatch(
    strategies={0: ConservativeStrategy(), 1: GreedyStrategy()},
)
match.run()
PY
```

Set `record_history=True` (the default) to keep a structured log of the match in `match.history` for further analysis.

