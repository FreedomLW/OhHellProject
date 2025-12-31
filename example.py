from rlohhell.games.ohhell.console import ConsoleOhHellMatch, load_model_strategy
from rlohhell.games.ohhell.strategies import HeuristicStrategy

bot = load_model_strategy("runs/maskable_ppo/best_model/best_model.zip")

# Example 1: take seat 0 and play against three random bots
ConsoleOhHellMatch(num_players=4, human_player=0, cheat_mode=True, strategies={
    1: bot,
    2: bot,
    3: bot
}).run()
