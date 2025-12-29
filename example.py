from rlohhell.games.ohhell.console import ConsoleOhHellMatch
from rlohhell.games.ohhell.strategies import HeuristicStrategy

# Example 1: take seat 0 and play against three random bots
ConsoleOhHellMatch(num_players=4, human_player=0, cheat_mode=True, strategies={
    1: HeuristicStrategy(),
    2: HeuristicStrategy(),
    3: HeuristicStrategy()
}).run()
