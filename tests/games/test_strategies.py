from rlohhell.games.ohhell.game import OhHellGame as Game
from rlohhell.games.ohhell.strategies import (
    ConservativeStrategy,
    GreedyStrategy,
    HeuristicStrategy,
    RandomStrategy,
)


def test_full_bot_game_runs_to_completion():
    strategies = [RandomStrategy(), GreedyStrategy(), ConservativeStrategy(), HeuristicStrategy()]
    game = Game(num_players=4, player_strategies=strategies)
    game.init_game()

    game.play_full_game()

    expected_cards = sum(game.round_sequence) * game.num_players
    assert game.is_over()
    assert len(game.previously_played_cards) == expected_cards
    assert game.get_payoffs() == tuple(game.scores)


def test_automated_play_stops_for_manual_slot():
    strategies = [RandomStrategy(), None, RandomStrategy(), RandomStrategy()]
    game = Game(num_players=4, player_strategies=strategies)
    state, player_id = game.init_game()

    state, player_id = game.play_automated()

    assert player_id == game.current_player
    assert game.player_strategies[player_id] is None
    assert state['current_player'] == player_id
