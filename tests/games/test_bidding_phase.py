import numpy as np

from rlohhell.games.ohhell.game import OhHellGame as Game
from rlohhell.games.ohhell.round import OhHellRound as Round
from rlohhell.games.ohhell.player import OhHellPlayer as Player


def test_last_bid_corrected_when_forbidden():
    """The last bidder should be nudged away from the forbidden value."""

    np_random = np.random.RandomState()
    round_inst = Round(
        round_number=3,
        num_players=4,
        np_random=np_random,
        dealer=None,
        last_winner=0,
        current_player=0,
    )
    players = [Player(i, np_random) for i in range(4)]

    for bid in [1, 1, 1]:
        round_inst.proceed_round(players, bid)

    forbidden_bid = 0  # would make sum equal number of cards
    legal_before = round_inst.get_legal_actions(players, round_inst.current_player)
    assert forbidden_bid not in legal_before

    round_inst.proceed_round(players, forbidden_bid)

    last_player_bid = players[3].proposed_tricks
    assert last_player_bid != forbidden_bid
    assert last_player_bid in legal_before


def test_bids_persist_for_scoring():
    """Bids are stored for each round so scoring can reference them later."""

    game = Game(num_players=4)
    game.init_game()

    # Only play the opening one-card round to keep the test short
    game.round_sequence = [1]
    game.current_round = 0

    while not game.is_over():
        legal_actions = game.get_legal_actions()
        game.step(legal_actions[0])

    assert len(game.bids_history) == 1
    assert len(game.bids_history[0]) == game.num_players
