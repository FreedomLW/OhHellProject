import random

from rlohhell.games.ohhell.game import OhHellGame as Game


def test_round_sequence_generation():
    """The round sequence should match the Odessa poker rule for four players."""

    game = Game(num_players=4)
    game.init_game()

    expected = [1, 2, 3, 4, 5, 6, 7, 8, 9] + [9] * 4 + [8, 7, 6, 5, 4, 3, 2, 1]
    assert game.round_sequence == expected
    assert game.round.round_number == 1


def test_full_game_progression_and_scoring():
    """Playing a full random game should consume the whole round sequence."""

    game = Game(num_players=4)
    game.init_game()

    total_expected_cards = sum(game.round_sequence) * game.num_players

    while not game.is_over():
        action = random.choice(game.get_legal_actions())
        game.step(action)

    # All cards from all rounds should have been played
    assert len(game.previously_played_cards) == total_expected_cards

    # Game should report payoffs equal to stored scores
    assert game.get_payoffs() == tuple(game.scores)
    assert game.current_round == len(game.round_sequence)
