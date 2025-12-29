import pytest

from rlohhell.games.base import Card
from rlohhell.games.ohhell.game import OhHellGame
from rlohhell.games.ohhell.utils import determine_winner, TRUMP_SUIT


def test_game_always_sets_diamond_trump():
    game = OhHellGame()
    game.init_game()

    assert game.trump_card.suit == TRUMP_SUIT


def test_diamond_card_wins_against_same_rank_other_suit():
    played_cards = [Card('D', '9'), Card('H', '9')]

    # Passing a non-diamond trump_card should not affect the outcome
    winner = determine_winner(played_cards, Card('S', 'A'))

    assert winner == 0
