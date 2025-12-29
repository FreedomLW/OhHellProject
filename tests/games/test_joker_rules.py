import numpy as np

from rlohhell.games.base import Card
from rlohhell.games.ohhell.judger import OhHellJudger
from rlohhell.games.ohhell.player import OhHellPlayer
from rlohhell.games.ohhell.round import OhHellRound
from rlohhell.games.ohhell.utils import get_fixed_trump_card


def _build_player(player_id, cards):
    player = OhHellPlayer(player_id, np.random.RandomState())
    player.hand = cards
    player.has_proposed = True
    return player


def test_high_joker_forces_highest_trump_and_wins_trick():
    np_random = np.random.RandomState(0)
    round_inst = OhHellRound(1, 3, np_random, dealer=None, last_winner=0, current_player=0)

    joker_high = Card('S', '7', joker_mode='high')
    players = [
        _build_player(0, [joker_high]),
        _build_player(1, [Card('D', 'A'), Card('D', '9')]),
        _build_player(2, [Card('C', 'A')]),
    ]

    round_inst.proceed_round(players, joker_high)

    legal_actions_second = round_inst.get_legal_actions(players, 1)
    assert legal_actions_second == [players[1].hand[0]]

    round_inst.proceed_round(players, legal_actions_second[0])
    round_inst.proceed_round(players, players[2].hand[0])

    judger = OhHellJudger(np_random)
    winner = judger.judge_round(round_inst.played_cards, get_fixed_trump_card())
    assert winner == 0


def test_low_joker_acts_as_spade_and_can_lose_to_trump():
    np_random = np.random.RandomState(1)
    round_inst = OhHellRound(1, 3, np_random, dealer=None, last_winner=0, current_player=0)

    joker_low = Card('S', '7', joker_mode='low')
    players = [
        _build_player(0, [joker_low]),
        _build_player(1, [Card('S', 'A')]),
        _build_player(2, [Card('D', '8')]),
    ]

    round_inst.proceed_round(players, joker_low)
    round_inst.proceed_round(players, players[1].hand[0])
    round_inst.proceed_round(players, players[2].hand[0])

    judger = OhHellJudger(np_random)
    winner = judger.judge_round(round_inst.played_cards, get_fixed_trump_card())
    assert winner == 2


def test_high_joker_wins_even_when_not_leading():
    np_random = np.random.RandomState(2)
    round_inst = OhHellRound(1, 3, np_random, dealer=None, last_winner=0, current_player=0)

    starter = Card('H', 'K')
    joker_high = Card('S', '7', joker_mode='high')
    players = [
        _build_player(0, [starter]),
        _build_player(1, [joker_high]),
        _build_player(2, [Card('H', 'A')]),
    ]

    round_inst.proceed_round(players, starter)
    round_inst.proceed_round(players, joker_high)
    round_inst.proceed_round(players, players[2].hand[0])

    judger = OhHellJudger(np_random)
    winner = judger.judge_round(round_inst.played_cards, get_fixed_trump_card())
    assert winner == 1
