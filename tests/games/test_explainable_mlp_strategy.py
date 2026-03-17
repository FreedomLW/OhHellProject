from rlohhell.games.base import Card
from rlohhell.games.ohhell.strategies import ExplainableMLPStrategy


class DummyRound:
    round_number = 3


def test_place_bid_returns_default_when_no_legal_actions():
    strategy = ExplainableMLPStrategy(seed=1)

    bid = strategy.place_bid(hand=[], round_state=DummyRound(), legal_actions=[])

    assert bid == 0


def test_place_bid_snaps_to_closest_legal_bid(monkeypatch):
    strategy = ExplainableMLPStrategy(seed=1)

    monkeypatch.setattr(strategy, "_forward", lambda _features, _mask: 4)

    bid = strategy.place_bid(hand=[], round_state=DummyRound(), legal_actions=[0, 2, 3])

    assert bid == 3


def test_play_card_returns_first_legal_if_model_picks_illegal(monkeypatch):
    strategy = ExplainableMLPStrategy(seed=1)
    hand = [Card("S", "6"), Card("H", "7")]
    legal_actions = [hand[1]]

    monkeypatch.setattr(strategy, "_forward", lambda _features, _mask: 0)

    chosen = strategy.play_card(hand=hand, trick_cards=[], legal_actions=legal_actions)

    assert chosen == hand[1]
