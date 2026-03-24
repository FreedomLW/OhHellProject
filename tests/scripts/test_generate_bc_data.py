import numpy as np
import pytest

from scripts.generate_bc_data import generate_dataset, action_to_id
from rlohhell.games.base import Card


def test_action_to_id_bid():
    assert action_to_id(3, []) == 3


def test_action_to_id_card():
    hand = [Card("S", "A"), Card("H", "6"), Card("D", "9")]
    assert action_to_id(Card("H", "6"), hand) == 1


def test_action_to_id_joker():
    hand = [Card("S", "7"), Card("H", "A")]
    low = Card("S", "7", joker_mode="low")
    high = Card("S", "7", joker_mode="high")
    assert action_to_id(low, hand) == len(hand)
    assert action_to_id(high, hand) == len(hand) + 1


def test_bc_data_generation_smoke():
    data = generate_dataset(num_seeds=3, round_sizes=[1, 2])

    obs = data["observations"]
    masks = data["action_masks"]
    actions = data["actions"]

    assert obs.ndim == 2
    assert obs.shape[1] == 127
    assert masks.ndim == 2
    assert masks.shape[1] == 11
    assert actions.ndim == 1
    assert len(obs) == len(masks) == len(actions)
    assert len(actions) > 0

    # Every action_id must be legal (within mask).
    for i in range(len(actions)):
        aid = actions[i]
        assert 0 <= aid < 11
        assert masks[i, aid] == 1, f"Sample {i}: action {aid} not in mask"
