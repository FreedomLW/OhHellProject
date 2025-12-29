import numpy as np

from rlohhell.envs.ohhell import OhHellEnv2, mask_from_state
from rlohhell.games.base import Card


def test_mask_prevents_illegal_last_bid():
    env = OhHellEnv2(num_players=3, agent_id=2)
    env.reset()

    game = env.game
    round_state = game.round
    round_state.round_number = 3
    round_state.players_proposed = game.num_players - 1
    round_state.proposed_tricks = [1, 1, 0]
    round_state.current_player = env.agent_id

    for idx, player in enumerate(game.players):
        player.has_proposed = idx < game.num_players - 1
        player.proposed_tricks = round_state.proposed_tricks[idx]
        player.hand = [Card("S", "6"), Card("H", "6"), Card("D", "6")]

    state = game.get_state(env.agent_id)
    mask = mask_from_state(state)

    assert mask.dtype == np.bool_
    assert mask.shape == (len(game.players[env.agent_id].hand) + 1,)
    assert bool(mask[1]) is False
    assert mask.sum() == mask.size - 1


def test_mask_enforces_following_suit_and_joker_modes():
    env = OhHellEnv2(num_players=2, agent_id=0)
    env.reset()

    game = env.game
    round_state = game.round
    round_state.round_number = 3
    round_state.players_proposed = game.num_players
    round_state.current_player = env.agent_id
    round_state.last_winner = 1
    round_state.played_cards = [Card("H", "9")]

    player = game.players[env.agent_id]
    player.has_proposed = True
    player.hand = [Card("H", "6"), Card("S", "7"), Card("S", "6")]

    state = game.get_state(env.agent_id)
    mask = mask_from_state(state)

    assert mask.dtype == np.bool_
    assert mask.shape == (len(player.hand) + 2,)

    assert bool(mask[0]) is True  # Follow suit with hearts
    assert bool(mask[2]) is False  # Cannot slough off other suits while holding hearts
    assert bool(mask[len(player.hand)]) is True  # 7♠ low option
    assert bool(mask[len(player.hand) + 1]) is True  # 7♠ high option
