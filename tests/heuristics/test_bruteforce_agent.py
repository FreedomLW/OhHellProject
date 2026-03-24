import copy

import numpy as np

from rlohhell.games.ohhell.game import OhHellGame
from rlohhell.games.ohhell.strategies import HeuristicStrategy, RandomStrategy
from rlohhell.heuristics.bruteforce_agent import BruteforceStrategy


def _make_single_round_game(round_size: int, seed: int = 0) -> OhHellGame:
    """Create a deterministic single-round game ready for bidding."""
    game = OhHellGame(num_players=4)
    game.np_random = np.random.RandomState(seed)
    game.current_player = seed % 4
    state, _ = game.init_game()
    game.round_sequence = [round_size]
    game.max_rounds = 1
    game.current_round = 0
    return game


# ------------------------------------------------------------------
# Scoring helper
# ------------------------------------------------------------------


def test_compute_score_matches_judger():
    assert BruteforceStrategy._compute_score(2, 2) == 20.0
    assert BruteforceStrategy._compute_score(0, 0) == 5.0
    assert BruteforceStrategy._compute_score(3, 1) == 3.0
    assert BruteforceStrategy._compute_score(1, 3) == -30.0
    assert BruteforceStrategy._compute_score(5, 5) == 50.0
    assert BruteforceStrategy._compute_score(0, 2) == -20.0


# ------------------------------------------------------------------
# Trivial round
# ------------------------------------------------------------------


def test_trivial_round_one_card():
    """Round size 1: bruteforce bids and plays, game completes."""
    game = _make_single_round_game(round_size=1, seed=42)
    bf = BruteforceStrategy()
    opp = HeuristicStrategy()

    while not game.is_over():
        pid = game.get_player_id()
        if pid == 0:
            action = bf.select_action(game, pid)
        else:
            action = opp.select_action(game, pid)
        game.step(action)

    assert game.is_over()
    assert any(s != 0 for s in game.scores)


# ------------------------------------------------------------------
# Determinism
# ------------------------------------------------------------------


def test_determinism():
    """Same game state produces the same action."""
    game = _make_single_round_game(round_size=3, seed=7)
    bf = BruteforceStrategy()

    game_a = copy.deepcopy(game)
    game_b = copy.deepcopy(game)

    action_a = bf.select_action(game_a, game_a.get_player_id())
    action_b = bf.select_action(game_b, game_b.get_player_id())

    # For bids (int) or cards (Card.__eq__)
    assert action_a == action_b


# ------------------------------------------------------------------
# Full round integration
# ------------------------------------------------------------------


def test_full_round_integration():
    """Complete a round with bruteforce at seat 0 vs heuristic opponents."""
    for seed in (0, 13, 99):
        for rs in (1, 2, 3):
            game = _make_single_round_game(round_size=rs, seed=seed)
            bf = BruteforceStrategy()
            opp = HeuristicStrategy()

            while not game.is_over():
                pid = game.get_player_id()
                if pid == 0:
                    action = bf.select_action(game, pid)
                else:
                    action = opp.select_action(game, pid)
                game.step(action)

            assert game.is_over()
            # Score must be non-zero for at least one player.
            assert any(s != 0 for s in game.scores)


# ------------------------------------------------------------------
# Score-awareness
# ------------------------------------------------------------------


def test_bid_search_prefers_exact_match():
    """Bruteforce should prefer a bid that can be matched exactly over one
    that leads to overbidding (consolation) when the exact-match score is
    higher."""
    # Run many seeds; on at least some, the bruteforce should pick a bid
    # that differs from the greedy "max tricks" approach.
    bf = BruteforceStrategy()
    greedy_mismatches = 0

    for seed in range(30):
        game = _make_single_round_game(round_size=3, seed=seed)
        pid = game.get_player_id()
        if pid != 0:
            # Skip if seat 0 doesn't bid first in this seed.
            continue
        bid = bf.select_action(game, 0)
        # A pure max-tricks oracle would always bid 3 for round_size=3
        # when holding strong cards; score-aware search may bid lower.
        if bid < 3:
            greedy_mismatches += 1

    # At least one seed should produce a non-max bid (score-aware behaviour).
    assert greedy_mismatches > 0, (
        "Expected bruteforce to bid below max on at least one seed"
    )
