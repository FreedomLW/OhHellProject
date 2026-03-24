"""Perfect-information bruteforce oracle strategy for Oh Hell.

Provides :class:`BruteforceStrategy` — a :class:`BaseStrategy` subclass that
sees all players' hands and uses exhaustive game-tree search to find the
action maximising the round score for the target player.

Intended for offline use: generating behaviour-cloning training data and
evaluation benchmarks.
"""

from __future__ import annotations

import copy
from typing import Optional, Union

from rlohhell.games.base import Card
from rlohhell.games.ohhell.game import OhHellGame
from rlohhell.games.ohhell.strategies import BaseStrategy, HeuristicStrategy


class BruteforceStrategy(BaseStrategy):
    """Exhaustive perfect-info search that maximises Oh Hell round score."""

    def __init__(self, opponent_strategy: Optional[BaseStrategy] = None):
        self.opponent_strategy = opponent_strategy or HeuristicStrategy()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def select_action(self, game: OhHellGame, player_id: int) -> Union[int, Card]:
        player = game.players[player_id]
        if not player.has_proposed:
            return self._search_bid(game, player_id)
        return self._search_play(game, player_id)

    def place_bid(self, hand, round_state, legal_actions=None):
        raise NotImplementedError(
            "BruteforceStrategy requires the full game object; "
            "use select_action(game, player_id) instead."
        )

    def play_card(self, hand, trick_cards, legal_actions=None):
        raise NotImplementedError(
            "BruteforceStrategy requires the full game object; "
            "use select_action(game, player_id) instead."
        )

    # ------------------------------------------------------------------
    # Bid search
    # ------------------------------------------------------------------

    def _search_bid(self, game: OhHellGame, player_id: int) -> int:
        legal_bids = game.get_legal_actions()
        initial_score = game.scores[player_id]
        best_bid = legal_bids[0]
        best_value = float("-inf")

        for bid in legal_bids:
            sim = copy.deepcopy(game)
            sim.step(bid)

            # Complete remaining opponent bids.
            while sim.round.players_proposed < sim.num_players:
                cp = sim.get_player_id()
                opp_action = self.opponent_strategy.select_action(sim, cp)
                sim.step(opp_action)

            score = self._search_play_tree(sim, player_id, initial_score)
            if score > best_value:
                best_value = score
                best_bid = bid

        return best_bid

    # ------------------------------------------------------------------
    # Play search
    # ------------------------------------------------------------------

    def _search_play(self, game: OhHellGame, player_id: int) -> Card:
        legal_actions = game.get_legal_actions()
        initial_score = game.scores[player_id]
        initial_round = game.current_round
        best_card = legal_actions[0]
        best_value = float("-inf")

        for card in legal_actions:
            sim = copy.deepcopy(game)
            sim.step(copy.deepcopy(card))
            score = self._recurse(sim, player_id, initial_round, initial_score)
            if score > best_value:
                best_value = score
                best_card = card

        return best_card

    # ------------------------------------------------------------------
    # Recursive tree search
    # ------------------------------------------------------------------

    def _search_play_tree(
        self, game: OhHellGame, target_seat: int, initial_score: float
    ) -> float:
        return self._recurse(game, target_seat, game.current_round, initial_score)

    def _recurse(
        self,
        game: OhHellGame,
        target_seat: int,
        initial_round: int,
        initial_score: float,
    ) -> float:
        # Terminal: round scored and either game over or next round started.
        if game.is_over() or game.current_round > initial_round:
            return float(game.scores[target_seat]) - initial_score

        player_id = game.get_player_id()

        if player_id == target_seat:
            best = float("-inf")
            for action in game.get_legal_actions():
                sim = copy.deepcopy(game)
                sim.step(copy.deepcopy(action))
                best = max(
                    best,
                    self._recurse(sim, target_seat, initial_round, initial_score),
                )
            return best

        # Opponent: resolve deterministically.
        action = self.opponent_strategy.select_action(game, player_id)
        sim = copy.deepcopy(game)
        sim.step(copy.deepcopy(action))
        return self._recurse(sim, target_seat, initial_round, initial_score)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_score(tricks_won: int, bid: int) -> float:
        """Replicate ``OhHellJudger.judge_game`` for a single player."""
        if tricks_won == bid:
            return float(10 * bid) if bid > 0 else 5.0
        if tricks_won > bid:
            return float(tricks_won)
        return float(-10 * bid)
