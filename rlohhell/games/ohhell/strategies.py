"""Simple, non-learning strategies for playing Oh Hell.

The classes in this module implement lightweight bidding and card play
heuristics.  They are intentionally deterministic (or controllably
stochastic) and rely solely on the public game/round state to choose
actions.  Every strategy exposes the same two methods so they can be
swapped freely:

``place_bid(hand, round_state)``
    Decide how many tricks to bid for the current round.

``play_card(hand, trick_cards)``
    Select a card to play for the current trick.

The :class:`BaseStrategy` class provides the glue logic required by the
:class:`~rlohhell.games.ohhell.game.OhHellGame` helper methods. Concrete
strategies only need to focus on the bidding/play heuristics.
"""

from __future__ import annotations

import json
import os
import random
from typing import Iterable, List, Optional

import numpy as np

import rlohhell
from rlohhell.games.base import Card
from rlohhell.games.ohhell.utils import TRUMP_SUIT, determine_winner, get_fixed_trump_card
from rlohhell.utils.utils import rank2int


def _card_strength(card: Card, lead_suit: Optional[str], trump_suit: str = TRUMP_SUIT):
    """Return a tuple describing how powerful ``card`` is in the current trick.

    Higher tuples indicate a stronger card.  Ordering is based on the Oh Hell
    rules: high joker > trump > following suit > off-suit.  Within each
    category the natural rank order applies.
    """

    is_high_joker = card.suit == "S" and card.rank == "7" and getattr(card, "joker_mode", "low") == "high"
    is_trump = card.suit == trump_suit
    follows_lead = lead_suit is not None and card.suit == lead_suit
    return (
        3 if is_high_joker else 2 if is_trump else 1 if follows_lead else 0,
        rank2int(card.rank),
    )


def _is_winning_card(candidate: Card, played_cards: Iterable[Card], trump_suit: str = TRUMP_SUIT) -> bool:
    """Check if ``candidate`` currently wins the trick against ``played_cards``."""

    cards = list(played_cards) + [candidate]
    if not cards:
        return True

    trump_card = get_fixed_trump_card()
    winning_index = determine_winner(cards, trump_card)
    return winning_index == len(cards) - 1


class BaseStrategy:
    """Common utilities shared by all deterministic strategy classes."""

    random_state = random.Random()

    def place_bid(self, hand: List[Card], round_state, legal_actions=None):
        raise NotImplementedError

    def play_card(self, hand: List[Card], trick_cards: List[Card], legal_actions=None):
        raise NotImplementedError

    def select_action(self, game, player_id):
        """Choose a legal action for ``player_id`` in ``game``.

        This method bridges the strategy interface with the game mechanics.
        It ensures the returned action is legal; if the strategy proposes an
        invalid action the first available legal option is used instead.
        """

        legal_actions = game.get_legal_actions()
        player = game.players[player_id]
        self._current_player_state = player
        self._round_state = game.round

        if not player.has_proposed:
            bid = self.place_bid(list(player.hand), game.round, legal_actions)
            self._current_bid = bid
            return bid if bid in legal_actions else legal_actions[0]

        trick_cards = list(game.round.played_cards)
        card = self.play_card(list(player.hand), trick_cards, legal_actions)
        return card if card in legal_actions else legal_actions[0]


class RandomStrategy(BaseStrategy):
    """Pick any legal bid or card uniformly at random."""

    def place_bid(self, hand: List[Card], round_state, legal_actions=None):
        legal_bids = legal_actions if legal_actions is not None else list(range(round_state.round_number + 1))
        return self.random_state.choice(list(legal_bids))

    def play_card(self, hand: List[Card], trick_cards: List[Card], legal_actions=None):
        legal_cards = legal_actions if legal_actions is not None else hand
        return self.random_state.choice(list(legal_cards))


class GreedyStrategy(BaseStrategy):
    """Bid high and attempt to win every trick with the smallest winning card."""

    def place_bid(self, hand: List[Card], round_state, legal_actions=None):
        target = round_state.round_number
        if legal_actions is None:
            return target
        return max(a for a in legal_actions if isinstance(a, int) and a <= target)

    def play_card(self, hand: List[Card], trick_cards: List[Card], legal_actions=None):
        legal_cards = list(legal_actions) if legal_actions is not None else hand
        lead_suit = trick_cards[0].suit if trick_cards else None

        winning_cards = [card for card in legal_cards if _is_winning_card(card, trick_cards)]
        if winning_cards:
            return min(winning_cards, key=lambda c: _card_strength(c, lead_suit))

        return min(legal_cards, key=lambda c: _card_strength(c, lead_suit))


class ConservativeStrategy(BaseStrategy):
    """Always bid zero and try to avoid winning tricks."""

    def place_bid(self, hand: List[Card], round_state, legal_actions=None):
        if legal_actions is None:
            return 0
        return 0 if 0 in legal_actions else min(a for a in legal_actions if isinstance(a, int))

    def play_card(self, hand: List[Card], trick_cards: List[Card], legal_actions=None):
        legal_cards = list(legal_actions) if legal_actions is not None else hand
        lead_suit = trick_cards[0].suit if trick_cards else None

        losing_cards = [card for card in legal_cards if not _is_winning_card(card, trick_cards)]
        if losing_cards:
            return min(losing_cards, key=lambda c: _card_strength(c, lead_suit))

        return min(legal_cards, key=lambda c: _card_strength(c, lead_suit))


class HeuristicStrategy(BaseStrategy):
    """Track the current bid and steer towards fulfilling it.

    The heuristic is intentionally simple: bid based on strong cards
    (trumps and high ranks) and, during play, attempt to win tricks only
    when more are required to match the bid.
    """

    def place_bid(self, hand: List[Card], round_state, legal_actions=None):
        trump_cards = [card for card in hand if card.suit == TRUMP_SUIT]
        high_cards = [card for card in hand if rank2int(card.rank) >= 12]
        estimate = min(round_state.round_number, max(0, len(trump_cards) + len(high_cards) // 2))

        if legal_actions is None or estimate in legal_actions:
            return estimate

        legal_bids = [a for a in legal_actions if isinstance(a, int)]
        if not legal_bids:
            return estimate
        return min(legal_bids, key=lambda bid: abs(bid - estimate))

    def play_card(self, hand: List[Card], trick_cards: List[Card], legal_actions=None):
        legal_cards = list(legal_actions) if legal_actions is not None else hand
        lead_suit = trick_cards[0].suit if trick_cards else None

        player = getattr(self, "_current_player_state", None)
        tricks_needed = 0
        if player is not None:
            tricks_needed = max(0, player.proposed_tricks - player.tricks_won)

        if tricks_needed > 0:
            winning_cards = [card for card in legal_cards if _is_winning_card(card, trick_cards)]
            if winning_cards:
                return min(winning_cards, key=lambda c: _card_strength(c, lead_suit))
            return max(legal_cards, key=lambda c: _card_strength(c, lead_suit))

        losing_cards = [card for card in legal_cards if not _is_winning_card(card, trick_cards)]
        if losing_cards:
            return min(losing_cards, key=lambda c: _card_strength(c, lead_suit))

        return min(legal_cards, key=lambda c: _card_strength(c, lead_suit))


class ExplainableMLPStrategy(BaseStrategy):
    """A tiny, human-auditable MLP for bidding and play selection.

    The model uses one-hot encodings for the player's hand and the global
    discard history plus a handful of scalar context features (current bid,
    tricks already won, and remaining tricks in the round). All weights are
    initialised deterministically to keep the behaviour reproducible and
    easy to reason about.
    """

    def __init__(self, hidden_dim: int = 32, seed: int = 13):
        super().__init__()
        self.random_state = random.Random(seed)
        self.rng = np.random.default_rng(seed)

        with open(os.path.join(rlohhell.__path__[0], "games/ohhell/card2index.json"), "r") as fh:
            self.card2index = json.load(fh)

        # 36 cards per short deck, mirrored for hand + played history, plus
        # three scalar context features.
        self.input_dim = 36 * 2 + 3
        self.hidden_dim = hidden_dim

        scale = 1.0 / np.sqrt(self.input_dim)
        self.w1 = self.rng.normal(0, scale, size=(self.input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.w2 = self.rng.normal(0, scale, size=(hidden_dim, 64))
        self.b2 = np.zeros(64)

    def _encode_cards(self, cards: List[Card]):
        encoded = np.zeros(36, dtype=np.float32)
        for card in cards:
            idx = self.card2index.get(card.get_str())
            if idx is not None:
                encoded[idx] = 1.0
        return encoded

    def _forward(self, features: np.ndarray, mask: List[int]):
        hidden = np.tanh(features @ self.w1 + self.b1)
        logits = hidden @ self.w2 + self.b2
        logits = logits[: len(mask)]
        logits = np.where(mask, logits, -np.inf)
        return int(np.argmax(logits))

    def _encode_features(self, hand: List[Card], round_state, played: List[Card], player_state):
        hand_vec = self._encode_cards(hand)
        played_vec = self._encode_cards(played)
        context = np.array(
            [
                getattr(player_state, "proposed_tricks", 0),
                getattr(player_state, "tricks_won", 0),
                getattr(round_state, "round_number", 0),
            ],
            dtype=np.float32,
        )
        return np.concatenate([hand_vec, played_vec, context])

    def place_bid(self, hand: List[Card], round_state, legal_actions=None):
        legal = list(legal_actions) if legal_actions is not None else list(range(round_state.round_number + 1))
        mask = [action in legal for action in range(max(legal) + 1)]
        player_state = getattr(self, "_current_player_state", None) or type("_P", (), {"proposed_tricks": 0, "tricks_won": 0})()
        features = self._encode_features(hand, round_state, played=[], player_state=player_state)
        bid = self._forward(features, mask)
        if bid not in legal:
            return min(legal, key=lambda a: abs(a - bid))
        return bid

    def play_card(self, hand: List[Card], trick_cards: List[Card], legal_actions=None):
        legal_cards = list(legal_actions) if legal_actions is not None else hand
        if not legal_cards:
            return None

        mask = [card in legal_cards for card in hand]
        player_state = getattr(self, "_current_player_state", None) or type("_P", (), {"proposed_tricks": 0, "tricks_won": 0})()
        round_state = getattr(self, "_round_state", round_state_placeholder())
        features = self._encode_features(hand, round_state, played=trick_cards, player_state=player_state)
        choice_index = self._forward(features, mask)
        if 0 <= choice_index < len(hand):
            chosen = hand[choice_index]
            if chosen in legal_cards:
                return chosen
        return legal_cards[0]


def round_state_placeholder():
    placeholder = type("_Round", (), {})()
    placeholder.round_number = 0
    return placeholder

