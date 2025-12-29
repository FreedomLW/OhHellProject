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
:class:`~rlohhell.games.ohhell.game.OhHellGame` helper methods.  Concrete
strategies only need to focus on the bidding/play heuristics.
"""

from __future__ import annotations

import random
from typing import Iterable, List, Optional

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

