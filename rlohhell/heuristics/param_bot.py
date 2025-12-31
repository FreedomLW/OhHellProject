from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from rlohhell.games.base import Card
from rlohhell.games.ohhell.utils import TRUMP_SUIT, determine_winner, get_fixed_trump_card
from rlohhell.utils.opponents import OpponentPolicy
from rlohhell.utils.utils import rank2int


RANK_ORDER = ["A", "K", "Q", "J", "T", "9", "8", "7", "6"]
SUITS = ["S", "H", "D", "C"]


def rank_value(rank: str) -> int:
    """Return a descending index for the rank (A high, 6 low)."""

    try:
        return RANK_ORDER.index(rank)
    except ValueError:
        return len(RANK_ORDER)


def is_trump(card: Card, trump_suit: str = TRUMP_SUIT) -> bool:
    """Check whether ``card`` belongs to the trump suit."""

    return getattr(card, "suit", None) == trump_suit


def _card_strength(card: Card, lead_suit: str | None, trump_suit: str = TRUMP_SUIT) -> Tuple[int, int]:
    """Comparable strength tuple for sorting cards within a trick."""

    is_high_joker = card.suit == "S" and card.rank == "7" and getattr(card, "joker_mode", "low") == "high"
    category = 3 if is_high_joker else 2 if card.suit == trump_suit else 1 if lead_suit and card.suit == lead_suit else 0
    return category, rank2int(card.rank)


def _is_winning_card(candidate: Card, played_cards: Sequence[Card]) -> bool:
    """Return True if ``candidate`` wins against the current trick."""

    cards = list(played_cards) + [candidate]
    if not cards:
        return True
    trump_card = get_fixed_trump_card()
    winner_idx = determine_winner(cards, trump_card)
    return winner_idx == len(cards) - 1


def _series_bonus(hand: Iterable[Card], bonus: float) -> float:
    """Award bonuses for consecutive high-card series within the same suit."""

    score = 0.0
    for suit in SUITS:
        ranks = sorted({rank_value(card.rank) for card in hand if card.suit == suit})
        for first, second in zip(ranks, ranks[1:]):
            if second == first + 1:
                score += bonus
    return score


def _void_bonus(hand: Iterable[Card], bonus: float) -> float:
    suits_present = {card.suit for card in hand}
    missing = [s for s in SUITS if s not in suits_present]
    return len(missing) * bonus


def _hcp_split(hand: Iterable[Card], theta: "ParamVector") -> Tuple[float, float]:
    """Return (non-trump HCP, trump HCP) for the given hand."""

    weights = {
        "A": theta.hcp_A,
        "K": theta.hcp_K,
        "Q": theta.hcp_Q,
        "J": theta.hcp_J,
        "T": theta.hcp_T,
        "9": theta.hcp_9,
        "8": theta.hcp_8,
        "7": theta.hcp_7,
        "6": theta.hcp_6,
    }
    hcp_trump = 0.0
    hcp_plain = 0.0
    for card in hand:
        weight = weights.get(card.rank, 0.0)
        if is_trump(card):
            hcp_trump += weight
        else:
            hcp_plain += weight
    return hcp_plain, hcp_trump


def _trump_len_bonus(hand: Iterable[Card], bonus: float) -> float:
    return sum(1 for card in hand if is_trump(card)) * bonus


def _estimate_bid(state, theta: "ParamVector", game) -> int:
    hand: List[Card] = state["hand"]
    legal_actions: List[int] = state.get("legal_actions", [])
    round_number = getattr(game.round, "round_number", len(hand))

    hcp_plain, hcp_trump = _hcp_split(hand, theta)
    est = hcp_plain + theta.trump_mult * hcp_trump
    est += _trump_len_bonus(hand, theta.trump_len_bonus)
    est += _void_bonus(hand, theta.void_bonus)
    est += _series_bonus(hand, theta.series_bonus)

    seat = getattr(game.round, "players_proposed", 0)
    if seat == game.num_players - 1:
        est += theta.pos_bonus
    elif seat == 0:
        est -= theta.pos_malus

    est = max(0.0, est - theta.lambda_conserv)
    target = int(round(min(round_number, est)))

    if not legal_actions:
        return target

    target = max(0, min(len(hand), target))
    bids_sum = sum(getattr(game.players[i], "proposed_tricks", 0) for i in range(game.num_players) if i != game.get_player_id())
    disallowed = None
    if seat == game.num_players - 1:
        forbidden = round_number - bids_sum
        if 0 <= forbidden <= round_number:
            disallowed = forbidden
    if disallowed is not None and target == disallowed:
        alternatives = [bid for bid in legal_actions if bid != disallowed]
        if alternatives:
            target = min(alternatives, key=lambda b: abs(b - target))

    if target not in legal_actions:
        legal_integers = [bid for bid in legal_actions if isinstance(bid, int)]
        if not legal_integers:
            return legal_actions[0]
        target = min(legal_integers, key=lambda b: abs(b - target))
    return target


def _prefer_low_joker(legal_actions: Sequence[Card], played_cards: Sequence[Card]) -> Card | None:
    for action in legal_actions:
        if action.suit == "S" and action.rank == "7" and getattr(action, "joker_mode", "low") == "low":
            if not _is_winning_card(action, played_cards):
                return action
    return None


def _select_winning_card(legal_actions: Sequence[Card], played_cards: Sequence[Card], hold_threshold: float) -> Card:
    lead_suit = played_cards[0].suit if played_cards else None
    winning_cards = [card for card in legal_actions if _is_winning_card(card, played_cards)]
    if not winning_cards:
        return min(legal_actions, key=lambda c: _card_strength(c, lead_suit))

    winning_cards = sorted(winning_cards, key=lambda c: _card_strength(c, lead_suit))
    dominant = [card for card in winning_cards if _card_strength(card, lead_suit)[0] >= 2]
    if hold_threshold > 0.5 and dominant and len(winning_cards) > 1:
        for candidate in winning_cards:
            if candidate not in dominant:
                return candidate
    return winning_cards[0]


def _select_losing_card(legal_actions: Sequence[Card], played_cards: Sequence[Card]) -> Card:
    lead_suit = played_cards[0].suit if played_cards else None
    losing_cards = [card for card in legal_actions if not _is_winning_card(card, played_cards)]
    if losing_cards:
        return min(losing_cards, key=lambda c: _card_strength(c, lead_suit))
    return min(legal_actions, key=lambda c: _card_strength(c, lead_suit))


@dataclass
class ParamVector:
    hcp_A: float = 1.0
    hcp_K: float = 0.6
    hcp_Q: float = 0.35
    hcp_J: float = 0.2
    hcp_T: float = 0.1
    hcp_9: float = 0.05
    hcp_8: float = 0.02
    hcp_7: float = 0.0
    hcp_6: float = 0.0
    trump_mult: float = 1.6
    trump_len_bonus: float = 0.25
    void_bonus: float = 0.25
    series_bonus: float = 0.1
    pos_bonus: float = 0.1
    pos_malus: float = 0.05
    lambda_conserv: float = 0.15
    seven_sp_high: float = 1.0
    seven_sp_extract: float = 0.2
    play_aggr: float = 0.5
    play_trump_aggr: float = 0.5
    play_hold_winner: float = 0.5

    @classmethod
    def bounds(cls) -> dict:
        return {
            "hcp_A": (0.0, 4.0),
            "hcp_K": (0.0, 3.0),
            "hcp_Q": (0.0, 2.0),
            "hcp_J": (0.0, 2.0),
            "hcp_T": (0.0, 1.5),
            "hcp_9": (0.0, 1.0),
            "hcp_8": (0.0, 1.0),
            "hcp_7": (0.0, 1.0),
            "hcp_6": (0.0, 1.0),
            "trump_mult": (0.5, 3.0),
            "trump_len_bonus": (0.0, 1.0),
            "void_bonus": (0.0, 1.0),
            "series_bonus": (0.0, 1.0),
            "pos_bonus": (0.0, 1.0),
            "pos_malus": (0.0, 1.0),
            "lambda_conserv": (0.0, 1.0),
            "seven_sp_high": (0.0, 2.0),
            "seven_sp_extract": (0.0, 1.0),
            "play_aggr": (0.0, 1.0),
            "play_trump_aggr": (0.0, 1.0),
            "play_hold_winner": (0.0, 1.0),
        }


class ParametricHeuristicOpponent(OpponentPolicy):
    def __init__(self, theta: ParamVector, name: str = "param_bot") -> None:
        self.theta = theta
        super().__init__(name=name, policy_fn=self._dispatch)

    def _dispatch(self, state, action_mask, obs_dict, game):
        legal_actions = state.get("legal_actions", [])
        if not legal_actions:
            return 0

        if isinstance(legal_actions[0], int):
            return _estimate_bid(state, self.theta, game)

        hand: List[Card] = state["hand"]
        played_cards: List[Card] = state.get("played_cards", [])
        tricks_won: int = state.get("tricks_won", 0)
        my_bid: int = state.get("proposed_tricks", 0)
        need = max(0, my_bid - tricks_won)

        if need > 0:
            chosen = _select_winning_card(legal_actions, played_cards, self.theta.play_hold_winner)
        else:
            low_joker = _prefer_low_joker(legal_actions, played_cards)
            if low_joker is not None:
                return low_joker
            chosen = _select_losing_card(legal_actions, played_cards)
        if chosen in legal_actions:
            return chosen

        if isinstance(legal_actions[0], Card):
            return legal_actions[0]
        return 0

    def act(self, state, action_mask, obs_dict, game):  # pragma: no cover - thin wrapper
        return self._dispatch(state, action_mask, obs_dict, game)


def theta_to_vector(theta: ParamVector) -> np.ndarray:
    return np.array([getattr(theta, f.name) for f in fields(theta)], dtype=float)


def vector_to_theta(vector: np.ndarray) -> ParamVector:
    values = {}
    bounds = ParamVector.bounds()
    for field, raw in zip(fields(ParamVector), vector):
        low, high = bounds.get(field.name, (None, None))
        clipped = float(np.clip(raw, low, high)) if low is not None else float(raw)
        values[field.name] = clipped
    return ParamVector(**values)


__all__ = [
    "ParamVector",
    "ParametricHeuristicOpponent",
    "theta_to_vector",
    "vector_to_theta",
    "rank_value",
    "is_trump",
]
