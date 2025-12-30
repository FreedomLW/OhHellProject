"""Text-based helpers for playing and observing Oh Hell matches.

The :class:`ConsoleOhHellMatch` runner supports two common workflows:

* **Human seat** – assign a human to any seat and the runner will prompt
  for bids and card plays while bots fill the remaining seats.
* **Replay mode** – let bots play each other while the full bidding and
  trick-by-trick history streams to stdout (and is optionally recorded
  for later analysis).

This module keeps the core game rules untouched; it only wraps the
existing :class:`~rlohhell.games.ohhell.game.OhHellGame` API with richer
console I/O.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3 import PPO

from rlohhell.envs.ohhell import OhHellEnv2
from rlohhell.games.base import Card
from rlohhell.games.ohhell.game import OhHellGame as Game
from rlohhell.games.ohhell.strategies import BaseStrategy, RandomStrategy


def _card_label(card: Card) -> str:
    """Return a human-friendly string for ``card``."""

    joker_mode = getattr(card, "joker_mode", None)
    label = f"{card.rank}{card.suit}"
    if joker_mode:
        label += f" ({joker_mode})"
    return label

def _map_rank(rank: str | None) -> int:
    if not rank:
        return 0
    if rank == 'A':
        return 14
    if rank == 'K':
        return 13
    if rank == 'Q':
        return 12
    if rank == 'J':
        return 11
    if rank == 'T':
        return 10
    return int(rank)
    

def _sort_cards(cards: List[Card]) -> List[Card]:
    return sorted(cards, key=lambda x: (x.suit, _map_rank(x.rank)), reverse=True)

def _format_hand(cards: List[Card]) -> str:
    cards = _sort_cards(cards)
    return "  ".join(_card_label(card) for card in cards)


class ModelPolicyStrategy(BaseStrategy):
    """Wrap a saved SB3 policy so it can take turns in console play."""

    def __init__(
        self,
        checkpoint_path: str,
        deterministic: bool = True,
        use_masks: Optional[bool] = None,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.model = self._load_model(checkpoint_path, device)
        self.deterministic = deterministic
        self.use_masks = use_masks if use_masks is not None else isinstance(self.model, MaskablePPO)
        self._env: Optional[OhHellEnv2] = None

    @staticmethod
    def _load_model(checkpoint_path: str, device: str):
        try:
            return MaskablePPO.load(checkpoint_path, device=device)
        except (OSError, ValueError):
            return PPO.load(checkpoint_path, device=device)

    def select_action(self, game: Game, player_id: int) -> Union[int, Card]:
        if self._env is None or self._env.num_players != game.num_players:
            self._env = OhHellEnv2(num_players=game.num_players, agent_id=player_id)

        self._env.game = game
        self._env.agent_id = player_id

        state = game.get_state(player_id)
        obs = self._env._get_obs_dict(state)
        action_mask = obs.get("action_mask")
        kwargs = {"deterministic": self.deterministic}
        if self.use_masks and action_mask is not None:
            kwargs["action_masks"] = action_mask

        action, _ = self.model.predict(obs, **kwargs)
        decoded = self._env._decode_action(int(action), state)
        legal_actions = game.get_legal_actions()
        return decoded if decoded in legal_actions else legal_actions[0]


def load_model_strategy(
    checkpoint_path: str,
    deterministic: bool = True,
    use_masks: Optional[bool] = None,
    device: str = "cpu",
) -> ModelPolicyStrategy:
    """Convenience helper to create a console-ready strategy from an SB3 zip."""

    return ModelPolicyStrategy(
        checkpoint_path=checkpoint_path,
        deterministic=deterministic,
        use_masks=use_masks,
        device=device,
    )


@dataclass
class ConsoleOhHellMatch:
    """Manage an interactive or fully-automated Oh Hell match.

    Args:
        num_players: Total seats at the table.
        human_player: Optional seat index to control manually.
        strategies: Mapping from seat index to bot strategies. Any seat
            not present defaults to :class:`RandomStrategy`.
        record_history: When ``True`` keep a structured log of bids and
            plays for later inspection.
        seed: Optional seed applied to ``random`` to make bot behaviour
            deterministic across runs.
    """

    num_players: int = 4
    human_player: Optional[int] = None
    cheat_mode: bool = False
    strategies: Dict[int, BaseStrategy] = field(default_factory=dict)
    record_history: bool = True
    seed: Optional[int] = None

    def __post_init__(self):
        if self.seed is not None:
            random.seed(self.seed)

        self._bots: Dict[int, BaseStrategy] = {
            seat: strategy for seat, strategy in self.strategies.items()
        }
        for seat in range(self.num_players):
            self._bots.setdefault(seat, RandomStrategy())

        self._history: List[dict] = []

    @property
    def history(self) -> List[dict]:
        return list(self._history)

    def _log(self, entry: dict):
        if self.record_history:
            self._history.append(entry)

    def _prompt_bid(self, legal_actions: List[int], player_id: int) -> int:
        while True:
            print(f"Player {player_id}, enter your bid {legal_actions}: ", end="")
            raw = input().strip()
            try:
                bid = int(raw)
            except ValueError:
                print("Please enter a number.")
                continue

            if bid in legal_actions:
                return bid

            print("Bid not legal; choose one of the offered values.")

    def _prompt_card(self, legal_actions: List[Card], player_id: int, hand: List[Card]) -> Card:
        print(f"Player {player_id} hand: {_format_hand(hand)}")
        legal_actions = _sort_cards(legal_actions)
        options = [
            f"{idx}: {_card_label(card)}" for idx, card in enumerate(legal_actions)
        ]
        print("Legal plays -> " + ",  ".join(options))

        while True:
            print("Choose card by number: ", end="")
            raw = input().strip()
            try:
                selection = int(raw)
            except ValueError:
                print("Please enter the index shown before the card.")
                continue

            if 0 <= selection < len(legal_actions):
                return legal_actions[selection]

            print("Choice not recognised; try again.")

    def _select_action(self, game: Game, player_id: int) -> Union[int, Card]:
        legal_actions = game.get_legal_actions()
        player = game.players[player_id]

        if player_id == self.human_player:
            if player.has_proposed:
                return self._prompt_card(legal_actions, player_id, player.hand)
            return self._prompt_bid(legal_actions, player_id)

        bot = self._bots[player_id]
        return bot.select_action(game, player_id)

    def run(self):
        game = Game(num_players=self.num_players)
        game.init_game()

        print(f"Trump suit: {game.trump_card.get_index()}")
        for seat, player in enumerate(game.players):
            if self.cheat_mode:
                label = _format_hand(player.hand)
            else: 
                label = _format_hand(player.hand) if seat == self.human_player else f"{len(player.hand)} cards"
            print(f"Player {seat} starting hand: {label}")

        # Bidding phase
        while game.round.players_proposed < game.num_players:
            player_id = game.current_player
            action = self._select_action(game, player_id)
            game.step(action)
            self._log({"type": "bid", "player": player_id, "bid": action})
            print(f"Player {player_id} bids {action}")

        print("All bids submitted: " + ", ".join(
            f"P{idx}={player.proposed_tricks}" for idx, player in enumerate(game.players)
        ))

        current_trick: List[tuple[int, Card]] = []
        previous_cards = 0

        while not game.is_over():
            action = self._select_action(game, game.current_player)
            print(f"  Player {game.current_player} -> {_card_label(action)}")
            current_trick.append((game.current_player, action))

            previous_cards = len(game.previously_played_cards)
            game.step(action)

            if len(game.previously_played_cards) > previous_cards:
                winner = game.last_winner
                print("Trick complete:")
                for seat, card in current_trick:
                    print(f"  Player {seat} -> {_card_label(card)}")
                print(f"Winner: Player {winner}\n")
                print('-'*80)
                if self.cheat_mode:
                    for seat, player in enumerate(game.players):
                        cards = _format_hand(player.hand)
                        print(f"Player {seat} hand: {cards}")
                self._log({
                    "type": "trick",
                    "cards": [(seat, _card_label(card)) for seat, card in current_trick],
                    "winner": winner,
                })
                current_trick = []

        scores = game.get_payoffs()
        print("Final scores: " + ", ".join(f"P{idx}: {score}" for idx, score in enumerate(scores)))
        self._log({"type": "result", "scores": scores})

