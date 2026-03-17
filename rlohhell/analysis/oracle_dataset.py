"""One-round oracle dataset generation via bounded/explicit search."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import random
from typing import Callable, Dict, Iterable, List, Sequence

import numpy as np

from rlohhell.games.base import Card
from rlohhell.games.ohhell.game import OhHellGame
from rlohhell.games.ohhell.strategies import BaseStrategy, HeuristicStrategy, RandomStrategy


@dataclass
class OracleSample:
    observation: Dict[str, object]
    legal_actions: List[str]
    chosen_action: str
    action_value_margin: float
    round_size: int
    phase: str
    seat: int
    seed: int
    opponent_profile: str


class OracleDatasetGenerator:
    """Generate oracle-labeled samples for isolated one-round episodes."""

    def __init__(
        self,
        num_players: int = 4,
        rollouts_per_action: int = 4,
        target_rollout_strategy: BaseStrategy | None = None,
        opponent_profile: str = "random",
        opponent_factory: Callable[[int], List[BaseStrategy]] | None = None,
        hand_samples_per_seed: int = 1,
        play_phase_only: bool = True,
    ):
        self.num_players = num_players
        self.rollouts_per_action = rollouts_per_action
        self.target_rollout_strategy = target_rollout_strategy or HeuristicStrategy()
        self.opponent_profile = opponent_profile
        self.opponent_factory = opponent_factory
        self.hand_samples_per_seed = hand_samples_per_seed
        self.play_phase_only = play_phase_only

    def generate(
        self,
        seeds: Sequence[int],
        round_sizes: Sequence[int],
        target_seat: int = 0,
    ) -> List[OracleSample]:
        samples: List[OracleSample] = []
        for seed in seeds:
            for round_size in round_sizes:
                for sample_idx in range(self.hand_samples_per_seed):
                    scenario_seed = seed + sample_idx * 100_003
                    samples.extend(
                        self._generate_for_scenario(
                            seed=scenario_seed,
                            round_size=round_size,
                            target_seat=target_seat,
                        )
                    )
        return samples

    def save_jsonl(self, samples: Iterable[OracleSample], path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            for sample in samples:
                fh.write(json.dumps(asdict(sample), ensure_ascii=False) + "\n")

    def _new_game(self, seed: int, round_size: int) -> OhHellGame:
        game = OhHellGame(num_players=self.num_players)
        game.np_random = np.random.RandomState(seed)
        game.current_player = random.Random(seed).randint(0, self.num_players - 1)
        game.init_game()
        game.round_sequence = [round_size]
        game.max_rounds = 1
        game.current_round = 0
        _, pid = game._start_round(round_size, keep_scores=False)
        game.current_player = pid
        return game

    def _generate_for_scenario(self, seed: int, round_size: int, target_seat: int) -> List[OracleSample]:
        game = self._new_game(seed=seed, round_size=round_size)
        opponent_strategies = self._build_opponents(seed)
        samples: List[OracleSample] = []

        while not game.is_over():
            current_player = game.get_player_id()
            if current_player == target_seat:
                legal_actions = game.get_legal_actions()
                in_play_phase = bool(game.players[current_player].has_proposed)
                if (not self.play_phase_only) or in_play_phase:
                    chosen_action, margin = self._oracle_action(
                        game=game,
                        legal_actions=legal_actions,
                        target_seat=target_seat,
                        seed=seed,
                    )
                    state = game.get_state(current_player)
                    samples.append(
                        OracleSample(
                            observation=self._serialize_state(state),
                            legal_actions=[self._action_to_str(a) for a in legal_actions],
                            chosen_action=self._action_to_str(chosen_action),
                            action_value_margin=float(margin),
                            round_size=round_size,
                            phase="bid" if not in_play_phase else "play",
                            seat=current_player,
                            seed=seed,
                            opponent_profile=self.opponent_profile,
                        )
                    )
                else:
                    chosen_action = self.target_rollout_strategy.select_action(game, current_player)
                game.step(chosen_action)
            else:
                action = opponent_strategies[current_player].select_action(game, current_player)
                game.step(action)

        return samples

    def _oracle_action(self, game: OhHellGame, legal_actions: Sequence[object], target_seat: int, seed: int):
        action_values = []
        for idx, action in enumerate(legal_actions):
            if self.play_phase_only:
                value = self._estimate_action_value_exhaustive_bins(game, action, target_seat=target_seat)
            else:
                value = self._estimate_action_value_rollout(game, action, target_seat=target_seat, seed=seed + idx * 17)
            action_values.append((action, value))

        action_values.sort(key=lambda item: item[1], reverse=True)
        best_action, best_value = action_values[0]
        second_value = action_values[1][1] if len(action_values) > 1 else best_value
        return best_action, best_value - second_value

    def _estimate_action_value_rollout(self, game: OhHellGame, action: object, target_seat: int, seed: int) -> float:
        scores: List[float] = []
        for rollout in range(self.rollouts_per_action):
            sim = deepcopy(game)
            sim.step(deepcopy(action))
            strategies = self._build_opponents(seed + rollout)
            while not sim.is_over():
                player_id = sim.get_player_id()
                if player_id == target_seat:
                    chosen = self.target_rollout_strategy.select_action(sim, player_id)
                else:
                    chosen = strategies[player_id].select_action(sim, player_id)
                sim.step(chosen)
            scores.append(float(sim.get_payoffs()[target_seat]))
        return float(np.mean(scores))

    def _estimate_action_value_exhaustive_bins(self, game: OhHellGame, action: object, target_seat: int) -> float:
        sim = deepcopy(game)
        sim.step(deepcopy(action))
        return float(self._max_bins_from_state(sim, target_seat=target_seat))

    def _max_bins_from_state(self, game: OhHellGame, target_seat: int) -> int:
        if game.is_over():
            return int(game.players[target_seat].tricks_won)

        player_id = game.get_player_id()
        if player_id == target_seat:
            best = -10**9
            for action in game.get_legal_actions():
                nxt = deepcopy(game)
                nxt.step(deepcopy(action))
                best = max(best, self._max_bins_from_state(nxt, target_seat=target_seat))
            return best

        # Opponents are resolved with their strategy to keep tree size manageable.
        strategies = self._build_opponents(0)
        action = strategies[player_id].select_action(game, player_id)
        nxt = deepcopy(game)
        nxt.step(deepcopy(action))
        return self._max_bins_from_state(nxt, target_seat=target_seat)

    def _build_opponents(self, seed: int) -> List[BaseStrategy]:
        if self.opponent_factory is not None:
            strategies = self.opponent_factory(seed)
            if len(strategies) != self.num_players:
                raise ValueError("opponent_factory must return one strategy per seat")
            return strategies

        strategies: List[BaseStrategy] = []
        for seat in range(self.num_players):
            if self.opponent_profile == "heuristic":
                strategy: BaseStrategy = HeuristicStrategy()
            else:
                strategy = RandomStrategy()
            strategy.random_state = random.Random(seed + seat * 997)
            strategies.append(strategy)
        return strategies

    @staticmethod
    def _action_to_str(action: object) -> str:
        if isinstance(action, Card):
            mode = getattr(action, "joker_mode", None)
            return f"{action.get_index()}:{mode}" if mode else action.get_index()
        return str(action)

    @staticmethod
    def _serialize_state(state: Dict[str, object]) -> Dict[str, object]:
        def card_list(cards):
            return [c.get_index() for c in cards]

        return {
            "hand": card_list(state.get("hand", [])),
            "played_cards": card_list(state.get("played_cards", [])),
            "proposed_tricks": int(state.get("proposed_tricks", 0)),
            "tricks_won": int(state.get("tricks_won", 0)),
            "players_tricks_won": [int(v) for v in state.get("players_tricks_won", [])],
            "current_player": int(state.get("current_player", 0)),
            "trump_card": str(state.get("trump_card")),
        }


def generate_oracle_dataset(
    seeds: Sequence[int],
    round_sizes: Sequence[int],
    target_seat: int = 0,
    rollouts_per_action: int = 4,
    opponent_profile: str = "random",
    hand_samples_per_seed: int = 1,
    play_phase_only: bool = True,
) -> List[OracleSample]:
    """Convenience function for one-shot dataset generation."""

    generator = OracleDatasetGenerator(
        rollouts_per_action=rollouts_per_action,
        opponent_profile=opponent_profile,
        hand_samples_per_seed=hand_samples_per_seed,
        play_phase_only=play_phase_only,
    )
    return generator.generate(seeds=seeds, round_sizes=round_sizes, target_seat=target_seat)
