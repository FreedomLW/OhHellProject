"""Opponent management utilities for Oh Hell self-play."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Sequence

import numpy as np
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.base_class import BaseAlgorithm

from rlohhell.games.ohhell.strategies import (
    BaseStrategy,
    ConservativeStrategy,
    GreedyStrategy,
    HeuristicStrategy,
    RandomStrategy,
)


if TYPE_CHECKING:
    from rlohhell.heuristics.param_bot import ParamVector

PolicyFn = Callable[[Dict, np.ndarray, Dict, object], object]


@dataclass
class OpponentPolicy:
    """Wrapper around a callable opponent policy."""

    name: str
    policy_fn: PolicyFn
    deterministic: bool = True

    def act(self, state, action_mask, obs_dict, game):
        return self.policy_fn(state, action_mask, obs_dict, game)


class StrategyOpponent(OpponentPolicy):
    """Adapter for :mod:`rlohhell.games.ohhell.strategies` classes."""

    def __init__(self, name: str, strategy: BaseStrategy):
        self.strategy = strategy
        super().__init__(name=name, policy_fn=self._select)

    def _select(self, state, action_mask, obs_dict, game):
        return self.strategy.select_action(game, game.get_player_id())


class ModelOpponent(OpponentPolicy):
    """Policy backed by a frozen SB3 model."""

    def __init__(
        self,
        name: str,
        model: BaseAlgorithm,
        use_masks: bool = True,
        deterministic: bool = True,
    ) -> None:
        super().__init__(name=name, policy_fn=self._predict, deterministic=deterministic)
        self.model = model
        self.use_masks = use_masks
        self.deterministic = deterministic

    def _predict(self, state, action_mask, obs_dict, game):
        kwargs = {"deterministic": self.deterministic}
        if self.use_masks:
            kwargs["action_masks"] = action_mask
        action, _ = self.model.predict(obs_dict, **kwargs)
        return int(action)


class OpponentPool:
    """Manage a pool of opponents and sample table compositions."""

    def __init__(self, opponents: Optional[Sequence[OpponentPolicy]] = None, seed: int = 0):
        self.opponents: List[OpponentPolicy] = list(opponents or [])
        self.random_state = random.Random(seed)
        self.snapshots: Dict[str, str] = {}

    def add_opponent(self, opponent: OpponentPolicy) -> None:
        if len(self.opponents) > 6:
            self.opponents = self.opponents[-6:]
        self.opponents.append(opponent)

    def sample_table(self, num_players: int, agent_id: int) -> Dict[int, OpponentPolicy]:
        assignments: Dict[int, OpponentPolicy] = {}
        if not self.opponents:
            return assignments

        selected = [self.random_state.choice(self.opponents) for _ in range(num_players - 1)]
        idx = 0
        for pid in range(num_players):
            if pid == agent_id:
                continue
            assignments[pid] = selected[idx]
            idx += 1
        return assignments

    def snapshot_model(
        self,
        model: BaseAlgorithm,
        step: int,
        save_dir: str,
        prefix: str = "snapshot",
    ) -> str:
        os.makedirs(save_dir, exist_ok=True)
        name = f"{prefix}_{step}"
        path = os.path.join(save_dir, f"{name}.zip")
        model.save(path)
        loaded = type(model).load(path, device="cpu")
        opponent = ModelOpponent(
            name=name,
            model=loaded,
            use_masks=isinstance(model, MaskablePPO),
            deterministic=True,
        )
        self.snapshots[name] = path
        self.add_opponent(opponent)
        return path

    def policies(self) -> List[OpponentPolicy]:
        return list(self.opponents)


def default_opponents() -> List[OpponentPolicy]:
    """Return baseline opponents: random, greedy and heuristic."""

    return [
        StrategyOpponent("random", RandomStrategy()),
        StrategyOpponent("greedy", GreedyStrategy()),
        StrategyOpponent("heuristic", HeuristicStrategy()),
        StrategyOpponent("conservative", ConservativeStrategy()),
    ]


def make_param_opponent(theta: ParamVector, name: str = "param_bot") -> OpponentPolicy:
    """Build a parametric heuristic opponent with the provided weights."""

    from rlohhell.heuristics.param_bot import ParametricHeuristicOpponent, ParamVector

    return ParametricHeuristicOpponent(theta=theta, name=name)

