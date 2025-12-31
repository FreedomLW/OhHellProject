"""Hall of Fame utilities for opponent co-evolution."""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, is_dataclass
from typing import List, Sequence, Tuple

from rlohhell.heuristics.param_bot import ParamVector
from rlohhell.utils.opponents import OpponentPolicy, make_param_opponent


class HallOfFame:
    """Archive of top-performing parameter vectors."""

    def __init__(self) -> None:
        self.entries: List[Tuple[ParamVector, float, int]] = []

    def add(self, theta: ParamVector, score: float, step: int) -> None:
        """Add a candidate to the archive, keeping it sorted by score."""

        self.entries.append((theta, score, step))
        self.entries.sort(key=lambda item: item[1], reverse=True)

    def top(self, k: int) -> List[Tuple[ParamVector, float, int]]:
        """Return the ``k`` highest-scoring entries."""

        return self.entries[:k]

    def save(self, path: str) -> None:
        """Persist the archive to ``path`` in JSON format."""

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        serialised = [
            {
                "theta": asdict(theta) if is_dataclass(theta) else dict(theta),
                "score": score,
                "step": step,
            }
            for theta, score, step in self.entries
        ]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(serialised, f)

    @classmethod
    def load(cls, path: str) -> "HallOfFame":
        """Load an archive from ``path``."""

        instance = cls()
        if not os.path.exists(path):
            return instance

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for record in data:
            theta = ParamVector(**record["theta"])
            instance.add(theta, float(record["score"]), int(record["step"]))
        return instance


def theta_opponent(theta: ParamVector, name: str) -> OpponentPolicy:
    """Wrap a :class:`ParamVector` into an :class:`OpponentPolicy`."""

    return make_param_opponent(theta, name=name)


def mix_opponents(
    hof: HallOfFame, builtin: Sequence[OpponentPolicy], k: int
) -> List[OpponentPolicy]:
    """Combine base opponents with a random sample from the hall of fame."""

    opponents = list(builtin)
    candidates = hof.top(k)
    if candidates:
        sample = random.sample(candidates, k=min(k, len(candidates)))
        opponents.extend(
            theta_opponent(theta, name=f"hof_{step}") for theta, _score, step in sample
        )
    return opponents
