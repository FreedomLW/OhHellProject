"""Optimization utilities for evolutionary search."""

from __future__ import annotations

from dataclasses import fields
from typing import Optional

import numpy as np

from rlohhell.heuristics.param_bot import ParamVector


class CEM:
    """A simple Cross-Entropy Method optimizer."""

    def __init__(
        self,
        init_mean: np.ndarray,
        init_std: np.ndarray,
        pop_size: int = 64,
        elite_frac: float = 0.2,
        min_std: float = 1e-3,
        decay: float = 0.99,
    ) -> None:
        self.pop_size = int(pop_size)
        self.elite_frac = float(elite_frac)
        self.min_std = float(min_std)
        self.decay = float(decay)

        self.mean = np.array(init_mean, dtype=float)
        self.std = np.maximum(np.array(init_std, dtype=float), self.min_std)
        self.num_elite = max(1, int(round(self.pop_size * self.elite_frac)))
        self._population: Optional[np.ndarray] = None

    def ask(self) -> np.ndarray:
        """Sample a population of candidate solutions."""

        self._population = np.random.normal(
            self.mean, self.std, size=(self.pop_size, self.mean.size)
        )
        return self._population

    def tell(self, scores: np.ndarray) -> None:
        """Update the search distribution based on candidate scores."""

        if self._population is None:
            raise ValueError("ask() must be called before tell().")

        scores = np.asarray(scores)
        if scores.shape[0] != self._population.shape[0]:
            raise ValueError("Scores must match the population size from ask().")

        elite_idx = np.argsort(scores)[-self.num_elite :]
        elites = self._population[elite_idx]

        elite_mean = elites.mean(axis=0)
        elite_std = elites.std(axis=0, ddof=0)

        self.mean = self.decay * self.mean + (1 - self.decay) * elite_mean
        self.std = self.decay * self.std + (1 - self.decay) * elite_std
        self.std = np.maximum(self.std, self.min_std)


def clip_to_bounds(vec: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Clip a vector to the provided bounds."""

    return np.clip(vec, lo, hi)


def _param_bounds() -> tuple[np.ndarray, np.ndarray]:
    bounds = ParamVector.bounds()
    field_names = [f.name for f in fields(ParamVector)]
    lo = np.array([bounds[name][0] for name in field_names], dtype=float)
    hi = np.array([bounds[name][1] for name in field_names], dtype=float)
    return lo, hi


def try_cma_es(init_mean: np.ndarray, init_sigma: float):
    """Return a CMA-ES optimizer wrapper if :mod:`cma` is available."""

    try:
        import cma  # type: ignore
    except ImportError:  # pragma: no cover - conditional dependency
        return None

    es = cma.CMAEvolutionStrategy(init_mean, init_sigma)

    class CMAWrapper:
        def __init__(self, strategy):
            self.es = strategy

        def ask(self) -> np.ndarray:
            self._solutions = self.es.ask()
            return np.array(self._solutions)

        def tell(self, scores: np.ndarray) -> None:
            if not hasattr(self, "_solutions"):
                raise ValueError("ask() must be called before tell().")

            scores = np.asarray(scores, dtype=float)
            if scores.shape[0] != len(self._solutions):
                raise ValueError("Scores must match the number of asked solutions.")

            costs = [-float(s) for s in scores]
            self.es.tell(self._solutions, costs)

        @property
        def result(self):
            class ResultProxy:
                @property
                def xbest(self):
                    return self_es.result.xbest

            self_es = self.es
            return ResultProxy()

    return CMAWrapper(es)


__all__ = ["CEM", "clip_to_bounds", "try_cma_es", "_param_bounds"]
