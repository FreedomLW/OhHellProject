import builtins
import sys
import types

import numpy as np
import pytest

from rlohhell.evo.optimizers import CEM, _param_bounds, clip_to_bounds, try_cma_es


def test_cem_updates_mean_and_std():
    np.random.seed(0)
    init_mean = np.zeros(2)
    init_std = np.ones(2)
    cem = CEM(init_mean=init_mean, init_std=init_std, pop_size=10, elite_frac=0.2, decay=0.0)

    population = cem.ask()
    scores = -np.sum((population - 1.0) ** 2, axis=1)
    cem.tell(scores)

    elite_idx = np.argsort(scores)[-cem.num_elite :]
    elites = population[elite_idx]
    expected_mean = elites.mean(axis=0)
    expected_std = np.maximum(elites.std(axis=0, ddof=0), cem.min_std)

    assert np.allclose(cem.mean, expected_mean)
    assert np.allclose(cem.std, expected_std)


def test_cem_respects_min_std():
    np.random.seed(1)
    cem = CEM(init_mean=np.zeros(1), init_std=np.array([0.0]), pop_size=4, elite_frac=0.5, min_std=0.1, decay=0.5)
    cem.ask()
    cem.tell(np.zeros(4))

    assert np.all(cem.std >= 0.1)


def test_clip_to_bounds_matches_paramvector_bounds():
    lo, hi = _param_bounds()
    vector = np.full_like(lo, -1.0)
    clipped = clip_to_bounds(vector, lo, hi)

    assert np.allclose(clipped, lo)


def test_try_cma_es_without_dependency(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "cma":
            raise ImportError
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert try_cma_es(np.zeros(2), 0.5) is None


def test_try_cma_es_with_stub(monkeypatch):
    class DummyResult:
        def __init__(self):
            self.xbest = [1.0, -1.0]

    class DummyStrategy:
        def __init__(self, init_mean, sigma):
            self.init_mean = init_mean
            self.sigma = sigma
            self.result = DummyResult()

        def ask(self):
            return [
                list(np.array(self.init_mean) + 1),
                list(np.array(self.init_mean) - 1),
            ]

        def tell(self, solutions, costs):
            self.last_solutions = solutions
            self.last_costs = costs

    dummy_module = types.SimpleNamespace(CMAEvolutionStrategy=lambda mean, sigma: DummyStrategy(mean, sigma))
    monkeypatch.setitem(sys.modules, "cma", dummy_module)

    optimizer = try_cma_es(np.zeros(2), 0.5)
    assert optimizer is not None

    candidates = optimizer.ask()
    optimizer.tell(np.array([1.0, 2.0]))

    assert candidates.shape == (2, 2)
    assert optimizer.es.last_costs == [-1.0, -2.0]
    assert optimizer.result.xbest == [1.0, -1.0]
