import math

import numpy as np

from rlohhell.evo.eval_core import evaluate_theta, fixed_seeds
from rlohhell.evo.optimizers import CEM
from rlohhell.heuristics.param_bot import ParamVector, theta_to_vector, vector_to_theta
from rlohhell.utils.opponents import OpponentPool, default_opponents


def test_theta_roundtrip():
    theta = ParamVector()
    vector = theta_to_vector(theta)
    restored = vector_to_theta(vector)

    assert restored == theta

    bounds = ParamVector.bounds()
    exaggerated = np.array([
        bounds[name][1] + 10.0 if idx % 2 == 0 else bounds[name][0] - 10.0
        for idx, name in enumerate(bounds.keys())
    ])
    clipped = vector_to_theta(exaggerated)
    for field, raw in zip(clipped.__dataclass_fields__.keys(), exaggerated):
        low, high = bounds[field]
        assert getattr(clipped, field) == np.clip(raw, low, high)


def test_fixed_seeds_determinism():
    first = fixed_seeds(128, base=42)
    second = fixed_seeds(128, base=42)
    assert first == second


def test_cem_converges_smoke():
    dim = 5
    rng = np.random.default_rng(123)
    init_mean = rng.normal(loc=5.0, scale=1.0, size=dim)
    init_std = np.ones(dim)
    cem = CEM(init_mean=init_mean, init_std=init_std, pop_size=64, elite_frac=0.25)

    def fitness(pop):
        return -np.sum(pop ** 2, axis=1)

    initial_score = -np.sum(init_mean ** 2)
    for _ in range(10):
        population = cem.ask()
        scores = fitness(population)
        cem.tell(scores)

    improved_score = -np.sum(cem.mean ** 2)
    assert improved_score > initial_score


def test_eval_smoke():
    theta = ParamVector()
    opponents = default_opponents()
    pool = OpponentPool(opponents)
    seeds = fixed_seeds(16)

    metrics = evaluate_theta(theta, opponents, seeds, pool, n_jobs=1)

    assert set(metrics.keys()) == {
        "win",
        "points",
        "points_per_round",
        "bid_zero_rate",
        "bid_one_rate",
    }

    assert metrics["win"] >= 0.0 and metrics["win"] <= 1.0
    assert 0.0 <= metrics["bid_zero_rate"] <= 1.0
    assert 0.0 <= metrics["bid_one_rate"] <= 1.0
    assert math.isfinite(metrics["points"])
    assert math.isfinite(metrics["points_per_round"])
    assert abs(metrics["points"]) < 1_000.0
    assert abs(metrics["points_per_round"]) < 50.0
