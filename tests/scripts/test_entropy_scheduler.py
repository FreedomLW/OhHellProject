import math

import pytest

sb3 = pytest.importorskip("stable_baselines3")
from scripts.train_maskable_self_play import EntropyScheduler


def test_entropy_scheduler_updates_maskable_ent_coef():
    class DummyModel:
        def __init__(self, ent_coef: float):
            self.ent_coef = ent_coef

    scheduler = EntropyScheduler(start=0.02, end=0.0, total_timesteps=4)
    scheduler.model = DummyModel(ent_coef=0.02)

    for timestep in range(1, 5):
        scheduler.num_timesteps = timestep
        assert scheduler._on_step()

    assert math.isclose(scheduler.model.ent_coef, 0.0, rel_tol=1e-6)
