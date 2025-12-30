import math

import pytest

sb3 = pytest.importorskip("stable_baselines3")
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

from rlohhell.envs.ohhell import OhHellEnv2
from scripts.train_maskable_self_play import EntropyScheduler

sb3_contrib = pytest.importorskip("sb3_contrib")
from sb3_contrib.ppo_mask import MaskablePPO


def test_entropy_scheduler_updates_maskable_ent_coef():
    env = make_vec_env(
        lambda: OhHellEnv2(num_players=4, agent_id=0), n_envs=1, vec_env_cls=DummyVecEnv
    )
    scheduler = EntropyScheduler(start=0.02, end=0.0, total_timesteps=4)
    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        ent_coef=0.02,
        n_steps=1,
        batch_size=1,
        verbose=0,
    )

    model.learn(total_timesteps=4, callback=CallbackList([scheduler]))

    assert math.isclose(model.ent_coef, 0.0, rel_tol=1e-6)
