import random

import numpy as np
import pytest

sb3 = pytest.importorskip("stable_baselines3")
PPO = sb3.PPO

from rlohhell.envs.ohhell import OhHellEnv2


def test_reset_and_step_returns_valid_shapes():
    env = OhHellEnv2()
    obs, _ = env.reset()
    assert set(obs.keys()) == {"observation", "action_mask"}
    assert obs["observation"].shape == (env.obs_size,)
    assert obs["action_mask"].shape == (env.MAX_ACTIONS,)

    action_mask = env._get_legal_actions()
    legal_actions = list(action_mask)
    action = random.choice(legal_actions)

    next_obs, reward, terminated, truncated, info = env.step(action)
    assert set(next_obs.keys()) == {"observation", "action_mask"}
    assert next_obs["observation"].shape == (env.obs_size,)
    assert next_obs["action_mask"].shape == (env.MAX_ACTIONS,)
    assert isinstance(reward, float)
    assert "legal_actions" in info
    assert "action_mask" in info
    assert info["action_mask"].dtype == np.bool_
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_short_training_loop_executes():
    env = OhHellEnv2()
    model = PPO("MultiInputPolicy", env, verbose=0, n_steps=16, batch_size=16)
    model.learn(total_timesteps=128)

    obs, _ = env.reset()
    action, _ = model.predict(obs, deterministic=True)
    assert not np.isnan(action)
