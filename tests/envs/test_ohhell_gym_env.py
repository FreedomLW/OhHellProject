import random

import numpy as np
from stable_baselines3 import PPO

from rlohhell.envs.ohhell import OhHellEnv2


def test_reset_and_step_returns_valid_shapes():
    env = OhHellEnv2()
    obs, _ = env.reset()
    assert obs.shape == (env.obs_size,)

    legal_actions = list(env._get_legal_actions())
    action = random.choice(legal_actions)

    next_obs, reward, terminated, truncated, info = env.step(action)
    assert next_obs.shape == (env.obs_size,)
    assert isinstance(reward, float)
    assert "legal_actions" in info
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_short_training_loop_executes():
    env = OhHellEnv2()
    model = PPO("MlpPolicy", env, verbose=0, n_steps=16, batch_size=16)
    model.learn(total_timesteps=128)

    obs, _ = env.reset()
    action, _ = model.predict(obs, deterministic=True)
    assert not np.isnan(action)
