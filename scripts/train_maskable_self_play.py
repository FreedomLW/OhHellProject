"""Self-play training entry point for Oh Hell! using SB3/MaskablePPO.

Key defaults:
- 4-player self-play
- 32 parallel environments
- 10M timesteps
- entropy annealing
- checkpointing and masked evaluation vs fixed bots
"""

import argparse
import os
from typing import Callable

import numpy as np
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.vec_env import DummyVecEnv

from rlohhell.envs.ohhell import OhHellEnv2
from rlohhell.policies import MaskableLstmPolicy


def random_bot(_, action_mask: np.ndarray, __: int) -> int:
    legal = np.flatnonzero(action_mask)
    return int(np.random.choice(legal))


class SharedPolicyOpponent:
    """Shares the learner's policy for opponent decisions."""

    def __init__(self) -> None:
        self.model: MaskablePPO | PPO | None = None

    def set_model(self, model) -> None:
        self.model = model

    def __call__(self, obs_dict, action_mask: np.ndarray, _: int) -> int:
        if self.model is None:
            return random_bot(obs_dict, action_mask, _)
        action, _ = self.model.predict(obs_dict, deterministic=False, action_masks=action_mask)
        return int(action)


def build_env(opponent_policy: Callable, seed: int):
    env = OhHellEnv2(num_players=4, agent_id=0, opponent_policy=opponent_policy)
    env.seed(seed)
    return env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-play MaskablePPO training")
    parser.add_argument("--total-timesteps", type=int, default=10_000_000)
    parser.add_argument("--n-envs", type=int, default=32)
    parser.add_argument("--log-dir", type=str, default="./runs/maskable_ppo")
    parser.add_argument("--checkpoint-freq", type=int, default=100_000)
    parser.add_argument("--eval-freq", type=int, default=250_000)
    parser.add_argument("--ent-coef", type=float, default=0.02)
    parser.add_argument("--final-ent-coef", type=float, default=0.0)
    parser.add_argument(
        "--use-recurrent",
        action="store_true",
        help="Use RecurrentPPO when masks are not required",
    )
    parser.add_argument(
        "--use-lstm-mask",
        action="store_true",
        help="Train a maskable LSTM policy instead of the default MLP",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.log_dir, exist_ok=True)

    opponent = SharedPolicyOpponent()

    train_env = make_vec_env(
        lambda: build_env(opponent_policy=opponent, seed=0),
        n_envs=args.n_envs,
        vec_env_cls=DummyVecEnv,
    )

    eval_env = make_vec_env(
        lambda: build_env(opponent_policy=random_bot, seed=123),
        n_envs=4,
        vec_env_cls=DummyVecEnv,
    )

    ent_schedule = get_linear_fn(args.ent_coef, args.final_ent_coef, 1.0)

    if args.use_recurrent and not args.use_lstm_mask:
        model = PPO(
            "MlpLstmPolicy",
            train_env,
            ent_coef=ent_schedule,
            verbose=1,
            tensorboard_log=args.log_dir,
        )
    else:
        policy = MaskableLstmPolicy if args.use_lstm_mask else "MultiInputPolicy"
        model = MaskablePPO(
            policy,
            train_env,
            ent_coef=ent_schedule,
            verbose=1,
            tensorboard_log=args.log_dir,
        )

    opponent.set_model(model)

    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq // args.n_envs,
        save_path=os.path.join(args.log_dir, "checkpoints"),
        name_prefix="maskable_ppo",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    eval_callback = MaskableEvalCallback(
        eval_env,
        best_model_save_path=os.path.join(args.log_dir, "best_model"),
        log_path=os.path.join(args.log_dir, "eval_logs"),
        eval_freq=args.eval_freq // args.n_envs,
        n_eval_episodes=8,
        deterministic=True,
    )

    callback = CallbackList([checkpoint_callback, eval_callback])
    model.learn(total_timesteps=args.total_timesteps, callback=callback)
    model.save(os.path.join(args.log_dir, "final_model"))


if __name__ == "__main__":
    main()
