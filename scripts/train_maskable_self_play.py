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
from typing import Callable, Dict, List

import numpy as np
from sb3_contrib.common.maskable import distributions as mask_distributions
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.vec_env import DummyVecEnv

from rlohhell.envs.ohhell import OhHellEnv2
from rlohhell.policies import MaskableLstmPolicy, StudentPokerActorCriticPolicy
from rlohhell.utils.opponents import (
    ModelOpponent,
    OpponentPolicy,
    OpponentPool,
    default_opponents,
)


def random_bot(_, action_mask: np.ndarray, __: int, ___=None) -> int:
    legal = np.flatnonzero(action_mask)
    return int(np.random.choice(legal))


def _patch_maskable_distribution() -> None:
    """Clamp MaskableCategorical logits to avoid invalid values during training."""

    original_init = mask_distributions.MaskableCategorical.__init__

    def _safe_init(self, probs=None, logits=None, validate_args=None):  # type: ignore[override]
        if logits is not None:
            logits = logits.nan_to_num()
        return original_init(self, probs=probs, logits=logits, validate_args=validate_args)

    mask_distributions.MaskableCategorical.__init__ = _safe_init


# Apply the patch on import so that short training runs and tests remain stable.
_patch_maskable_distribution()


class SharedPolicyOpponent:
    """Shares the learner's policy for opponent decisions."""

    def __init__(self) -> None:
        self.model: MaskablePPO | PPO | None = None

    def set_model(self, model) -> None:
        self.model = model

    def __call__(self, obs_dict, action_mask: np.ndarray, _: int, __=None) -> int:
        if self.model is None:
            return random_bot(obs_dict, action_mask, _)
        action, _ = self.model.predict(obs_dict, deterministic=False, action_masks=action_mask)
        return int(action)


def build_env(opponent_pool: OpponentPool, seed: int, opponent_selector=None):
    env = OhHellEnv2(
        num_players=4,
        agent_id=0,
        opponent_pool=opponent_pool,
        opponent_selector=opponent_selector,
    )
    env.seed(seed)
    return env


def evaluate_match(
    agent: OpponentPolicy,
    opponent: OpponentPolicy,
    env_builder: Callable[..., OhHellEnv2],
    episodes: int,
    pool: OpponentPool,
    seed_offset: int = 10_000,
):
    wins: List[float] = []
    scores: List[float] = []
    per_round_scores: List[float] = []

    for idx in range(episodes):
        selector = (
            lambda num_players, agent_id, _: {
                pid: opponent for pid in range(num_players) if pid != agent_id
            }
        )
        env = env_builder(opponent_pool=pool, seed=seed_offset + idx, opponent_selector=selector)
        obs, info = env.reset()
        done = False
        while not done:
            mask = info["action_mask"]
            state = env.game.get_state(env.agent_id)
            proposed = agent.act(state, mask, obs, env.game)
            if isinstance(proposed, int):
                action_id = proposed
            else:
                legal_ids = list(env._get_legal_actions())
                matches = [
                    lid
                    for lid in legal_ids
                    if env._decode_action(lid, state) == proposed
                ]
                action_id = matches[0] if matches else legal_ids[0]
            obs, reward, done, _, info = env.step(int(action_id))

        payoffs = env.game.get_payoffs()
        max_score = max(payoffs)
        wins.append(1.0 if payoffs[env.agent_id] >= max_score else 0.0)
        scores.append(payoffs[env.agent_id])
        per_round_scores.append(payoffs[env.agent_id] / env.game.max_rounds)

    return float(np.mean(wins)), float(np.mean(scores)), float(np.mean(per_round_scores))


class TournamentLogger(BaseCallback):
    """Compute win-rates, per-round scores, cross-play matrix and Elo."""

    def __init__(
        self,
        pool: OpponentPool,
        env_builder: Callable[..., OhHellEnv2],
        eval_episodes: int,
        log_dir: str,
        eval_freq: int = 250_000,
    ) -> None:
        super().__init__(verbose=0)
        self.pool = pool
        self.eval_episodes = eval_episodes
        self.env_builder = env_builder
        self.eval_freq = eval_freq
        self.log_dir = log_dir
        self.elo: Dict[str, float] = {}

    def _init_callback(self) -> None:
        os.makedirs(self.log_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        agents = [
            ModelOpponent(
                name="learner",
                model=self.model,
                use_masks=isinstance(self.model, MaskablePPO),
                deterministic=False,
            )
        ] + self.pool.policies()

        matrix = np.zeros((len(agents), len(agents)))
        win_rates: Dict[str, float] = {}
        per_round: Dict[str, float] = {}

        for i, agent in enumerate(agents):
            agent_wins: List[float] = []
            agent_points: List[float] = []
            for j, opponent in enumerate(agents):
                win, points, per_round_score = evaluate_match(
                    agent,
                    opponent,
                    self.env_builder,
                    self.eval_episodes,
                    self.pool,
                    seed_offset=self.num_timesteps + j,
                )
                matrix[i, j] = per_round_score
                agent_wins.append(win)
                agent_points.append(per_round_score)
                if i < j:
                    self._update_elo(agent.name, opponent.name, win)

            win_rates[agent.name] = float(np.mean(agent_wins))
            per_round[agent.name] = float(np.mean(agent_points))

        learner_win = win_rates.get("learner", 0.0)
        learner_points = per_round.get("learner", 0.0)
        self.logger.record("tournament/win_rate", learner_win)
        self.logger.record("tournament/points_per_round", learner_points)
        self.logger.record("tournament/elo", self.elo.get("learner", 1000.0))
        self._dump_cross_play(matrix, [agent.name for agent in agents])
        return True

    def _dump_cross_play(self, matrix: np.ndarray, labels: List[str]) -> None:
        path = os.path.join(self.log_dir, f"cross_play_{self.num_timesteps}.npz")
        np.savez(path, matrix=matrix, labels=np.array(labels))

    def _update_elo(self, player: str, opponent: str, score: float, k: float = 24.0) -> None:
        if player == opponent:
            return
        self.elo.setdefault(player, 1000.0)
        self.elo.setdefault(opponent, 1000.0)
        ra = self.elo[player]
        rb = self.elo[opponent]
        expected = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
        self.elo[player] = ra + k * (score - expected)
        self.elo[opponent] = rb + k * ((1.0 - score) - (1.0 - expected))


class SnapshotCallback(BaseCallback):
    """Periodically clone the learner into the opponent pool."""

    def __init__(self, pool: OpponentPool, save_dir: str, snapshot_freq: int) -> None:
        super().__init__(verbose=0)
        self.pool = pool
        self.save_dir = save_dir
        self.snapshot_freq = snapshot_freq
        self._last_snapshot = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_snapshot < self.snapshot_freq:
            return True
        self.pool.snapshot_model(self.model, self.num_timesteps, self.save_dir)
        self._last_snapshot = self.num_timesteps
        return True


class EntropyScheduler(BaseCallback):
    """Linearly anneal the entropy coefficient for algorithms that expect floats."""

    def __init__(self, start: float, end: float, total_timesteps: int) -> None:
        super().__init__(verbose=0)
        self.start = start
        self.end = end
        self.total_timesteps = max(1, total_timesteps)

    def _on_step(self) -> bool:
        progress = min(1.0, self.num_timesteps / self.total_timesteps)
        self.model.ent_coef = self.start + progress * (self.end - self.start)
        return True


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
    parser.add_argument(
        "--use-student-poker",
        action="store_true",
        help="Train the StudentPoker backbone with maskable self-play",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to a saved checkpoint (.zip) to resume training from",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    _patch_maskable_distribution()
    os.makedirs(args.log_dir, exist_ok=True)

    opponent_pool = OpponentPool(default_opponents(), seed=0)

    train_env = make_vec_env(
        lambda: build_env(opponent_pool=opponent_pool, seed=0),
        n_envs=args.n_envs,
        vec_env_cls=DummyVecEnv,
    )

    eval_env = make_vec_env(
        lambda: build_env(opponent_pool=opponent_pool, seed=123),
        n_envs=4,
        vec_env_cls=DummyVecEnv,
    )

    ent_schedule = get_linear_fn(args.ent_coef, args.final_ent_coef, 1.0)
    policy_cls: str | type = "MultiInputPolicy"
    if args.use_student_poker:
        policy_cls = StudentPokerActorCriticPolicy
    elif args.use_lstm_mask:
        policy_cls = MaskableLstmPolicy

    if args.resume_from:
        loader = PPO.load if args.use_recurrent and not args.use_lstm_mask else MaskablePPO.load
        model = loader(args.resume_from, env=train_env, tensorboard_log=args.log_dir)
        entropy_scheduler: BaseCallback | None = None
    elif args.use_recurrent and not args.use_lstm_mask and not args.use_student_poker:
        model = PPO(
            "MlpLstmPolicy",
            train_env,
            ent_coef=ent_schedule,
            verbose=1,
            tensorboard_log=args.log_dir,
        )
        entropy_scheduler = None
    else:
        entropy_scheduler = None
        if args.ent_coef != args.final_ent_coef:
            entropy_scheduler = EntropyScheduler(
                start=args.ent_coef, end=args.final_ent_coef, total_timesteps=args.total_timesteps
            )
        model = MaskablePPO(
            policy_cls,
            train_env,
            ent_coef=args.ent_coef,
            verbose=1,
            tensorboard_log=args.log_dir,
        )

    opponent_pool.add_opponent(
        ModelOpponent("current_policy", model, use_masks=isinstance(model, MaskablePPO), deterministic=False)
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq // args.n_envs,
        save_path=os.path.join(args.log_dir, "checkpoints"),
        name_prefix="maskable_ppo",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    snapshot_callback = SnapshotCallback(
        pool=opponent_pool,
        save_dir=os.path.join(args.log_dir, "snapshots"),
        snapshot_freq=args.eval_freq,
    )

    tournament_callback = TournamentLogger(
        pool=opponent_pool,
        env_builder=lambda **kwargs: build_env(**kwargs),
        eval_episodes=4,
        log_dir=os.path.join(args.log_dir, "tournament"),
        eval_freq=args.eval_freq // args.n_envs,
    )

    eval_callback = MaskableEvalCallback(
        eval_env,
        best_model_save_path=os.path.join(args.log_dir, "best_model"),
        log_path=os.path.join(args.log_dir, "eval_logs"),
        eval_freq=args.eval_freq // args.n_envs,
        n_eval_episodes=8,
        deterministic=True,
    )

    callbacks = [checkpoint_callback, eval_callback, snapshot_callback, tournament_callback]
    if entropy_scheduler is not None:
        callbacks.append(entropy_scheduler)

    callback = CallbackList(callbacks)
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callback,
        reset_num_timesteps=not bool(args.resume_from),
    )
    model.save(os.path.join(args.log_dir, "final_model"))


if __name__ == "__main__":
    main()
