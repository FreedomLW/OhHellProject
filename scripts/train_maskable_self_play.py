"""Self-play training entry point for Oh Hell! using SB3/MaskablePPO.

Key defaults:
- 4-player self-play
- 32 parallel environments
- 10M timesteps
- entropy annealing
- checkpointing and masked evaluation vs fixed bots
- curriculum on hand sizes (1–4, 1–6, then full cycle) with automatic promotion
  once the agent clears a target win rate vs baseline bots
"""

import argparse
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.vec_env import DummyVecEnv

from rlohhell.envs.ohhell import OhHellEnv2
from rlohhell.policies import MaskableLstmPolicy
from rlohhell.utils.opponents import (
    ModelOpponent,
    OpponentPolicy,
    OpponentPool,
    default_opponents,
)


def random_bot(_, action_mask: np.ndarray, __: int, ___=None) -> int:
    legal = np.flatnonzero(action_mask)
    return int(np.random.choice(legal))


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


@dataclass
class TrainingPhase:
    """Configuration for a curriculum phase."""

    name: str
    max_hand_size: Optional[int]
    timesteps: int


def build_env(
    opponent_pool: OpponentPool,
    seed: int,
    opponent_selector=None,
    max_hand_size: Optional[int] = None,
):
    env = OhHellEnv2(
        num_players=4,
        agent_id=0,
        opponent_pool=opponent_pool,
        opponent_selector=opponent_selector,
        max_hand_size=max_hand_size,
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


class PhasePromotionCallback(BaseCallback):
    """Evaluate win-rate against baseline bots and stop the phase early."""

    def __init__(
        self,
        env_builder: Callable[..., OhHellEnv2],
        eval_episodes: int,
        target_win_rate: float,
        eval_freq: int,
        phase_name: str,
        max_hand_size: Optional[int],
    ) -> None:
        super().__init__(verbose=0)
        self.env_builder = env_builder
        self.eval_episodes = eval_episodes
        self.target_win_rate = target_win_rate
        self.eval_freq = max(1, eval_freq)
        self.phase_name = phase_name
        self.max_hand_size = max_hand_size
        self.passed = False
        self.baseline_pool = OpponentPool(default_opponents(), seed=42)

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        agent = ModelOpponent(
            name="learner_eval",
            model=self.model,
            use_masks=isinstance(self.model, MaskablePPO),
            deterministic=True,
        )

        win_rates: List[float] = []
        for idx, opponent in enumerate(self.baseline_pool.policies()):
            win_rate, _, _ = evaluate_match(
                agent,
                opponent,
                self.env_builder,
                self.eval_episodes,
                self.baseline_pool,
                seed_offset=self.num_timesteps + idx,
            )
            win_rates.append(win_rate)

        mean_win_rate = float(np.mean(win_rates)) if win_rates else 0.0
        self.logger.record(f"phase/{self.phase_name}_win_rate", mean_win_rate)

        if mean_win_rate >= self.target_win_rate:
            self.passed = True
            return False

        return True


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
        "--target-win-rate",
        type=float,
        default=0.6,
        help="Mean win rate vs default bots required to advance to the next phase",
    )
    parser.add_argument(
        "--phase-eval-episodes",
        type=int,
        default=12,
        help="Number of evaluation episodes per baseline opponent during curriculum checks",
    )
    parser.add_argument(
        "--phase-eval-freq",
        type=int,
        default=200_000,
        help="How often to run curriculum evaluations (in environment steps)",
    )
    parser.add_argument(
        "--phase-steps",
        type=int,
        default=None,
        help="Timesteps to allocate to each curriculum phase; defaults to an even split",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.log_dir, exist_ok=True)

    opponent_pool = OpponentPool(default_opponents(), seed=0)

    phase_steps = args.phase_steps or max(1, args.total_timesteps // 3)
    remaining_steps = max(args.total_timesteps - (phase_steps * 2), phase_steps)
    phases = [
        TrainingPhase("hand_4", 4, phase_steps),
        TrainingPhase("hand_6", 6, phase_steps),
        TrainingPhase("full_cycle", None, remaining_steps),
    ]

    def make_train_env(phase: TrainingPhase, seed: int):
        return make_vec_env(
            lambda: build_env(
                opponent_pool=opponent_pool, seed=seed, max_hand_size=phase.max_hand_size
            ),
            n_envs=args.n_envs,
            vec_env_cls=DummyVecEnv,
        )

    def make_eval_env(phase: TrainingPhase, seed: int):
        return make_vec_env(
            lambda: build_env(
                opponent_pool=opponent_pool, seed=seed, max_hand_size=phase.max_hand_size
            ),
            n_envs=4,
            vec_env_cls=DummyVecEnv,
        )

    train_env = make_train_env(phases[0], seed=0)
    eval_env = make_eval_env(phases[0], seed=123)

    ent_schedule = get_linear_fn(args.ent_coef, args.final_ent_coef, 1.0)

    if args.use_recurrent and not args.use_lstm_mask:
        model = PPO(
            "MlpLstmPolicy",
            train_env,
            ent_coef=ent_schedule,
            verbose=1,
            tensorboard_log=args.log_dir,
        )
        entropy_scheduler: BaseCallback | None = None
    else:
        policy = MaskableLstmPolicy if args.use_lstm_mask else "MultiInputPolicy"
        entropy_scheduler = None
        if args.ent_coef != args.final_ent_coef:
            entropy_scheduler = EntropyScheduler(
                start=args.ent_coef, end=args.final_ent_coef, total_timesteps=args.total_timesteps
            )
        model = MaskablePPO(
            policy,
            train_env,
            ent_coef=args.ent_coef,
            verbose=1,
            tensorboard_log=args.log_dir,
        )

    opponent_pool.add_opponent(
        ModelOpponent("current_policy", model, use_masks=isinstance(model, MaskablePPO), deterministic=False)
    )

    for idx, phase in enumerate(phases):
        if idx > 0:
            train_env.close()
            eval_env.close()
            train_env = make_train_env(phase, seed=idx)
            eval_env = make_eval_env(phase, seed=123 + idx)
            model.set_env(train_env)

        checkpoint_callback = CheckpointCallback(
            save_freq=max(1, args.checkpoint_freq // args.n_envs),
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
            env_builder=lambda **kwargs: build_env(**kwargs, max_hand_size=phase.max_hand_size),
            eval_episodes=4,
            log_dir=os.path.join(args.log_dir, "tournament"),
            eval_freq=max(1, args.eval_freq // args.n_envs),
        )

        eval_callback = MaskableEvalCallback(
            eval_env,
            best_model_save_path=os.path.join(args.log_dir, "best_model"),
            log_path=os.path.join(args.log_dir, "eval_logs"),
            eval_freq=max(1, args.eval_freq // args.n_envs),
            n_eval_episodes=8,
            deterministic=True,
        )

        promotion_callback = PhasePromotionCallback(
            env_builder=lambda **kwargs: build_env(**kwargs, max_hand_size=phase.max_hand_size),
            eval_episodes=args.phase_eval_episodes,
            target_win_rate=args.target_win_rate,
            eval_freq=max(1, args.phase_eval_freq // args.n_envs),
            phase_name=phase.name,
            max_hand_size=phase.max_hand_size,
        )

        callbacks = [
            checkpoint_callback,
            eval_callback,
            snapshot_callback,
            tournament_callback,
            promotion_callback,
        ]
        if entropy_scheduler is not None:
            callbacks.append(entropy_scheduler)

        callback = CallbackList(callbacks)
        model.learn(total_timesteps=phase.timesteps, callback=callback, reset_num_timesteps=False)

        if promotion_callback.passed:
            print(
                f"Advancing from phase {phase.name} after reaching win rate {args.target_win_rate:.0%}"
            )
        else:
            print(f"Completed phase {phase.name} without meeting the target win rate")

    train_env.close()
    eval_env.close()
    model.save(os.path.join(args.log_dir, "final_model"))


if __name__ == "__main__":
    main()
