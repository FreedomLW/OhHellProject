"""Core evaluation helpers for evolutionary search."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from rlohhell.envs.ohhell import OhHellEnv2
from rlohhell.heuristics.param_bot import ParamVector
from rlohhell.utils.opponents import OpponentPolicy, OpponentPool, make_param_opponent


def fixed_seeds(batch_size: int, base: int = 12345) -> List[int]:
    """Return a deterministic list of seeds for batch evaluations."""

    return [base + offset for offset in range(batch_size)]


def build_env_fn(
    pool: OpponentPool, seed: int, opponent_selector: Optional[Callable] = None
) -> OhHellEnv2:
    """Create a configured :class:`OhHellEnv2` instance.

    The environment always treats player ``0`` as the learning agent and draws
    opponents either from ``opponent_selector`` or ``pool``.
    """

    env = OhHellEnv2(
        num_players=4,
        agent_id=0,
        opponent_pool=pool,
        opponent_selector=opponent_selector,
    )
    env.seed(seed)
    return env


def _choose_action(agent, env: OhHellEnv2, state, action_mask, obs):
    if hasattr(agent, "act"):
        chosen = agent.act(state, action_mask, obs, env.game)
    else:
        try:
            chosen = agent(obs, action_mask, env.agent_id, state)
        except TypeError:
            chosen = agent(obs, action_mask, env.agent_id)

    if isinstance(chosen, int):
        return int(chosen)

    legal_ids = list(env._get_legal_actions())
    matches = [lid for lid in legal_ids if env._decode_action(lid, state) == chosen]
    return int(matches[0] if matches else legal_ids[0])


def _selector_for_opponent(opponent: OpponentPolicy | Callable):
    def selector(num_players: int, agent_id: int, _pool: OpponentPool) -> Dict[int, object]:
        return {pid: opponent for pid in range(num_players) if pid != agent_id}

    return selector


def _bid_stats(bids_history: Sequence[Sequence[int]], agent_id: int) -> Mapping[str, float]:
    bids = [round_bids[agent_id] for round_bids in bids_history if len(round_bids) > agent_id]
    if not bids:
        return {"bid_zero_rate": 0.0, "bid_one_rate": 0.0}
    total = len(bids)
    return {
        "bid_zero_rate": float(sum(b == 0 for b in bids) / total),
        "bid_one_rate": float(sum(b == 1 for b in bids) / total),
    }


def play_episode(
    agent: OpponentPolicy,
    opponents: OpponentPolicy | Callable | None,
    seed: int,
    pool: OpponentPool,
) -> Dict[str, float]:
    """Play a single episode against fixed opponents and return metrics."""

    selector = None if opponents is None else _selector_for_opponent(opponents)
    env = build_env_fn(pool, seed, opponent_selector=selector)

    obs, info = env.reset(seed=seed)
    done = False
    while not done:
        action_mask = info["action_mask"]
        state = env.game.get_state(env.agent_id)
        action_id = _choose_action(agent, env, state, action_mask, obs)
        obs, _, done, _, info = env.step(int(action_id))

    payoffs = env.game.get_payoffs()
    agent_score = float(payoffs[env.agent_id])
    metrics: Dict[str, float] = {
        "win": 1.0 if agent_score >= max(payoffs) else 0.0,
        "points": agent_score,
        "points_per_round": agent_score / max(1, env.game.max_rounds),
    }

    bids_history = getattr(env.game, "bids_history", [])
    if bids_history:
        metrics.update(_bid_stats(bids_history, env.agent_id))
    else:
        metrics.update({"bid_zero_rate": 0.0, "bid_one_rate": 0.0})

    return metrics


def _episode_worker(
    theta: ParamVector, opponent: OpponentPolicy | Callable, seed: int, pool: OpponentPool
) -> Dict[str, float]:
    agent = make_param_opponent(theta)
    return play_episode(agent=agent, opponents=opponent, seed=seed, pool=pool)


def _episode_worker_from_args(args):
    return _episode_worker(*args)


def _aggregate_metrics(results: Iterable[Mapping[str, float]]) -> Dict[str, float]:
    totals: Dict[str, List[float]] = {}
    for entry in results:
        for key, value in entry.items():
            totals.setdefault(key, []).append(float(value))

    return {key: float(np.mean(values)) for key, values in totals.items() if values}


def evaluate_theta(
    theta: ParamVector,
    opponents: List[OpponentPolicy],
    seeds: List[int],
    pool: OpponentPool,
    n_jobs: int = 8,
) -> Dict[str, float]:
    """Evaluate ``theta`` against opponents on fixed seeds in parallel."""

    tasks = [(theta, opponent, seed, pool) for seed in seeds for opponent in opponents]
    if not tasks:
        return {}

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        results = list(executor.map(_episode_worker_from_args, tasks))

    return _aggregate_metrics(results)
