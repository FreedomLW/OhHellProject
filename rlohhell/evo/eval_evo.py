"""Evaluate parametric Oh Hell agents against built-in and hall-of-fame opponents."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from sb3_contrib.ppo_mask import MaskablePPO

from rlohhell.evo.eval_core import build_env_fn, fixed_seeds
from rlohhell.evo.hof import HallOfFame, theta_opponent
from rlohhell.heuristics.param_bot import ParamVector
from rlohhell.utils.opponents import (
    ModelOpponent,
    OpponentPolicy,
    OpponentPool,
    default_opponents,
    make_param_opponent,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--theta-json", type=str, required=True, help="Path to theta JSON file")
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--log-dir", type=str, default="./runs/evo_eval")
    parser.add_argument("--hof-top-k", type=int, default=6, help="Number of HoF opponents to sample")
    parser.add_argument("--checkpoint", action="append", default=[], help="Extra SB3 checkpoints to include")
    parser.add_argument("--seed", type=int, default=12345)
    return parser.parse_args(argv)


def _load_theta(path: str) -> ParamVector:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return ParamVector(**data)


def _load_checkpoint(path: str) -> BaseAlgorithm:
    try:
        return MaskablePPO.load(path, device="cpu")
    except Exception:
        return PPO.load(path, device="cpu")


def _select_opponents(theta_path: str, hof_top_k: int, checkpoints: List[str]) -> List[OpponentPolicy]:
    opponents: List[OpponentPolicy] = []
    opponents.extend(default_opponents())

    hof_path = os.path.join(os.path.dirname(os.path.abspath(theta_path)), "hof.json")
    if os.path.exists(hof_path):
        hof = HallOfFame.load(hof_path)
        for theta, _score, step in hof.top(hof_top_k):
            opponents.append(theta_opponent(theta, name=f"hof_{step}"))

    for ckpt in checkpoints:
        try:
            model = _load_checkpoint(ckpt)
            opponents.append(
                ModelOpponent(
                    name=os.path.splitext(os.path.basename(ckpt))[0],
                    model=model,
                    use_masks=isinstance(model, MaskablePPO),
                    deterministic=True,
                )
            )
        except Exception as exc:  # pragma: no cover - defensive
            print(f"Failed to load checkpoint {ckpt}: {exc}")

    return opponents


def _collect_agent_bids(bids_history: Sequence[Sequence[int]], agent_id: int) -> List[int]:
    bids: List[int] = []
    for round_bids in bids_history:
        if len(round_bids) > agent_id:
            try:
                bids.append(int(round_bids[agent_id]))
            except (TypeError, ValueError):
                continue
    return bids


def _choose_action(agent, env, state, action_mask, obs):
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


def _play_episode(agent: OpponentPolicy, opponent: OpponentPolicy, seed: int, pool: OpponentPool) -> Dict[str, object]:
    env = build_env_fn(pool, seed, opponent_selector=lambda n, a, _p: {pid: opponent for pid in range(n) if pid != a})
    obs, info = env.reset(seed=seed)
    done = False
    while not done:
        action_mask = info["action_mask"]
        state = env.game.get_state(env.agent_id)
        action_id = _choose_action(agent, env, state, action_mask, obs)
        obs, _, done, _, info = env.step(int(action_id))

    payoffs = env.game.get_payoffs()
    score = float(payoffs[env.agent_id])
    bids_history = getattr(env.game, "bids_history", [])
    return {
        "win": 1.0 if score >= max(payoffs) else 0.0,
        "points": score,
        "points_per_round": score / max(1, env.game.max_rounds),
        "bids": _collect_agent_bids(bids_history, env.agent_id),
    }


def _episode_worker(args):
    theta, opponent, seed, pool = args
    agent = make_param_opponent(theta)
    return _play_episode(agent, opponent, seed, pool)


def _aggregate(results: Iterable[Mapping[str, object]]) -> Dict[str, float]:
    episodes = list(results)
    if not episodes:
        return {}

    wins = [float(ep.get("win", 0.0)) for ep in episodes]
    points = [float(ep.get("points", 0.0)) for ep in episodes]
    ppr = [float(ep.get("points_per_round", 0.0)) for ep in episodes]
    bids: List[int] = [bid for ep in episodes for bid in ep.get("bids", [])]

    dist = Counter(bids)
    total_bids = sum(dist.values())
    bid_distribution = {
        f"bid_{bid}_share": (count / total_bids if total_bids else 0.0)
        for bid, count in sorted(dist.items())
    }

    metrics: Dict[str, float] = {
        "episodes": float(len(episodes)),
        "win_rate": float(np.mean(wins)),
        "avg_points": float(np.mean(points)),
        "points_per_round": float(np.mean(ppr)),
        "total_bids": float(total_bids),
    }
    metrics.update(bid_distribution)
    return metrics


def evaluate_opponent(
    theta: ParamVector,
    opponent: OpponentPolicy,
    seeds: List[int],
    pool: OpponentPool,
    n_jobs: int,
) -> Dict[str, float]:
    tasks = [(theta, opponent, seed, pool) for seed in seeds]
    if isinstance(opponent, ModelOpponent) or n_jobs <= 1:
        results = [_episode_worker(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            results = list(executor.map(_episode_worker, tasks))

    return _aggregate(results)


def _write_csv(rows: List[Dict[str, float]], path: str) -> None:
    if not rows:
        return

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys() if key != "opponent"})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["opponent"] + fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    theta = _load_theta(args.theta_json)
    seeds = fixed_seeds(args.episodes, base=args.seed)

    opponents = _select_opponents(args.theta_json, args.hof_top_k, args.checkpoint)
    pool = OpponentPool(opponents=default_opponents(), seed=args.seed)

    results: List[Dict[str, float]] = []
    for opponent in opponents:
        metrics = evaluate_opponent(theta, opponent, seeds=seeds, pool=pool, n_jobs=args.jobs)
        metrics["opponent"] = opponent.name
        results.append(metrics)
        print(
            f"Opponent={opponent.name} | win_rate={metrics.get('win_rate', 0):.3f} | "
            f"points_per_round={metrics.get('points_per_round', 0):.3f}"
        )

    csv_path = os.path.join(args.log_dir, "eval_metrics.csv")
    _write_csv(results, csv_path)
    print(f"Saved metrics to {csv_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
