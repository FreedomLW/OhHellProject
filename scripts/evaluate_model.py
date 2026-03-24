"""Evaluate an Oh Hell model checkpoint against a portfolio of opponents.

Plays N full games against each opponent type and reports win_rate,
avg_score, and avg_per_round_score.  Output is a JSON file suitable for
the experiment runner.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
from sb3_contrib.ppo_mask import MaskablePPO

from rlohhell.envs.ohhell import OhHellEnv2, mask_from_state
from rlohhell.games.ohhell.strategies import (
    ConservativeStrategy,
    GreedyStrategy,
    HeuristicStrategy,
    RandomStrategy,
)
from rlohhell.heuristics.bruteforce_agent import BruteforceStrategy
from rlohhell.utils.opponents import (
    ModelOpponent,
    OpponentPool,
    StrategyOpponent,
)

OPPONENT_REGISTRY = {
    "random": lambda: StrategyOpponent("random", RandomStrategy()),
    "greedy": lambda: StrategyOpponent("greedy", GreedyStrategy()),
    "conservative": lambda: StrategyOpponent("conservative", ConservativeStrategy()),
    "heuristic": lambda: StrategyOpponent("heuristic", HeuristicStrategy()),
    "bruteforce": lambda: StrategyOpponent("bruteforce", BruteforceStrategy()),
}


def evaluate_vs_opponent(
    model: MaskablePPO,
    opponent,
    episodes: int,
    seed_offset: int = 10_000,
) -> Dict[str, float]:
    """Play *episodes* full games and return aggregate metrics."""
    wins, scores, per_round = [], [], []

    for idx in range(episodes):
        pool = OpponentPool(opponents=[opponent], seed=idx)
        selector = lambda np_, aid, _pool, opp=opponent: {
            pid: opp for pid in range(np_) if pid != aid
        }
        env = OhHellEnv2(
            num_players=4,
            agent_id=0,
            opponent_pool=pool,
            opponent_selector=selector,
        )
        obs, info = env.reset(seed=seed_offset + idx)
        done = False
        while not done:
            mask = info["action_mask"]
            action, _ = model.predict(obs, deterministic=True, action_masks=mask)
            obs, reward, done, _, info = env.step(int(action))

        payoffs = env.game.get_payoffs()
        agent_score = payoffs[env.agent_id]
        max_score = max(payoffs)
        wins.append(1.0 if agent_score >= max_score else 0.0)
        scores.append(float(agent_score))
        per_round.append(float(agent_score) / max(1, env.game.max_rounds))

    return {
        "win_rate": float(np.mean(wins)),
        "avg_score": float(np.mean(scores)),
        "avg_per_round_score": float(np.mean(per_round)),
        "std_score": float(np.std(scores)),
        "episodes": episodes,
    }


def evaluate_vs_baselines(
    model: MaskablePPO,
    baseline_paths: list[str],
    episodes: int = 1000,
    seed_offset: int = 30_000,
) -> Dict[str, float]:
    """Play *episodes* full games against randomly sampled baseline models."""
    import random as _rng
    baselines = []
    for p in baseline_paths:
        m = MaskablePPO.load(p, device="cpu")
        baselines.append(ModelOpponent(os.path.basename(p), m, use_masks=True, deterministic=False))

    wins, scores, per_round = [], [], []
    rng = _rng.Random(42)

    for idx in range(episodes):
        opp = rng.choice(baselines)
        pool = OpponentPool(opponents=[opp], seed=idx)
        selector = lambda np_, aid, _pool, o=opp: {
            pid: o for pid in range(np_) if pid != aid
        }
        env = OhHellEnv2(num_players=4, agent_id=0, opponent_pool=pool, opponent_selector=selector)
        obs, info = env.reset(seed=seed_offset + idx)
        done = False
        while not done:
            mask = info["action_mask"]
            action, _ = model.predict(obs, deterministic=True, action_masks=mask)
            obs, reward, done, _, info = env.step(int(action))

        payoffs = env.game.get_payoffs()
        agent_score = payoffs[env.agent_id]
        max_score = max(payoffs)
        wins.append(1.0 if agent_score >= max_score else 0.0)
        scores.append(float(agent_score))
        per_round.append(float(agent_score) / max(1, env.game.max_rounds))

        if (idx + 1) % 100 == 0:
            print(f"  [{idx+1}/{episodes}] wr={np.mean(wins):.3f} score={np.mean(scores):.1f}")

    return {
        "win_rate": float(np.mean(wins)),
        "avg_score": float(np.mean(scores)),
        "avg_per_round_score": float(np.mean(per_round)),
        "std_score": float(np.std(scores)),
        "episodes": episodes,
    }


BASELINE_MODELS = [
    "models/bc_hybrid_stage3.zip",
    "models/rl_finetune_v1.zip",
    "models/rl_chain_v3.zip",
]


def main():
    parser = argparse.ArgumentParser(description="Evaluate an Oh Hell model checkpoint")
    parser.add_argument("--model", required=True, help="Path to SB3 .zip checkpoint")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument(
        "--opponents",
        type=str,
        default="random,greedy,conservative,heuristic",
    )
    parser.add_argument("--self-play-episodes", type=int, default=0)
    parser.add_argument(
        "--vs-baselines",
        type=int,
        default=0,
        help="Number of games vs baseline models (1000 recommended)",
    )
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    model = MaskablePPO.load(args.model, device="cpu")
    opponent_names = [n.strip() for n in args.opponents.split(",")]

    results: Dict[str, dict] = {}
    for name in opponent_names:
        if name not in OPPONENT_REGISTRY:
            print(f"Warning: unknown opponent '{name}', skipping")
            continue
        t0 = time.monotonic()
        n_eps = min(args.episodes, 10) if name == "bruteforce" else args.episodes
        opponent = OPPONENT_REGISTRY[name]()
        metrics = evaluate_vs_opponent(model, opponent, n_eps)
        metrics["wall_seconds"] = round(time.monotonic() - t0, 2)
        results[f"vs_{name}"] = metrics
        print(
            f"vs {name}: win_rate={metrics['win_rate']:.3f}  "
            f"avg_score={metrics['avg_score']:.1f}  "
            f"per_round={metrics['avg_per_round_score']:.3f}  "
            f"({metrics['wall_seconds']:.1f}s)"
        )

    if args.self_play_episodes > 0:
        t0 = time.monotonic()
        self_opp = ModelOpponent("self", model, use_masks=True, deterministic=True)
        self_metrics = evaluate_vs_opponent(model, self_opp, args.self_play_episodes)
        self_metrics["wall_seconds"] = round(time.monotonic() - t0, 2)
        results["vs_self"] = self_metrics
        print(
            f"vs self: win_rate={self_metrics['win_rate']:.3f}  "
            f"avg_score={self_metrics['avg_score']:.1f}"
        )

    if args.vs_baselines > 0:
        existing = [p for p in BASELINE_MODELS if os.path.exists(p)]
        if existing:
            t0 = time.monotonic()
            print(f"\nEvaluating {args.vs_baselines} games vs baselines: {existing}")
            bl_metrics = evaluate_vs_baselines(model, existing, args.vs_baselines)
            bl_metrics["wall_seconds"] = round(time.monotonic() - t0, 2)
            results["vs_baselines"] = bl_metrics
            print(
                f"\nvs baselines ({args.vs_baselines} games): "
                f"win_rate={bl_metrics['win_rate']:.3f}  "
                f"avg_score={bl_metrics['avg_score']:.1f}  "
                f"({bl_metrics['wall_seconds']:.1f}s)"
            )

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved to {args.output}")
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
