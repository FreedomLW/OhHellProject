"""Evolutionary training script for parametric Oh Hell agents."""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import random
import sys
from dataclasses import asdict
from typing import List, Sequence

import numpy as np

from rlohhell.evo.eval_core import evaluate_population, evaluate_theta, fixed_seeds
from rlohhell.evo.hof import HallOfFame, mix_opponents
from rlohhell.evo.optimizers import CEM, _param_bounds, clip_to_bounds, try_cma_es
from rlohhell.heuristics.param_bot import ParamVector, theta_to_vector, vector_to_theta
from rlohhell.utils.opponents import OpponentPool, OpponentPolicy, default_opponents


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algo", choices=["cem", "cma"], default="cem")
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument("--pop-size", type=int, default=64)
    parser.add_argument("--batch-seeds", type=int, default=512)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--log-dir", type=str, default="./runs/evo")
    parser.add_argument("--hof-size", type=int, default=32)
    parser.add_argument("--k-opponents", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=20,
        help="Generations between anchor tournaments and checkpoints.",
    )
    parser.add_argument(
        "--anchor-batch",
        type=int,
        default=256,
        help="Number of episodes for anchor tournament evaluations.",
    )
    return parser.parse_args(argv)


def _init_optimizer(args: argparse.Namespace, mean: np.ndarray, std: np.ndarray):
    if args.algo == "cma":
        cma = try_cma_es(mean, float(np.mean(std)))
        if cma is not None:
            return cma
        print("[train_evo] CMA not available, falling back to CEM", file=sys.stderr)
    return CEM(init_mean=mean, init_std=std, pop_size=args.pop_size)


def _prepare_logging(log_dir: str):
    os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, "metrics.csv")
    csv_file = open(csv_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(
        csv_file,
        fieldnames=[
            "generation",
            "candidate",
            "score",
            "points_per_round",
            "win_rate",
            "bid_zero_rate",
            "bid_one_rate",
        ],
    )
    if csv_file.tell() == 0:
        writer.writeheader()
    try:
        from torch.utils.tensorboard import SummaryWriter

        tb_writer = SummaryWriter(log_dir=log_dir)
    except Exception:  # pragma: no cover - optional dependency
        tb_writer = None
    return csv_file, writer, tb_writer


def _log_distribution(mean: np.ndarray, std: np.ndarray) -> None:
    try:
        from rlohhell.heuristics.param_bot import ParamVector

        names = [f.name for f in ParamVector.__dataclass_fields__.values()]
        stats = ", ".join(f"{name}: μ={m:.3f} σ={s:.3f}" for name, m, s in zip(names, mean, std))
    except Exception:  # pragma: no cover - defensive logging only
        stats = f"mean={mean}, std={std}"
    print(f"[train_evo] Distribution -> {stats}")


def _fitness_from_metrics(metrics: dict) -> float:
    points = float(metrics.get("points_per_round", 0.0))
    win_rate = float(metrics.get("win", 0.0))
    penalty_share = 0.5 * (
        float(metrics.get("bid_zero_rate", 0.0))
        + float(metrics.get("bid_one_rate", 0.0))
    )
    return 0.7 * points + 0.3 * win_rate - 0.05 * penalty_share


def _save_theta(theta: ParamVector, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(theta), f, indent=2)


def _anchor_tournament(
    theta: ParamVector,
    anchors: List[OpponentPolicy],
    seeds: List[int],
    pool: OpponentPool,
    n_jobs: int,
) -> dict:
    return evaluate_theta(theta, opponents=anchors, seeds=seeds, pool=pool, n_jobs=n_jobs)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    np.random.seed(args.seed)
    random.seed(args.seed)
    theta0 = ParamVector()
    mean = theta_to_vector(theta0)
    lo, hi = _param_bounds()
    std = np.maximum((hi - lo) * 0.1, 1e-3)

    opt = _init_optimizer(args, mean=mean, std=std)

    builtin_opponents = default_opponents()
    opponent_pool = OpponentPool(opponents=builtin_opponents, seed=args.seed)
    hof = HallOfFame()

    csv_file, csv_writer, tb_writer = _prepare_logging(args.log_dir)
    best_theta = theta0
    best_score = float("-inf")

    anchor_seeds = fixed_seeds(args.anchor_batch, base=args.seed + 123456)
    checkpoints_dir = os.path.join(args.log_dir, "checkpoints")

    try:
        for gen in range(args.generations):
            seeds = fixed_seeds(args.batch_seeds, base=args.seed + gen * 10000)
            population = np.asarray(opt.ask(), dtype=float)
            population = clip_to_bounds(population, lo, hi)

            opponents = mix_opponents(hof, builtin_opponents, args.k_opponents)

            thetas = [vector_to_theta(vec) for vec in population]
            metrics_batch = evaluate_population(
                thetas, opponents=opponents, seeds=seeds, pool=opponent_pool, n_jobs=args.jobs
            )

            scores = []
            gen_best = None
            for idx, (theta, metrics) in enumerate(zip(thetas, metrics_batch)):
                fitness = _fitness_from_metrics(metrics)
                scores.append(fitness)

                row = {
                    "generation": gen,
                    "candidate": idx,
                    "score": fitness,
                    "points_per_round": metrics.get("points_per_round", 0.0),
                    "win_rate": metrics.get("win", 0.0),
                    "bid_zero_rate": metrics.get("bid_zero_rate", 0.0),
                    "bid_one_rate": metrics.get("bid_one_rate", 0.0),
                }
                csv_writer.writerow(row)
                csv_file.flush()

                if gen_best is None or fitness > gen_best[0]:
                    gen_best = (fitness, theta)

            if not scores:
                print("No candidates evaluated; stopping.", file=sys.stderr)
                break

            opt.tell(np.array(scores, dtype=float))

            avg_score = statistics.mean(scores)
            best_score_gen = max(scores)
            print(
                f"[train_evo] Gen {gen}: best={best_score_gen:.3f}, avg={avg_score:.3f}, "
                f"hof_size={len(hof.entries)}"
            )

            if hasattr(opt, "mean") and hasattr(opt, "std"):
                try:
                    _log_distribution(np.asarray(opt.mean, dtype=float), np.asarray(opt.std, dtype=float))
                except Exception:  # pragma: no cover - logging only
                    pass

            if gen_best is not None:
                gen_score, gen_theta = gen_best
                hof.add(gen_theta, gen_score, gen)
                if len(hof.entries) > args.hof_size:
                    hof.entries = hof.entries[: args.hof_size]

                if gen_score > best_score:
                    best_score = gen_score
                    best_theta = gen_theta

                if tb_writer is not None:
                    tb_writer.add_scalar("fitness/gen_best", gen_score, gen)
                    tb_writer.add_scalar("fitness/best_so_far", best_score, gen)

            if (gen + 1) % args.eval_interval == 0:
                anchor_metrics = _anchor_tournament(
                    best_theta,
                    anchors=builtin_opponents,
                    seeds=anchor_seeds,
                    pool=opponent_pool,
                    n_jobs=args.jobs,
                )
                anchor_score = _fitness_from_metrics(anchor_metrics)
                if tb_writer is not None:
                    for key, value in anchor_metrics.items():
                        tb_writer.add_scalar(f"anchor/{key}", value, gen)
                    tb_writer.add_scalar("anchor/fitness", anchor_score, gen)
                snapshot_path = os.path.join(checkpoints_dir, f"theta_gen_{gen+1}.json")
                _save_theta(best_theta, snapshot_path)

    finally:
        csv_file.close()
        if tb_writer is not None:
            tb_writer.close()

    _save_theta(best_theta, os.path.join(args.log_dir, "best_theta.json"))
    hof.save(os.path.join(args.log_dir, "hof.json"))


if __name__ == "__main__":  # pragma: no cover
    main()
