#!/usr/bin/env python3
"""Teach a student policy using iterative oracle bootstrapping and parameter search."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rlohhell.analysis.curriculum_eval import find_optimal_oracle_params, run_oracle_bootstrap


def _csv_ints(raw: str) -> list[int]:
    values = [int(v.strip()) for v in raw.split(",") if v.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected comma-separated integers")
    return values


def _build_seed_list(base_seed: int, num_seeds: int, explicit: list[int] | None) -> list[int]:
    if explicit is not None:
        return explicit
    if num_seeds <= 0:
        raise ValueError("--num-seeds must be > 0")
    return [base_seed + i for i in range(num_seeds)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0, help="Base scenario seed")
    parser.add_argument("--num-seeds", type=int, default=1, help="How many consecutive seeds to use from --seed")
    parser.add_argument(
        "--seeds",
        type=_csv_ints,
        default=None,
        help="Optional explicit comma-separated seeds (overrides --seed/--num-seeds)",
    )
    parser.add_argument("--round-sizes", type=_csv_ints, default=[1, 2], help="Comma-separated one-round sizes")
    parser.add_argument("--target-seat", type=int, default=0, help="Seat that gets taught")
    parser.add_argument("--search", action="store_true", help="Run hyperparameter search for oracle-teaching settings")
    parser.add_argument("--iterations", type=int, default=3, help="Iterations for fixed-parameter run")
    parser.add_argument("--rollouts-per-action", type=int, default=4, help="Oracle rollouts per legal action")
    parser.add_argument("--hand-samples-per-seed", type=int, default=1, help="How many different hand deals to sample per seed")
    parser.add_argument("--include-bid-phase", action="store_true", help="Also learn oracle decisions in bidding phase")
    parser.add_argument(
        "--iterations-candidates",
        type=_csv_ints,
        default=[2, 3],
        help="Comma-separated iteration counts considered in --search mode",
    )
    parser.add_argument(
        "--rollouts-candidates",
        type=_csv_ints,
        default=[2, 4],
        help="Comma-separated rollout counts considered in --search mode",
    )
    parser.add_argument("--output-json", default="", help="Optional output file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = _build_seed_list(base_seed=args.seed, num_seeds=args.num_seeds, explicit=args.seeds)

    if args.search:
        best = find_optimal_oracle_params(
            seeds=seeds,
            round_sizes=args.round_sizes,
            iterations_candidates=args.iterations_candidates,
            rollout_candidates=args.rollouts_candidates,
            target_seat=args.target_seat,
            hand_samples_per_seed=args.hand_samples_per_seed,
            play_phase_only=not args.include_bid_phase,
        )
        rendered = best.to_json()
    else:
        report = run_oracle_bootstrap(
            seeds=seeds,
            round_sizes=args.round_sizes,
            iterations=args.iterations,
            rollouts_per_action=args.rollouts_per_action,
            target_seat=args.target_seat,
            hand_samples_per_seed=args.hand_samples_per_seed,
            play_phase_only=not args.include_bid_phase,
        )
        rendered = report.to_json()

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as fh:
            fh.write(rendered + "\n")

    print(rendered)


if __name__ == "__main__":
    main()
