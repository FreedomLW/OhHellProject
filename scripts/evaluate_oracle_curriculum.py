#!/usr/bin/env python3
"""Run iterative oracle bootstrapping from oracle dataset scenarios."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rlohhell.analysis.curriculum_eval import load_oracle_scenarios, run_oracle_bootstrap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-jsonl", required=True, help="Path to oracle dataset JSONL")
    parser.add_argument("--iterations", type=int, default=3, help="How many oracle->teach cycles to run")
    parser.add_argument("--rollouts-per-action", type=int, default=4, help="Oracle rollouts per legal action")
    parser.add_argument("--output-json", default="", help="Optional report JSON output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds, round_sizes = load_oracle_scenarios(args.dataset_jsonl)
    report = run_oracle_bootstrap(
        seeds=seeds,
        round_sizes=round_sizes,
        iterations=args.iterations,
        rollouts_per_action=args.rollouts_per_action,
    )
    rendered = report.to_json()

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as fh:
            fh.write(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
