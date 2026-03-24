"""Autoresearch-style experiment runner for Oh Hell.

Reads a JSON config, runs ordered phases (data_gen -> bc_train -> rl_train ->
evaluate) as subprocesses with a hard 10-minute wall-clock budget.  Each
experiment is saved to experiments/<id>/ with config, result, log and model.

Usage:
    python scripts/run_experiment.py experiments/configs/bc_small.json
    python scripts/run_experiment.py experiments/configs/bc_small.json --id my_run
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
PYTHON = sys.executable


def generate_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{uuid.uuid4().hex[:6]}"


# ------------------------------------------------------------------
# Phase runners
# ------------------------------------------------------------------

def _run_subprocess(cmd: list[str], timeout: float, cwd: str | None = None, log_file=None) -> dict:
    """Run a command with timeout, streaming output to console and log file."""
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=cwd or str(PROJECT_ROOT),
        )
        deadline = time.monotonic() + max(timeout, 1)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                proc.kill()
                proc.wait()
                return {
                    "returncode": -1,
                    "stdout": "".join(stdout_lines),
                    "stderr": "",
                    "timeout": True,
                }
            line = proc.stdout.readline()
            if not line and proc.poll() is not None:
                break
            if line:
                sys.stdout.write(line)
                sys.stdout.flush()
                stdout_lines.append(line)
                if log_file:
                    log_file.write(line)
                    log_file.flush()
        proc.wait()
        return {
            "returncode": proc.returncode,
            "stdout": "".join(stdout_lines),
            "stderr": "",
        }
    except Exception as e:
        return {
            "returncode": -1,
            "stdout": "".join(stdout_lines),
            "stderr": str(e),
        }


def run_data_gen(params: dict, exp_dir: Path, timeout: float, log_file=None, shared: dict | None = None) -> dict:
    output = str(exp_dir / "bc_data.npz")
    cmd = [
        PYTHON, str(PROJECT_ROOT / "scripts" / "generate_bc_data.py"),
        "--num-seeds", str(params.get("num_seeds", 100)),
        "--round-sizes", str(params.get("round_sizes", "1,2,3,4,5")),
        "--target-seat", str(params.get("target_seat", 0)),
        "--output", output,
    ]
    # Use a trained model as opponent if specified in params or from prior phase
    opponent_model = params.get("opponent_model") or (shared or {}).get("model_path")
    if opponent_model and os.path.exists(opponent_model):
        cmd.extend(["--opponent-model", opponent_model])
    result = _run_subprocess(cmd, timeout, log_file=log_file)
    result["data_path"] = output
    return result


def run_bc_train(params: dict, exp_dir: Path, shared: dict, timeout: float, log_file=None) -> dict:
    data_path = params.get("data_path") or shared.get("data_path") or str(exp_dir / "bc_data.npz")
    output_dir = str(exp_dir / "model")
    cmd = [
        PYTHON, str(PROJECT_ROOT / "scripts" / "train_bc.py"),
        "--data", data_path,
        "--output", output_dir,
        "--epochs", str(params.get("epochs", 50)),
        "--batch-size", str(params.get("batch_size", 256)),
        "--lr", str(params.get("lr", 3e-4)),
        "--device", str(params.get("device", "cpu")),
    ]
    result = _run_subprocess(cmd, timeout, log_file=log_file)
    result["model_path"] = os.path.join(output_dir, "bc_model.zip")
    return result


def run_rl_train(params: dict, exp_dir: Path, shared: dict, timeout: float, log_file=None) -> dict:
    log_dir = str(exp_dir / "rl_logs")
    resume_from = params.get("resume_from") or shared.get("model_path")
    cmd = [
        PYTHON, str(PROJECT_ROOT / "scripts" / "train_maskable_self_play.py"),
        "--total-timesteps", str(params.get("total_timesteps", 100_000)),
        "--n-envs", str(params.get("n_envs", 8)),
        "--log-dir", log_dir,
        "--checkpoint-freq", str(params.get("checkpoint_freq", 50_000)),
        "--eval-freq", str(params.get("eval_freq", 50_000)),
        "--ent-coef", str(params.get("ent_coef", 0.02)),
    ]
    if params.get("final_ent_coef") is not None:
        cmd.extend(["--final-ent-coef", str(params["final_ent_coef"])])
    if resume_from and os.path.exists(resume_from):
        cmd.extend(["--resume-from", resume_from])
    if params.get("use_bruteforce_opponent"):
        cmd.append("--use-bruteforce-opponent")

    result = _run_subprocess(cmd, timeout, log_file=log_file)

    # Find the best available model: final_model > best_model > latest checkpoint
    final_model = os.path.join(log_dir, "final_model.zip")
    best_model = os.path.join(log_dir, "best_model", "best_model.zip")
    if os.path.exists(final_model):
        result["model_path"] = final_model
    elif os.path.exists(best_model):
        result["model_path"] = best_model
    else:
        ckpt_dir = os.path.join(log_dir, "checkpoints")
        if os.path.isdir(ckpt_dir):
            ckpts = sorted(Path(ckpt_dir).glob("*.zip"))
            if ckpts:
                result["model_path"] = str(ckpts[-1])
    return result


def run_evaluate(params: dict, exp_dir: Path, shared: dict, timeout: float, log_file=None) -> dict:
    model_path = shared.get("model_path")
    if not model_path or not os.path.exists(model_path):
        return {
            "returncode": 1,
            "stdout": "",
            "stderr": f"Model not found: {model_path}",
        }

    output = str(exp_dir / "eval_metrics.json")
    opponents = params.get("opponents", ["random", "greedy", "conservative", "heuristic"])
    if isinstance(opponents, list):
        opponents = ",".join(opponents)
    cmd = [
        PYTHON, str(PROJECT_ROOT / "scripts" / "evaluate_model.py"),
        "--model", model_path,
        "--episodes", str(params.get("episodes_per_opponent", 50)),
        "--opponents", opponents,
        "--output", output,
    ]
    self_play = params.get("self_play_episodes", 0)
    if self_play > 0:
        cmd.extend(["--self-play-episodes", str(self_play)])

    result = _run_subprocess(cmd, timeout, log_file=log_file)
    result["eval_output"] = output
    return result


PHASE_RUNNERS = {
    "data_gen": lambda p, d, s, t, lf=None: run_data_gen(p, d, t, log_file=lf, shared=s),
    "bc_train": run_bc_train,
    "rl_train": run_rl_train,
    "evaluate": run_evaluate,
}


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def _banner(msg: str, log_file=None):
    line = f"\n{'='*60}\n  {msg}\n{'='*60}\n"
    sys.stdout.write(line)
    sys.stdout.flush()
    if log_file:
        log_file.write(line)
        log_file.flush()


def run_experiment(config: dict, experiment_id: str | None = None) -> dict:
    """Execute all phases and return the result dict."""
    exp_id = experiment_id or config.get("experiment_id") or generate_id()
    exp_dir = EXPERIMENTS_DIR / exp_id
    os.makedirs(exp_dir, exist_ok=True)

    with open(exp_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    total_budget = config.get("total_budget_seconds", 600)
    deadline = time.monotonic() + total_budget
    shared: dict = {}
    phase_results = []
    status = "success"

    log_path = exp_dir / "log.txt"
    log_file = open(log_path, "w")

    _banner(f"Experiment {exp_id}: {config.get('description', '')}", log_file)

    try:
        for phase_cfg in config.get("phases", []):
            name = phase_cfg["name"]
            fraction = phase_cfg.get("budget_fraction", 0.25)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                status = "timeout"
                _banner(f"[{name}] SKIPPED — no time remaining", log_file)
                break

            phase_timeout = min(fraction * total_budget, remaining)
            params = phase_cfg.get("params", {})

            _banner(f"[{name}] starting (budget {phase_timeout:.0f}s, {remaining:.0f}s left)", log_file)
            t0 = time.monotonic()

            runner = PHASE_RUNNERS.get(name)
            if runner is None:
                _banner(f"[{name}] unknown phase, skipping", log_file)
                continue

            result = runner(params, exp_dir, shared, phase_timeout, log_file)
            elapsed = time.monotonic() - t0
            result["elapsed_seconds"] = round(elapsed, 2)

            # Propagate shared state
            for key in ("data_path", "model_path", "eval_output"):
                if key in result:
                    shared[key] = result[key]

            phase_results.append({"phase": name, **result})

            phase_status = "TIMEOUT" if result.get("timeout") else (
                "OK" if result["returncode"] == 0 else f"ERROR (rc={result['returncode']})"
            )
            _banner(f"[{name}] {phase_status} in {elapsed:.1f}s", log_file)

            if result.get("timeout"):
                status = "timeout"
                break
            if result["returncode"] != 0:
                status = "error"
                break
    finally:
        log_file.close()

    total_elapsed = round(time.monotonic() + total_budget - deadline, 2)

    # Load eval metrics if available
    metrics = {}
    eval_path = shared.get("eval_output")
    if eval_path and os.path.exists(eval_path):
        with open(eval_path) as f:
            metrics = json.load(f)

    result_json = {
        "experiment_id": exp_id,
        "description": config.get("description", ""),
        "status": status,
        "total_elapsed_seconds": total_elapsed,
        "model_path": shared.get("model_path"),
        "phases": phase_results,
        "metrics": metrics,
    }

    with open(exp_dir / "result.json", "w") as f:
        json.dump(result_json, f, indent=2)

    # Final summary
    wr_h = metrics.get("vs_heuristic", {}).get("win_rate", "N/A")
    wr_r = metrics.get("vs_random", {}).get("win_rate", "N/A")
    print(f"\n{'='*60}")
    print(f"  DONE: {exp_id}")
    print(f"  Status: {status}  |  Total: {total_elapsed:.0f}s")
    print(f"  WR vs heuristic: {wr_h}  |  WR vs random: {wr_r}")
    print(f"  Results: {exp_dir / 'result.json'}")
    print(f"{'='*60}")

    return result_json


def main():
    parser = argparse.ArgumentParser(description="Run an Oh Hell experiment (10-min budget)")
    parser.add_argument("config", help="Path to experiment config JSON")
    parser.add_argument("--id", dest="experiment_id", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    result = run_experiment(config, args.experiment_id)
    sys.exit(0 if result["status"] == "success" else 1)


if __name__ == "__main__":
    main()
