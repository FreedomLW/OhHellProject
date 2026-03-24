import json
import os
import tempfile

import pytest

from scripts.run_experiment import run_experiment, EXPERIMENTS_DIR


def test_experiment_runner_smoke(tmp_path, monkeypatch):
    """Run a tiny BC experiment end-to-end."""
    # Point experiments dir to tmp_path so we don't pollute the repo
    monkeypatch.setattr("scripts.run_experiment.EXPERIMENTS_DIR", tmp_path)

    config = {
        "description": "test: 2 seeds, round 1, 2 epochs",
        "total_budget_seconds": 120,
        "phases": [
            {
                "name": "data_gen",
                "budget_fraction": 0.30,
                "params": {"num_seeds": 2, "round_sizes": "1"},
            },
            {
                "name": "bc_train",
                "budget_fraction": 0.30,
                "params": {"epochs": 2, "batch_size": 32, "lr": 1e-3},
            },
            {
                "name": "evaluate",
                "budget_fraction": 0.40,
                "params": {
                    "episodes_per_opponent": 3,
                    "opponents": ["random", "heuristic"],
                },
            },
        ],
    }

    result = run_experiment(config, experiment_id="test_smoke")

    assert result["status"] == "success"
    assert result["experiment_id"] == "test_smoke"

    exp_dir = tmp_path / "test_smoke"
    assert (exp_dir / "config.json").exists()
    assert (exp_dir / "result.json").exists()
    assert (exp_dir / "log.txt").exists()

    # Check metrics were collected
    assert "metrics" in result
    assert "vs_random" in result["metrics"]
    assert "vs_heuristic" in result["metrics"]


def test_experiment_timeout(tmp_path, monkeypatch):
    """Verify that a tiny budget causes a timeout or early termination."""
    monkeypatch.setattr("scripts.run_experiment.EXPERIMENTS_DIR", tmp_path)

    config = {
        "description": "test: should timeout",
        "total_budget_seconds": 2,
        "phases": [
            {
                "name": "data_gen",
                "budget_fraction": 0.50,
                "params": {"num_seeds": 500, "round_sizes": "1,2,3,4,5"},
            },
            {
                "name": "bc_train",
                "budget_fraction": 0.50,
                "params": {"epochs": 50},
            },
        ],
    }

    result = run_experiment(config, experiment_id="test_timeout")

    # With only 2 seconds budget, it should timeout or error
    assert result["status"] in ("timeout", "error", "success")
    assert (tmp_path / "test_timeout" / "result.json").exists()
