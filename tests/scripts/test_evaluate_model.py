import json
import os
import tempfile

import numpy as np
import pytest

from scripts.generate_bc_data import generate_dataset
from scripts.train_bc import train_bc


def _make_tiny_model(tmpdir: str) -> str:
    """Generate a tiny BC dataset and train a model, return checkpoint path."""
    data = generate_dataset(num_seeds=2, round_sizes=[1, 2])
    data_path = os.path.join(tmpdir, "bc_data.npz")
    np.savez_compressed(data_path, **data)
    return train_bc(data_path=data_path, output_dir=tmpdir, epochs=2, batch_size=32, lr=1e-3, device="cpu")


def test_evaluate_smoke():
    """Evaluate a tiny BC model against 2 opponents for 3 episodes each."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = _make_tiny_model(tmpdir)
        output_path = os.path.join(tmpdir, "eval.json")

        from scripts.evaluate_model import main as eval_main
        import sys
        orig_argv = sys.argv
        sys.argv = [
            "evaluate_model.py",
            "--model", model_path,
            "--episodes", "3",
            "--opponents", "random,heuristic",
            "--output", output_path,
        ]
        try:
            eval_main()
        finally:
            sys.argv = orig_argv

        assert os.path.exists(output_path)
        with open(output_path) as f:
            results = json.load(f)

        assert "vs_random" in results
        assert "vs_heuristic" in results
        for key in ("vs_random", "vs_heuristic"):
            assert "win_rate" in results[key]
            assert "avg_score" in results[key]
            assert "avg_per_round_score" in results[key]
            assert 0.0 <= results[key]["win_rate"] <= 1.0
