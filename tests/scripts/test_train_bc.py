import os
import tempfile

import numpy as np
import pytest

from scripts.generate_bc_data import generate_dataset
from scripts.train_bc import train_bc


def test_bc_training_smoke():
    """Generate tiny dataset, train 2 epochs, verify checkpoint is saved."""
    data = generate_dataset(num_seeds=2, round_sizes=[1, 2])

    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = os.path.join(tmpdir, "bc_data.npz")
        np.savez_compressed(data_path, **data)

        model_path = train_bc(
            data_path=data_path,
            output_dir=tmpdir,
            epochs=2,
            batch_size=32,
            lr=1e-3,
            device="cpu",
        )

        assert os.path.exists(model_path), f"Checkpoint not found at {model_path}"

        # Verify the checkpoint loads as MaskablePPO.
        from sb3_contrib.ppo_mask import MaskablePPO

        loaded = MaskablePPO.load(model_path, device="cpu")
        assert loaded is not None
