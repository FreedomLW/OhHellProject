"""Behaviour-cloning trainer for Oh Hell using bruteforce oracle labels.

Loads an NPZ dataset produced by ``generate_bc_data.py`` and trains a
MaskablePPO policy via supervised cross-entropy loss.  The resulting
checkpoint is SB3-compatible and can be loaded for console play or
fine-tuned with ``train_maskable_self_play.py --resume-from``.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

from rlohhell.envs.ohhell import OhHellEnv2


def _build_dummy_env():
    """Create a dummy vectorized env for MaskablePPO initialisation."""
    return make_vec_env(
        lambda: OhHellEnv2(num_players=4, agent_id=0),
        n_envs=1,
        vec_env_cls=DummyVecEnv,
    )


def train_bc(
    data_path: str,
    output_dir: str,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 3e-4,
    device: str = "auto",
    net_arch: list | None = None,
):
    """Run the full BC training pipeline and save an SB3 checkpoint."""

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    data = np.load(data_path)
    obs_np = data["observations"]
    masks_np = data["action_masks"]
    actions_np = data["actions"]
    print(f"Loaded {len(actions_np)} samples from {data_path}")

    obs_t = torch.tensor(obs_np, dtype=torch.float32)
    masks_t = torch.tensor(masks_np, dtype=torch.bool)
    actions_t = torch.tensor(actions_np, dtype=torch.long)
    dataset = TensorDataset(obs_t, masks_t, actions_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # ------------------------------------------------------------------
    # Create model (same architecture as self-play)
    # ------------------------------------------------------------------
    dummy_env = _build_dummy_env()
    policy_kwargs = {}
    if net_arch:
        policy_kwargs["net_arch"] = net_arch
    model = MaskablePPO("MultiInputPolicy", dummy_env, verbose=0, policy_kwargs=policy_kwargs or None)
    policy = model.policy
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = policy.to(device)
    policy.train()

    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        for obs_batch, mask_batch, action_batch in loader:
            obs_batch = obs_batch.to(device)
            mask_batch = mask_batch.to(device)
            action_batch = action_batch.to(device)

            # Build the observation dict that the policy expects.
            obs_dict = {
                "observation": obs_batch,
                "action_mask": mask_batch.to(torch.int8),
            }

            # Forward pass: features → latent → logits
            features = policy.extract_features(obs_dict, policy.features_extractor)
            latent_pi, _ = policy.mlp_extractor(features)
            logits = policy.action_net(latent_pi)

            # Mask illegal actions
            logits[~mask_batch] = float("-inf")

            loss = F.cross_entropy(logits, action_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(action_batch)
            preds = logits.argmax(dim=-1)
            correct += (preds == action_batch).sum().item()
            total += len(action_batch)

        avg_loss = total_loss / max(total, 1)
        accuracy = correct / max(total, 1)
        print(f"Epoch {epoch + 1}/{epochs}  loss={avg_loss:.4f}  acc={accuracy:.3f}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "bc_model")
    model.save(save_path)
    print(f"Saved BC model to {save_path}.zip")

    dummy_env.close()
    return save_path + ".zip"


def main():
    parser = argparse.ArgumentParser(description="Behaviour-cloning trainer for Oh Hell")
    parser.add_argument("--data", type=str, required=True, help="Path to NPZ dataset")
    parser.add_argument("--output", type=str, default="runs/bc")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--net-arch", type=str, default=None, help="Comma-separated layer sizes, e.g. 256,256")
    args = parser.parse_args()

    na = [int(x) for x in args.net_arch.split(",")] if args.net_arch else None
    train_bc(
        data_path=args.data,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        net_arch=na,
    )


if __name__ == "__main__":
    main()
