"""Supervised training pipeline for :class:`StudentPokerPolicy`.

This script collects trajectories from the existing heuristic bot and trains a
two-headed recurrent policy to imitate its bidding and play decisions. The
pipeline is intentionally lightweight and relies only on PyTorch plus the
existing Oh Hell environment. Collected samples are split into bidding and play
phases so that the appropriate policy head is optimised with cross-entropy.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import trange

from rlohhell.envs.ohhell import OhHellEnv2
from rlohhell.policies import StudentPokerPolicy
from rlohhell.utils.opponents import StrategyOpponent
from rlohhell.games.ohhell.strategies import HeuristicStrategy


Phase = Literal["bid", "play"]


@dataclass
class StudentSample:
    """Single observation-action pair for StudentPokerPolicy training."""

    hand: torch.Tensor
    public_state: torch.Tensor
    trick: torch.Tensor
    action: int
    mask: torch.Tensor
    phase: Phase


class StudentDataset(Dataset[StudentSample]):
    """Dataset of StudentPokerPolicy experiences."""

    def __init__(self, samples: Sequence[StudentSample]):
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> StudentSample:
        return self.samples[idx]


def collate_batch(batch: Sequence[StudentSample]) -> Dict[str, torch.Tensor]:
    hands = torch.stack([sample.hand for sample in batch])
    public_state = torch.stack([sample.public_state for sample in batch])
    trick = torch.stack([sample.trick for sample in batch])
    actions = torch.tensor([sample.action for sample in batch], dtype=torch.long)
    masks = torch.stack([sample.mask for sample in batch])
    phases = [sample.phase for sample in batch]
    return {
        "hand": hands,
        "public_state": public_state,
        "trick": trick,
        "action": actions,
        "mask": masks,
        "phase": phases,
    }


def split_phase_indices(phases: Sequence[Phase]) -> Dict[Phase, List[int]]:
    indices = {"bid": [], "play": []}
    for idx, phase in enumerate(phases):
        indices[phase].append(idx)
    return indices


def select_subset(obs_dict: Dict[str, torch.Tensor], indices: Iterable[int]) -> Dict[str, torch.Tensor]:
    idx_tensor = torch.tensor(list(indices), dtype=torch.long)
    return {key: value.index_select(0, idx_tensor) for key, value in obs_dict.items()}


def determine_phase(state) -> Phase:
    legal = state.get("legal_actions", [])
    if legal and isinstance(legal[0], int):
        return "bid"
    return "play"


def action_id_from_teacher(choice, env: OhHellEnv2, state) -> int:
    legal_ids = list(env._get_legal_actions())
    if isinstance(choice, int):
        return choice if choice in legal_ids else legal_ids[0]

    matches = [lid for lid in legal_ids if env._decode_action(lid, state) == choice]
    return matches[0] if matches else legal_ids[0]


def extract_student_obs(obs: Dict[str, torch.Tensor], action_mask) -> Dict[str, torch.Tensor]:
    observation = torch.as_tensor(obs["observation"], dtype=torch.float32)
    hand_size = 36
    trick_size = 36
    hand = observation[:hand_size]
    trick = observation[hand_size : hand_size + trick_size]
    public_state = observation[hand_size + trick_size :]
    mask = torch.as_tensor(action_mask, dtype=torch.float32)
    return {"hand": hand, "public_state": public_state, "trick": trick, "mask": mask}


def collect_dataset(episodes: int, seed: int) -> List[StudentSample]:
    env = OhHellEnv2(opponent_policy=None)
    teacher = StrategyOpponent("teacher", HeuristicStrategy())
    samples: List[StudentSample] = []

    for episode in trange(episodes, desc="Collecting episodes"):
        obs, info = env.reset(seed=seed + episode)
        done = False

        while not done:
            state = env.game.get_state(env.agent_id)
            phase: Phase = determine_phase(state)
            teacher_choice = teacher.act(state, info["action_mask"], obs, env.game)
            action_id = action_id_from_teacher(teacher_choice, env, state)

            student_obs = extract_student_obs(obs, info["action_mask"])
            samples.append(
                StudentSample(
                    hand=student_obs["hand"],
                    public_state=student_obs["public_state"],
                    trick=student_obs["trick"],
                    action=action_id,
                    mask=student_obs["mask"],
                    phase=phase,
                )
            )

            obs, _, done, _, info = env.step(action_id)

    return samples


def train(
    dataset: StudentDataset,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    save_path: str,
) -> None:
    if len(dataset) == 0:
        raise ValueError("Cannot train StudentPokerPolicy on an empty dataset")

    sample = dataset[0]
    model = StudentPokerPolicy(
        public_state_dim=sample.public_state.shape[-1],
        trick_embedding_dim=sample.trick.shape[-1],
        bid_action_size=sample.mask.shape[-1],
        play_action_size=sample.mask.shape[-1],
        hand_size=sample.hand.shape[-1],
    ).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)

    for epoch in trange(epochs, desc="Training"):
        epoch_loss = 0.0
        for batch in dataloader:
            observations = {
                "hand": batch["hand"].to(device),
                "public_state": batch["public_state"].to(device),
                "trick": batch["trick"].to(device),
            }
            masks = batch["mask"].to(device)
            actions = batch["action"].to(device)
            phases: List[Phase] = batch["phase"]
            phase_indices = split_phase_indices(phases)

            loss = torch.tensor(0.0, device=device)

            if phase_indices["bid"]:
                subset = select_subset(observations, phase_indices["bid"])
                bid_actions = actions.index_select(0, torch.tensor(phase_indices["bid"], device=device))
                bid_masks = masks.index_select(0, torch.tensor(phase_indices["bid"], device=device))
                logits, _ = model(subset, phase="bid", action_mask=bid_masks)
                loss = loss + nn.functional.cross_entropy(logits, bid_actions)

            if phase_indices["play"]:
                subset = select_subset(observations, phase_indices["play"])
                play_actions = actions.index_select(0, torch.tensor(phase_indices["play"], device=device))
                play_masks = masks.index_select(0, torch.tensor(phase_indices["play"], device=device))
                logits, _ = model(subset, phase="play", action_mask=play_masks)
                loss = loss + nn.functional.cross_entropy(logits, play_actions)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            epoch_loss += float(loss.detach().cpu())

        print(f"Epoch {epoch + 1}: loss={epoch_loss / max(1, len(dataloader)):.4f}")

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Saved StudentPokerPolicy to {save_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train StudentPokerPolicy via imitation")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to collect")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for optimisation")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--save-path", type=str, default="./runs/student_poker_policy.pt", help="Destination for the trained weights"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for data collection")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = collect_dataset(episodes=args.episodes, seed=args.seed)
    dataset = StudentDataset(samples)
    train(
        dataset=dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        save_path=args.save_path,
    )


if __name__ == "__main__":
    main()

