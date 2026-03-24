"""Generate behaviour-cloning dataset using the bruteforce oracle.

For each seed × round_size combination, creates a single-round game, plays it
with BruteforceStrategy at the target seat, and records (observation,
action_mask, action_id) at every target-player decision point.

Output is a compressed NPZ file loadable by ``scripts/train_bc.py``.
"""

from __future__ import annotations

import argparse
import os
from typing import List, Sequence

import numpy as np

from rlohhell.envs.ohhell import OhHellEnv2, mask_from_state
from rlohhell.games.ohhell.game import OhHellGame
from rlohhell.games.ohhell.strategies import BaseStrategy, HeuristicStrategy
from rlohhell.heuristics.bruteforce_agent import BruteforceStrategy


def action_to_id(action, hand) -> int:
    """Convert a game action (int bid or Card) to a discrete action_id."""
    if isinstance(action, int):
        return action
    mode = getattr(action, "joker_mode", None)
    if mode is not None:
        return len(hand) + (1 if mode == "high" else 0)
    for idx, card in enumerate(hand):
        if card == action:
            return idx
    raise ValueError(f"Action {action} not found in hand {hand}")


def collect_single_round(
    seed: int,
    round_size: int,
    target_seat: int,
    bruteforce: BruteforceStrategy,
    bid_strategy: BaseStrategy,
    opponent: BaseStrategy,
    env: OhHellEnv2,
) -> List[dict]:
    """Play one round and return BC samples for the target seat.

    Bidding labels come from ``bid_strategy`` (partial-info, e.g. Heuristic).
    Card-play labels come from ``bruteforce`` (perfect-info oracle).
    This avoids teaching the model to overbid based on hidden information.
    """
    game = OhHellGame(num_players=4)
    game.np_random = np.random.RandomState(seed)
    game.current_player = seed % 4
    game.init_game()
    game.round_sequence = [round_size]
    game.max_rounds = 1
    game.current_round = 0

    # Borrow the env's observation encoding.
    env.game = game
    env.agent_id = target_seat

    samples = []
    while not game.is_over():
        pid = game.get_player_id()
        if pid == target_seat:
            state = game.get_state(pid)
            obs_dict = env._get_obs_dict(state)
            obs = obs_dict["observation"]
            mask = obs_dict["action_mask"]
            hand = list(game.players[pid].hand)

            player = game.players[pid]
            if not player.has_proposed:
                # Bidding: use partial-info heuristic (not oracle)
                action = bid_strategy.select_action(game, pid)
            else:
                # Card play: use perfect-info oracle
                action = bruteforce.select_action(game, pid)

            aid = action_to_id(action, hand)
            samples.append({"obs": obs, "mask": mask, "action": aid})
            game.step(action)
        else:
            action = opponent.select_action(game, pid)
            game.step(action)

    return samples


def generate_dataset(
    num_seeds: int,
    round_sizes: Sequence[int],
    target_seat: int = 0,
    opponent_model: str | None = None,
    opponent_models: Sequence[str] | None = None,
) -> dict:
    """Generate full BC dataset and return as numpy arrays.

    Bidding labels always come from HeuristicStrategy (partial info).
    Card-play labels come from BruteforceStrategy (perfect-info oracle).
    """
    import random as _rng
    from rlohhell.games.ohhell.strategies import RandomStrategy, GreedyStrategy, ConservativeStrategy

    bruteforce = BruteforceStrategy()
    bid_strategy = HeuristicStrategy()

    # Build opponent pool
    opponent_pool: List[BaseStrategy] = [
        HeuristicStrategy(),
        RandomStrategy(),
        GreedyStrategy(),
        ConservativeStrategy(),
    ]
    model_paths = list(opponent_models or [])
    if opponent_model:
        model_paths.append(opponent_model)
    if model_paths:
        from rlohhell.games.ohhell.console import ModelPolicyStrategy
        for p in model_paths:
            opponent_pool.append(ModelPolicyStrategy(p, deterministic=False))
        print(f"Opponent pool: 4 heuristics + {len(model_paths)} model(s)")
    else:
        print(f"Opponent pool: 4 heuristics")

    rng = _rng.Random(42)
    env = OhHellEnv2(num_players=4, agent_id=target_seat)

    all_obs, all_masks, all_actions = [], [], []

    for seed in range(num_seeds):
        opponent = rng.choice(opponent_pool)
        for rs in round_sizes:
            samples = collect_single_round(
                seed=seed,
                round_size=rs,
                target_seat=target_seat,
                bruteforce=bruteforce,
                bid_strategy=bid_strategy,
                opponent=opponent,
                env=env,
            )
            for s in samples:
                all_obs.append(s["obs"])
                all_masks.append(s["mask"])
                all_actions.append(s["action"])

    return {
        "observations": np.array(all_obs, dtype=np.float32),
        "action_masks": np.array(all_masks, dtype=np.int8),
        "actions": np.array(all_actions, dtype=np.int64),
    }


def main():
    parser = argparse.ArgumentParser(description="Generate BC dataset from bruteforce oracle")
    parser.add_argument("--num-seeds", type=int, default=500)
    parser.add_argument(
        "--round-sizes",
        type=str,
        default="1,2,3,4,5,6,7,8,9",
        help="Comma-separated round sizes",
    )
    parser.add_argument("--target-seat", type=int, default=0)
    parser.add_argument("--output", type=str, default="data/bc_train.npz")
    parser.add_argument(
        "--opponent-model",
        type=str,
        default=None,
        help="Path to SB3 .zip checkpoint to add to opponent pool",
    )
    parser.add_argument(
        "--opponent-models",
        type=str,
        default=None,
        help="Comma-separated paths to SB3 checkpoints for opponent pool",
    )
    args = parser.parse_args()

    round_sizes = [int(x) for x in args.round_sizes.split(",")]
    opp_models = args.opponent_models.split(",") if args.opponent_models else None
    data = generate_dataset(
        num_seeds=args.num_seeds,
        round_sizes=round_sizes,
        target_seat=args.target_seat,
        opponent_model=args.opponent_model,
        opponent_models=opp_models,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.savez_compressed(args.output, **data)
    print(
        f"Saved {len(data['actions'])} samples to {args.output} "
        f"(seeds={args.num_seeds}, round_sizes={round_sizes})"
    )


if __name__ == "__main__":
    main()
