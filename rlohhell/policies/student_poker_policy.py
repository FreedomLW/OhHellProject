from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from gymnasium import spaces
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class StudentPokerPolicy(nn.Module):
    """Recurrent policy with separate bidding and playing heads."""

    def __init__(
        self,
        public_state_dim: int,
        trick_embedding_dim: int,
        bid_action_size: int,
        play_action_size: int,
        hand_size: int = 36,
        hidden_size: int = 256,
    ) -> None:
        super().__init__()
        self.hand_size = hand_size
        self.public_state_dim = public_state_dim
        self.trick_embedding_dim = trick_embedding_dim
        self.hidden_size = hidden_size

        input_dim = hand_size + public_state_dim + trick_embedding_dim
        self.input_projection = nn.Linear(input_dim, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)

        self.bid_head = nn.Linear(hidden_size, bid_action_size)
        self.play_head = nn.Linear(hidden_size, play_action_size)
        self.bid_value_head = nn.Linear(hidden_size, 1)
        self.play_value_head = nn.Linear(hidden_size, 1)

    def forward(
        self,
        obs: Dict[str, Union[torch.Tensor, "numpy.ndarray"]],
        phase: str,
        action_mask: Optional[torch.Tensor] = None,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Return masked logits for the requested phase.

        Args:
            obs: Observation dictionary containing ``hand``, ``public_state``, and
                ``trick`` tensors.
            phase: Either ``"bid"`` or ``"play"``.
            action_mask: Optional boolean mask indicating legal actions for the
                selected phase.
            hidden_state: Optional initial LSTM hidden and cell state.

        Returns:
            A tuple of masked logits for the selected head and the updated LSTM
            state.
        """

        body, next_state = self.encode_body(obs, hidden_state=hidden_state)

        logits = self._select_head(body, phase)
        logits = self._apply_mask(logits, action_mask)
        return logits, next_state

    def encode_body(
        self,
        obs: Dict[str, Union[torch.Tensor, "numpy.ndarray"]],
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        features = self._encode_observation(obs)
        lstm_input = torch.relu(self.input_projection(features)).unsqueeze(1)
        lstm_output, next_state = self.lstm(lstm_input, hidden_state)
        body = lstm_output[:, -1, :]
        return body, next_state

    def value(
        self,
        obs: Dict[str, Union[torch.Tensor, "numpy.ndarray"]],
        phase: str,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Return the scalar value estimate for the requested phase."""

        features = self._encode_observation(obs)
        lstm_input = torch.relu(self.input_projection(features)).unsqueeze(1)
        lstm_output, next_state = self.lstm(lstm_input, hidden_state)
        body = lstm_output[:, -1, :]

        if phase == "bid":
            value = self.bid_value_head(body)
        elif phase == "play":
            value = self.play_value_head(body)
        else:
            raise ValueError(f"Unknown phase '{phase}'.")

        return value.squeeze(-1), next_state

    def _encode_observation(
        self, obs: Dict[str, Union[torch.Tensor, "numpy.ndarray"]]
    ) -> torch.Tensor:
        try:
            hand = torch.as_tensor(obs["hand"], dtype=torch.float32)
            public_state = torch.as_tensor(obs["public_state"], dtype=torch.float32)
            trick = torch.as_tensor(obs["trick"], dtype=torch.float32)
        except KeyError as exc:  # pragma: no cover - defensive branch
            raise KeyError("Observation is missing required keys") from exc

        hand = self._ensure_batch(hand)
        public_state = self._ensure_batch(public_state)
        trick = self._ensure_batch(trick)
        return torch.cat([hand, public_state, trick], dim=-1)

    def _ensure_batch(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() == 1:
            return tensor.unsqueeze(0)
        return tensor

    def _select_head(self, body: torch.Tensor, phase: str) -> torch.Tensor:
        if phase == "bid":
            return self.bid_head(body)
        if phase == "play":
            return self.play_head(body)
        raise ValueError(f"Unknown phase '{phase}'.")

    def _apply_mask(
        self, logits: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if mask is None:
            return logits

        bool_mask = mask.to(logits.device).bool()
        if bool_mask.dim() == 1:
            bool_mask = bool_mask.unsqueeze(0)
        if bool_mask.shape != logits.shape:
            raise ValueError(
                "Action mask must have the same shape as the output logits."
            )
        return logits.masked_fill(~bool_mask, -1e9)


def _split_student_observation(flat: torch.Tensor, hand_size: int, trick_size: int) -> Dict[str, torch.Tensor]:
    hand = flat[..., :hand_size]
    trick = flat[..., hand_size : hand_size + trick_size]
    public_state = flat[..., hand_size + trick_size :]
    return {
        "hand": hand,
        "trick": trick,
        "public_state": public_state,
    }


class StudentPokerExtractor(BaseFeaturesExtractor):
    """Feature extractor that reuses :class:`StudentPokerPolicy`'s body."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        base_model: Optional[StudentPokerPolicy] = None,
        hand_size: int = 36,
        trick_size: int = 36,
        public_state_dim: Optional[int] = None,
    ) -> None:
        obs_size = int(np.prod(observation_space["observation"].shape))
        inferred_public_dim = public_state_dim or max(1, obs_size - hand_size - trick_size)
        model = base_model or StudentPokerPolicy(
            public_state_dim=inferred_public_dim,
            trick_embedding_dim=trick_size,
            bid_action_size=observation_space["action_mask"].shape[-1],
            play_action_size=observation_space["action_mask"].shape[-1],
            hand_size=hand_size,
        )
        self.student = model
        self.hand_size = hand_size
        self.trick_size = trick_size
        super().__init__(observation_space, features_dim=model.hidden_size)

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:  # type: ignore[override]
        flat = torch.as_tensor(observations["observation"], dtype=torch.float32)
        obs_dict = _split_student_observation(flat, self.hand_size, self.trick_size)
        body, _ = self.student.encode_body(obs_dict)
        return body


class IdentityMlpExtractor(nn.Module):
    """Bypass extractor that forwards latent features to policy and value heads."""

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.latent_dim_pi = latent_dim
        self.latent_dim_vf = latent_dim

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return features, features


class StudentPokerActorCriticPolicy(MaskableActorCriticPolicy):
    """Maskable PPO policy that swaps in the StudentPoker recurrent backbone."""

    def __init__(self, observation_space: spaces.Dict, action_space: spaces.Discrete, lr_schedule, **kwargs):
        self._student_extractor: Optional[StudentPokerExtractor] = None
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=StudentPokerExtractor,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = IdentityMlpExtractor(self.features_dim)

