from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn


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

        features = self._encode_observation(obs)
        lstm_input = torch.relu(self.input_projection(features)).unsqueeze(1)
        lstm_output, next_state = self.lstm(lstm_input, hidden_state)
        body = lstm_output[:, -1, :]

        logits = self._select_head(body, phase)
        logits = self._apply_mask(logits, action_mask)
        return logits, next_state

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
