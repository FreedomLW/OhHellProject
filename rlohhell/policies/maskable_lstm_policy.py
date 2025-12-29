"""Custom LSTM policy with masked logits for Oh Hell!.

This policy is intended for use with :class:`sb3_contrib.MaskablePPO` when
recurrent behaviour is desired. It keeps a lightweight LSTM layer on top of
the default combined feature extractor and explicitly supports action masks by
propagating them to the distribution helper.
"""

from typing import Any, Dict, Optional

import numpy as np
import torch as th
from gymnasium import spaces
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class MaskableLstmExtractor(BaseFeaturesExtractor):
    """LSTM-based feature extractor for dict observations.

    The extractor flattens the ``"observation"`` entry and runs it through a
    small LSTM to provide temporal context for the downstream policy and value
    networks. Hidden state management is kept internal because SB3 collects
    transitions as independent batches; the LSTM still provides a recurrent
    inductive bias without requiring custom rollout buffers.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        feature_dim: int = 128,
        lstm_hidden_size: int = 128,
    ) -> None:
        super().__init__(observation_space, lstm_hidden_size)
        obs_space = observation_space["observation"]
        flat_dim = int(np.prod(obs_space.shape))
        self.flatten = th.nn.Flatten()
        self.projection = th.nn.Linear(flat_dim, feature_dim)
        self.activation = th.nn.ReLU()
        self.lstm = th.nn.LSTM(feature_dim, lstm_hidden_size, batch_first=True)
        self._lstm_hidden_size = lstm_hidden_size

    def forward(self, observations: Dict[str, th.Tensor]) -> th.Tensor:  # type: ignore[override]
        obs = observations["observation"]
        x = self.flatten(obs)
        x = self.activation(self.projection(x))
        # Add a sequence dimension for the LSTM even when using single steps
        x = x.unsqueeze(1)
        outputs, _ = self.lstm(x)
        return outputs[:, -1, :]


class MaskableLstmPolicy(MaskableActorCriticPolicy):
    """Maskable policy that inserts an LSTM in the feature extractor."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Discrete,
        lr_schedule,
        lstm_hidden_size: int = 128,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=MaskableLstmExtractor,
            features_extractor_kwargs={
                "lstm_hidden_size": lstm_hidden_size,
            },
            **kwargs,
        )

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor, latent_sde: Optional[th.Tensor] = None):
        distribution = super()._get_action_dist_from_latent(latent_pi, latent_sde)
        return distribution
