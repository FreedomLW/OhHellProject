import json
import os
import json
import os
import random
from collections import OrderedDict
from typing import Callable, Dict, Iterable, Optional, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding

import rlohhell
from rlohhell.envs import Env
from rlohhell.games.ohhell import Game
from rlohhell.games.base import Card
from rlohhell.games.ohhell.utils import ACTION_LIST, ACTION_SPACE

DEFAULT_GAME_CONFIG = {
    "game_num_players": 4,
}

# Max number of discrete actions available to the agent (cards in hand + two
# joker toggles). With a 36-card deck split between four players, the largest
# hand size is nine, so the action mask length stays fixed at 11.
MAX_ACTIONS = 11


def mask_from_state(state, max_actions: int = MAX_ACTIONS):
    """Build a fixed-size boolean action mask from a raw game state.

    During bidding the mask covers the bid range ``[0, hand_size]`` and
    excludes the forbidden final bid when applicable. During card play the
    mask spans the player's hand plus two joker toggles for ``7♠`` in low and
    high modes. The returned mask is always ``max_actions`` long so it remains
    compatible with vectorized environments.
    """

    legal_actions = state.get("legal_actions", [])
    hand = state.get("hand", [])
    mask = np.zeros(max_actions, dtype=np.bool_)
    if not legal_actions:
        return mask

    if isinstance(legal_actions[0], int):
        hand_size = len(hand)
        for bid in legal_actions:
            if 0 <= bid < max_actions:
                mask[bid] = True
        return mask

    seven_spades = Card("S", "7")
    for idx, card in enumerate(hand):
        if idx >= max_actions - 2:
            break
        if card in legal_actions:
            mask[idx] = True
        if card == seven_spades:
            low_idx = len(hand)
            high_idx = len(hand) + 1
            if low_idx < max_actions and any(
                getattr(action, "joker_mode", "low") == "low"
                and action.suit == seven_spades.suit
                and action.rank == seven_spades.rank
                for action in legal_actions
            ):
                mask[low_idx] = True
            if high_idx < max_actions and any(
                getattr(action, "joker_mode", "low") == "high"
                and action.suit == seven_spades.suit
                and action.rank == seven_spades.rank
                for action in legal_actions
            ):
                mask[high_idx] = True

    return mask


class OhHellEnv(Env):
    """OhHell Environment compatible with the rlcard-style API."""

    def __init__(self, config):
        self.name = "ohhell"
        self.default_game_config = DEFAULT_GAME_CONFIG
        self.game = Game()
        super().__init__(config)
        self.state_shape = [[111] for _ in range(self.num_players)]
        self.action_shape = [None for _ in range(self.num_players)]

        with open(
            os.path.join(rlohhell.__path__[0], "games/ohhell/card2index.json"), "r"
        ) as file:
            self.card2index = json.load(file)

    def _extract_state(self, state):
        played_cards = state["played_cards"]
        hand = state["hand"]
        trump_card = state["trump_card"]
        tricks_won = state["tricks_won"]
        proposed_tricks = state["proposed_tricks"]
        players_tricks_won = state["players_tricks_won"]

        idx1 = [self.card2index[card] for card in played_cards]
        idx2 = list(np.array([self.card2index[card] for card in hand]) + 51)

        obs = np.zeros(111)
        obs[idx1] = 1
        obs[idx2] = 1
        obs[104] = self.card2index[trump_card]
        obs[105] = tricks_won
        obs[106] = proposed_tricks
        obs[107:] = players_tricks_won

        legal_action_id = self._get_legal_actions()
        extracted_state = {"obs": obs, "legal_actions": legal_action_id}
        extracted_state["raw_obs"] = state
        extracted_state["raw_legal_actions"] = [a for a in state["legal_actions"]]
        extracted_state["action_record"] = self.action_recorder

        return extracted_state

    def _get_legal_actions(self):
        state = self.game.get_state(self.agent_id)
        mask = mask_from_state(state)
        legal_ids = OrderedDict()
        for idx, allowed in enumerate(mask):
            if allowed:
                legal_ids[idx] = None
        return legal_ids

    def get_payoffs(self):
        return self.game.get_payoffs()

    def _decode_action(self, action_id):
        legal_ids = self._get_legal_actions()
        if self.game.round.players_proposed == self.game.num_players:
            if action_id in list(legal_ids):
                return Card(ACTION_LIST[action_id][0], ACTION_LIST[action_id][1])
            random_card = ACTION_LIST[random.choice(list(legal_ids))]
            return Card(random_card[0], random_card[1])

        if action_id in list(legal_ids):
            return int(ACTION_LIST[action_id])
        return int(ACTION_LIST[random.choice(list(legal_ids))])

    def get_perfect_information(self):
        state = {}
        state["tricks_won"] = [self.game.players[i].tricks_won for i in range(self.num_players)]
        state["trump_card"] = self.game.trump_card
        state["played_cards"] = [c.get_index() for c in self.game.round.played_cards]
        state["hand_cards"] = [
            [c.get_index() for c in self.game.players[i].hand] for i in range(self.num_players)
        ]
        state["current_player"] = self.game.current_player
        state["legal_actions"] = self.game.get_legal_actions()
        return state


class OhHellEnv2(gym.Env):
    """Single-agent Gymnasium wrapper for Oh Hell!

    The learning agent always controls ``agent_id``. Opponents can either use
    a provided policy callable or default to random play. Observations are
    returned as a dict suitable for ``MultiInputPolicy`` with an ``action_mask``
    entry that can be consumed by :class:`sb3_contrib.MaskablePPO`.
    Rewards are granted as the change in the agent's score after each
    completed trick or after the final bonus calculation when the game ends.
    """

    metadata = {"render.modes": []}
    MAX_ACTIONS = MAX_ACTIONS

    def __init__(
        self,
        num_players: int = 4,
        agent_id: int = 0,
        opponent_policy: Optional[
            Callable[[Dict[str, np.ndarray], np.ndarray, int], int]
        ] = None,
        opponent_pool=None,
        opponent_selector: Optional[Callable[[int, int, object], Dict[int, object]]] = None,
    ):
        super().__init__()
        self.num_players = num_players
        self.agent_id = agent_id
        self.game = Game(num_players=num_players)
        self.opponent_policy = opponent_policy
        self.opponent_pool = opponent_pool
        self.opponent_selector = opponent_selector
        self._opponent_table: Dict[int, object] = {}

        with open(
            os.path.join(rlohhell.__path__[0], "games/ohhell/card2index.json"), "r"
        ) as file:
            self.card2index: Dict[str, int] = json.load(file)

        self.obs_size = 108 + 2 + (self.num_players * 3) + self.num_players + 1
        self.action_space = spaces.Discrete(self.MAX_ACTIONS)
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(
                    low=0.0, high=1.0, shape=(self.obs_size,), dtype=np.float32
                ),
                "action_mask": spaces.MultiBinary(self.MAX_ACTIONS),
            }
        )

        self.np_random = None
        self._final_scores_applied = False
        self._last_score = 0
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, seed=None, **kwargs):
        if seed is not None:
            self.seed(seed)
        self.game = Game(num_players=self.num_players)
        self.game.init_game()
        self._final_scores_applied = False
        self._last_score = self._current_score()
        self._assign_opponents()
        self._advance_to_agent()
        state = self.game.get_state(self.agent_id)
        obs = self._get_obs_dict(state)
        action_mask = mask_from_state(state, self.MAX_ACTIONS)
        info = {"action_mask": action_mask, "action_masks": action_mask}
        return obs, info

    def step(self, action):
        total_reward = 0.0

        if self.game.get_player_id() != self.agent_id:
            total_reward += self._advance_to_agent()

        state = self.game.get_state(self.agent_id)
        action_mask = mask_from_state(state, self.MAX_ACTIONS)
        if action >= len(action_mask) or not action_mask[action]:
            action = int(np.flatnonzero(action_mask)[0])

        decoded_action = self._decode_action(action, state)
        self.game.step(decoded_action)
        total_reward += self._update_score_reward()

        total_reward += self._advance_to_agent()
        done = self.game.is_over()
        if done:
            total_reward += self._apply_final_scores()

        state = self.game.get_state(self.agent_id)
        obs = self._get_obs_dict(state)
        next_mask = mask_from_state(state, self.MAX_ACTIONS)
        info = {
            "legal_actions": list(self._get_legal_actions()),
            "action_mask": next_mask,
            "action_masks": next_mask,
        }
        return obs, float(total_reward), bool(done), False, info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _decode_action(self, action_id: int, state=None):
        state = state or self.game.get_state(self.agent_id)
        legal_actions = state["legal_actions"]

        if not legal_actions:
            raise ValueError("No legal actions available to decode")

        if isinstance(legal_actions[0], int):
            if action_id in legal_actions:
                return int(action_id)
            return int(legal_actions[0])

        hand = state["hand"]
        seven_spades = Card("S", "7")
        if action_id == len(hand):
            candidate = Card("S", "7", joker_mode="low")
        elif action_id == len(hand) + 1:
            candidate = Card("S", "7", joker_mode="high")
        else:
            candidate = hand[action_id] if 0 <= action_id < len(hand) else None

        if candidate is None:
            return legal_actions[0]

        for option in legal_actions:
            if not isinstance(option, Card):
                continue
            if option == candidate and getattr(option, "joker_mode", None) == getattr(
                candidate, "joker_mode", None
            ):
                return option

            if (
                option.suit == seven_spades.suit
                and option.rank == seven_spades.rank
                and getattr(option, "joker_mode", None)
                == getattr(candidate, "joker_mode", None)
            ):
                return option

        return legal_actions[0]

    def _get_legal_actions(self):
        legal_actions = self.game.get_legal_actions()
        if self.game.round.players_proposed == self.game.num_players:
            legal_ids = {ACTION_SPACE[action.get_index()]: None for action in legal_actions}
        else:
            legal_ids = {ACTION_SPACE[str(action)]: None for action in legal_actions}
        return OrderedDict(legal_ids)

    def _extract_state(self, state):
        obs = np.zeros(self.obs_size, dtype=np.float32)

        hand = state["hand"]
        played_cards = state["played_cards"]
        previously_played = state.get("previously_played_cards", [])
        trump_card = state["trump_card"]
        bids = state.get("players_tricks_proposed", [0] * self.num_players)
        tricks_won = state.get("players_tricks_won", [0] * self.num_players)
        current_player = state.get("current_player", self.game.get_player_id())

        hand_idx = [self._index_card(card) for card in hand]
        played_idx = [self._index_card(card) for card in played_cards]
        previous_idx = [self._index_card(card) for card in previously_played]

        obs[hand_idx] = 1
        obs[36 + np.array(played_idx, dtype=int)] = 1
        obs[72 + np.array(previous_idx, dtype=int)] = 1

        obs[108] = self._index_card(trump_card) / max(1, len(self.card2index) - 1)
        max_cards = self.game.round.round_number
        obs[109] = len(hand) / max_cards

        offset = 110
        for bid in bids:
            obs[offset] = bid / max_cards
            offset += 1
        for tw in tricks_won:
            obs[offset] = tw / max_cards
            offset += 1
        for player in self.game.players:
            obs[offset] = len(player.hand) / max_cards
            offset += 1

        for pid in range(self.num_players):
            obs[offset + pid] = 1.0 if pid == current_player else 0.0
        offset += self.num_players

        obs[offset] = self.game.round_counter / max_cards
        return obs

    def _get_obs_dict(self, state):
        action_mask = mask_from_state(state, self.MAX_ACTIONS)
        return {
            "observation": self._extract_state(state),
            "action_mask": action_mask.astype(np.int8),
        }

    def action_masks(self) -> np.ndarray:
        """Return the current legal action mask for SB3 MaskablePPO."""

        state = self.game.get_state(self.agent_id)
        return mask_from_state(state, self.MAX_ACTIONS)

    def _current_score(self) -> int:
        return self.game.players[self.agent_id].tricks_won

    def _update_score_reward(self) -> float:
        current_score = self._current_score()
        delta = current_score - self._last_score
        self._last_score = current_score
        return float(delta)

    def _advance_to_agent(self) -> float:
        reward = 0.0
        while not self.game.is_over() and self.game.get_player_id() != self.agent_id:
            current_player = self.game.get_player_id()
            opponent_action = self._sample_opponent_action(current_player)
            self.game.step(opponent_action)
            reward += self._update_score_reward()
        if self.game.is_over():
            reward += self._apply_final_scores()
        return reward

    def _apply_final_scores(self) -> float:
        if self._final_scores_applied:
            return 0.0
        final_scores = self.game.get_payoffs()
        self._final_scores_applied = True
        bonus = final_scores[self.agent_id] - self._last_score
        self._last_score = final_scores[self.agent_id]
        return float(bonus)

    def _index_card(self, card) -> int:
        if isinstance(card, str):
            return self.card2index[card]
        return self.card2index[card.get_index()]

    def _assign_opponents(self) -> None:
        """Randomly sample opponents for every non-agent seat."""

        assignments: Dict[int, object] = {}
        if self.opponent_selector is not None:
            assignments = self.opponent_selector(
                self.num_players, self.agent_id, self.opponent_pool
            )
        elif self.opponent_pool is not None:
            assignments = getattr(self.opponent_pool, "sample_table", lambda *_: {})(
                self.num_players, self.agent_id
            )

        self._opponent_table = assignments or {}

    def _sample_opponent_action(self, player_id: int):
        legal_actions = list(self.game.get_legal_actions())
        if not legal_actions:
            raise ValueError("Opponents have no legal actions to play")

        policy = self._opponent_table.get(player_id, self.opponent_policy)

        if policy is not None:
            state = self.game.get_state(player_id)
            action_mask = mask_from_state(state)
            obs_dict = self._get_obs_dict(state)

            if hasattr(policy, "act"):
                chosen = policy.act(state, action_mask, obs_dict, self.game)
            else:
                try:
                    chosen = policy(obs_dict, action_mask, player_id, state)
                except TypeError:
                    chosen = policy(obs_dict, action_mask, player_id)

            if isinstance(chosen, (Card, int)) and chosen in legal_actions:
                return chosen

            if isinstance(chosen, int) and 0 <= chosen < len(action_mask) and action_mask[chosen]:
                return self._decode_action(chosen, state)

        raw_action = random.choice(legal_actions)
        return raw_action

