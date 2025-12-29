from rlohhell.utils import *


class Env(object):
    """
    Base environment class mirroring the RL-Card style API used elsewhere
    in the project. Only the functions relied on by the Oh Hell environment
    are kept here.
    """

    def __init__(self, config):
        self.allow_step_back = self.game.allow_step_back = config.get("allow_step_back", False)
        self.action_recorder = []

        self.game.configure(config)
        self.num_players = self.game.get_num_players()
        self.num_actions = self.game.get_num_actions()
        self.timestep = 0

    def reset(self):
        state, player_id = self.game.init_game()
        self.action_recorder = []
        return self._extract_state(state), player_id

    def step(self, action, raw_action=False):
        if not raw_action:
            action = self._decode_action(action)

        self.timestep += 1
        self.action_recorder.append((self.get_player_id(), action))
        next_state, player_id = self.game.step(action)

        return self._extract_state(next_state), player_id

    def get_player_id(self):
        return self.game.get_player_id()

    def get_state(self, player_id):
        return self._extract_state(self.game.get_state(player_id))

    def is_over(self):
        return self.game.is_over()

    def get_payoffs(self):
        return self.game.get_payoffs()
