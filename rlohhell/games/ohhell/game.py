from copy import deepcopy, copy
import numpy as np
import random

from rlohhell.games.ohhell.dealer import OhHellDealer as Dealer
from rlohhell.games.ohhell.player import OhHellPlayer as Player
from rlohhell.games.ohhell.judger import OhHellJudger as Judger
from rlohhell.games.ohhell.round import OhHellRound as Round


class OhHellGame:

    def __init__(self, allow_step_back=False, num_players=4, player_strategies=None):
        ''' Initialize the class ohhell Game
        '''
        self.allow_step_back = allow_step_back
        self.np_random = np.random.RandomState()
        self.num_players = num_players
        self.payoffs = [0 for _ in range(num_players)]
        self.current_player = random.randint(0, self.num_players-1)
        self.player_strategies = player_strategies or [None for _ in range(num_players)]
        self.scores = [0 for _ in range(num_players)]
        self.bids_history = []
        self.round_sequence = []
        self.current_round = 0


    def configure(self, game_config):
        ''' Specifiy some game specific parameters, such as number of players
        '''
        self.num_players = game_config['game_num_players']

    def init_game(self):
        ''' Initialilze the game of Oh Hell

        This version supports up to four-player OhHell

        Returns:
            (tuple): Tuple containing:

                (dict): The first state of the game
                (int): Current player's id
        '''
        # Initilize a dealer that can deal cards
        self.dealer = Dealer(self.np_random)

        # Initilize players to play the game
        self.players = [Player(i, self.np_random) for i in range(self.num_players)]

        # Initialize a judger class which will decide who wins in the end
        self.judger = Judger(self.np_random)

        max_cards = len(self.dealer.deck) // self.num_players
        climb = list(range(1, max_cards + 1))
        plateau = [max_cards for _ in range(self.num_players)]
        descend = list(range(max_cards - 1, 0, -1))
        self.round_sequence = climb + plateau + descend
        self.current_round = 0

        self.history = []
        self.previously_played_cards = []
        self.last_winner = 0

        state, player_id = self._start_round(self.round_sequence[self.current_round])
        return state, player_id

    def _reset_players(self):
        for player in self.players:
            player.hand = []
            player.played_cards = []
            player.has_proposed = False
            player.proposed_tricks = 0
            player.tricks_won = 0

    def _start_round(self, cards_per_player, keep_scores=True):
        if not keep_scores:
            self.scores = [0 for _ in range(self.num_players)]

        self.dealer = Dealer(self.np_random)
        self._reset_players()

        for i in range(cards_per_player * self.num_players):
            self.players[i % self.num_players].hand.append(self.dealer.deal_card())

        self.trump_card = self.dealer.flip_trump_card()
        self.played_cards = []
        self.round = Round(
            np_random=self.np_random,
            dealer=self.dealer,
            num_players=self.num_players,
            round_number=cards_per_player,
            last_winner=self.current_player,
            current_player=self.current_player,
        )
        self.round_counter = 0

        player_id = self.round.current_player
        state = self.get_state(player_id)
        return state, player_id



    def step(self, action):
        ''' Get the next state

        Args:
            action (str): A specific action

        Returns:
            (tuple): Tuple containing:

                (dict): next player's state
                (int): next plater's id
        '''

        if self.allow_step_back:
            # First snapshot the current state
            r = deepcopy(self.round)
            b = self.round.current_player
            r_c = self.round_counter
            d = deepcopy(self.dealer)
            p = deepcopy(self.played_cards)
            ps = deepcopy(self.players)
            lw = copy(self.last_winner)
            self.history.append((r, b, r_c, d, p, ps, lw))

        # Then we proceed to the next round
        self.current_player = self.round.proceed_round(self.players, action)
        self.played_cards = self.round.played_cards
        
        # If a round is over, we refresh the played cards
        if self.round.is_over():
            self.last_winner = (
                self.round.last_winner
                + self.judger.judge_round(self.round.played_cards, self.trump_card)
            ) % self.num_players
            self.round.last_winner = self.last_winner
            self.current_player = self.last_winner
            self.round.current_player = self.last_winner
            self.players[self.last_winner].tricks_won += 1
            self.previously_played_cards = self.previously_played_cards + self.played_cards
            self.played_cards = []
            self.round.played_cards = []
            self.round_counter += 1

            if self.round_counter >= self.round.round_number:
                round_scores = self.judger.judge_game(self.players)
                self.scores = list(np.array(self.scores) + np.array(round_scores))
                self.current_round += 1

                if self.current_round < len(self.round_sequence):
                    next_cards = self.round_sequence[self.current_round]
                    self.current_player = self.last_winner
                    state, _ = self._start_round(next_cards, keep_scores=True)
                else:
                    state = self.get_state(self.current_player)
                return state, self.current_player

        if (
            self.round.players_proposed == self.num_players
            and len(self.bids_history) == self.current_round
        ):
            self.bids_history.append(list(self.round.proposed_tricks))




        state = self.get_state(self.current_player)

        return state, self.current_player


    def get_state(self, player_id):
        ''' Return player's state

        Args:
            player_id (int): player id

        Returns:
            (dict): The state of the player
        '''

        state = self.round.get_state(self.players, player_id)
        state['current_player'] = self.round.current_player
        state['trump_card'] = self.trump_card.get_index()
        state['previously_played_cards'] = [c.get_index() for c in self.previously_played_cards]
        state['players_tricks_proposed'] = [player.proposed_tricks for player in self.players]
        state['players_previously_played_cards'] = [player.played_cards for player in self.players]
        return state

    
    def step_back(self):
        ''' Return to the previous state of the game

        Returns:
            (bool): True if the game steps back successfully
        '''
        if len(self.history) > 0:
            self.round, self.current_player, self.round_counter, self.dealer, self.played_cards, self.players, self.history_winners = self.history.pop()
            return True
        return False
    
    def get_num_players(self):
        ''' Return the number of players in Oh Hell

        Returns:
            (int): The number of players in the game
        '''
        return self.num_players

    
    def is_over(self):
        ''' Check if the game is over

        Returns:
            (boolean): True if the game is over
        '''

        return self.current_round >= len(self.round_sequence)

    def get_payoffs(self):
        ''' Return the scores of the players

        Returns:
            (list): The final scores of the players
        '''
        return tuple(self.scores)



    def play_automated(self):
        """Play until a manual slot is reached or the game finishes."""

        if not getattr(self, "players", None):
            state, player_id = self.init_game()
        else:
            player_id = self.current_player
            state = self.get_state(player_id)

        while not self.is_over():
            strategy = self.player_strategies[player_id]
            if strategy is None:
                return state, player_id

            action = strategy.select_action(self, player_id)
            state, player_id = self.step(action)

        self.scores = list(self.judger.judge_game(self.players))
        self.payoffs = tuple(self.scores)
        return state, player_id

    def play_full_game(self):
        """Play a complete automated game using configured strategies."""

        self.init_game()
        self.play_automated()
        if not self.is_over():
            raise RuntimeError("Automated play halted before completion")

        self.payoffs = tuple(self.judger.judge_game(self.players))
        self.scores = list(self.payoffs)
        return self.payoffs

    def get_legal_actions(self):
        ''' Return the legal actions for current player

        Returns:
            (list): A list of legal actions
        '''
        return self.round.get_legal_actions(self.players, self.round.current_player)

    @staticmethod
    def get_num_actions():
        ''' Return the number of applicable actions

        Returns:
            (int): The number of actions. There are at most 63 possible actions.
        '''
        return 63

    def get_player_id(self):
        ''' Return the current player's id

        Returns:
            (int): current player's id
        '''
        return self.round.current_player

