"""Core game logic for the simplified Oh Hell environment.

This module previously modelled a single deal of the game with a fixed
number of tricks.  In Odessa poker however a complete ``game`` consists of
multiple deals.  The number of cards dealt to each player starts at one and
increases by one card each deal until a maximum value is reached.  After the
maximum a number of full deals equal to the number of players are played and
finally the deal size decreases again until it reaches one card.  For
example, with four players the sequence of deals is ::

    1,2,3,4,5,6,7,8,9,9,9,9,9,8,7,6,5,4,3,2,1

The original implementation in this repository only supported a single deal
with a fixed number of tricks.  The aim of this patch is to extend the game
class so that it can run through the full sequence of deals within one game
instance.  In addition a score board is maintained across the deals so that
the final result of the whole game can be returned when the last deal has
finished.

The implementation below borrows heavily from the previous version but adds
the following major changes:

* ``round_sequence`` – a pre–computed list containing the number of cards to
  deal in each round of the game.
* ``current_round`` and ``tricks_played`` – counters keeping track of the
  current round index and the number of tricks already played in the current
  round.
* ``scores`` – cumulative scores for all players that persist across rounds.
* Logic in :func:`init_game` and :func:`step` to automatically progress from
  one round to the next, reshuffling and redealing the deck when needed.

The public API of the class remains largely unchanged which allows existing
agents and environments to keep functioning.  The new attributes are exposed
through the state dictionary so that learning agents can make use of the
additional information if desired.
"""

from copy import deepcopy, copy
import random
import numpy as np

from rlohhell.games.ohhell import Dealer, Player, Judger, Round


class OhHellGame:
    """Environment class modelling a full game of Odessa poker."""

    def __init__(self, allow_step_back: bool = False, num_players: int = 4):
        self.allow_step_back = allow_step_back
        self.np_random = np.random.RandomState()
        self.num_players = num_players

        # Runtime attributes initialised in :meth:`init_game`
        self.dealer = None
        self.players = []
        self.judger = None
        self.round = None

        # Round/score bookkeeping
        self.round_sequence = []           # pre-computed list of cards per round
        self.current_round = 0             # index into ``round_sequence``
        self.tricks_played = 0             # tricks completed in current round
        self.scores = [0 for _ in range(num_players)]

        self.payoffs = [0 for _ in range(num_players)]
        self.current_player = random.randint(0, self.num_players - 1)
        self.game_over = False

        # The following variables are created in ``init_game`` but defined here
        # for type checking tools and clarity
        self.played_cards = []
        self.previously_played_cards = []
        self.trump_card = None
        self.last_winner = 0
        self.history = []

    # ---------------------------------------------------------------------
    # Utility methods
    # ---------------------------------------------------------------------
    def _compute_round_sequence(self):
        """Create the list describing the number of cards in each round.

        Odessa poker uses a 36 card deck.  The maximum number of cards that
        can be dealt to each player is therefore ``36 // num_players``.  After
        reaching this maximum we play ``num_players`` additional full rounds
        and then decrease back to one card.
        """

        max_cards = 36 // self.num_players
        ascending = list(range(1, max_cards + 1))
        plateau = [max_cards] * self.num_players  # additional full rounds
        descending = list(range(max_cards - 1, 0, -1))
        return ascending + plateau + descending

    def _start_new_round(self, num_cards: int):
        """Reset deck and players for a new round dealing ``num_cards`` cards."""

        # Fresh deck and shuffle
        self.dealer = Dealer(self.np_random)
        self.dealer.shuffle()

        # Reset players' round specific variables
        for player in self.players:
            player.hand = []
            player.played_cards = []
            player.has_proposed = False
            player.proposed_tricks = 0
            player.tricks_won = 0

        # Deal cards for the new round
        for i in range(num_cards * self.num_players):
            self.players[i % self.num_players].hand.append(self.dealer.deal_card())

        # If the deck is exhausted there is no trump card this round
        if len(self.dealer.deck) > 0:
            self.trump_card = self.dealer.flip_trump_card()
        else:
            self.trump_card = None
        self.played_cards = []
        self.round = Round(
            round_number=num_cards,
            num_players=self.num_players,
            np_random=self.np_random,
            dealer=self.dealer,
            last_winner=self.current_player,
            current_player=self.current_player,
        )
        self.tricks_played = 0

    # ------------------------------------------------------------------
    # Public API expected by RLCard style environments
    # ------------------------------------------------------------------
    def configure(self, game_config):
        self.num_players = game_config['game_num_players']

    # Game initialisation ------------------------------------------------
    def init_game(self):
        """Initialise a brand new game and return the first state."""

        self.dealer = Dealer(self.np_random)
        self.players = [Player(i, self.np_random) for i in range(self.num_players)]
        self.judger = Judger(self.np_random)

        self.round_sequence = self._compute_round_sequence()
        self.current_round = 0
        self.tricks_played = 0
        self.scores = [0 for _ in range(self.num_players)]
        self.game_over = False
        self.previously_played_cards = []
        self.history = []

        # Determine starting player at random
        self.current_player = random.randint(0, self.num_players - 1)
        self.last_winner = self.current_player

        # Set up the first round with a single card per player
        self._start_new_round(self.round_sequence[self.current_round])

        player_id = self.round.current_player
        state = self.get_state(player_id)
        return state, player_id

    # Game progression ---------------------------------------------------
    def step(self, action):
        """Advance the game by applying ``action`` for the current player."""

        if self.allow_step_back:
            # Snapshot current state for potential rollback
            snapshot = (
                deepcopy(self.round),
                self.current_player,
                self.tricks_played,
                deepcopy(self.dealer),
                deepcopy(self.played_cards),
                deepcopy(self.players),
                copy(self.last_winner),
                self.current_round,
                list(self.scores),
                deepcopy(self.previously_played_cards),
            )
            self.history.append(snapshot)

        # Proceed with the action
        self.current_player = self.round.proceed_round(self.players, action)
        self.played_cards = self.round.played_cards

        if self.round.is_over():
            # Determine trick winner
            winner_offset = self.judger.judge_round(self.round.played_cards, self.trump_card)
            self.last_winner = (self.round.last_winner + winner_offset) % self.num_players
            self.round.last_winner = self.last_winner
            self.current_player = self.last_winner
            self.round.current_player = self.last_winner
            self.players[self.last_winner].tricks_won += 1

            # Record played cards and reset for next trick
            self.previously_played_cards += self.played_cards
            self.played_cards = []
            self.round.played_cards = []

            self.tricks_played += 1

            # If all tricks for this round have been played, score the round
            if self.tricks_played >= self.round.round_number:
                round_scores = self.judger.judge_game(self.players)
                for i, s in enumerate(round_scores):
                    self.scores[i] += s

                # Move to the next round or finish the game
                self.current_round += 1
                if self.current_round >= len(self.round_sequence):
                    self.game_over = True
                else:
                    self._start_new_round(self.round_sequence[self.current_round])

        state = self.get_state(self.current_player)
        return state, self.current_player

    # Game state ---------------------------------------------------------
    def get_state(self, player_id):
        state = self.round.get_state(self.players, player_id)
        state['current_player'] = self.round.current_player
        state['trump_card'] = self.trump_card
        state['previously_played_cards'] = self.previously_played_cards
        state['players_tricks_proposed'] = [p.proposed_tricks for p in self.players]
        state['players_previously_played_cards'] = [p.played_cards for p in self.players]
        state['round_index'] = self.current_round
        state['scores'] = list(self.scores)
        return state

    # ------------------------------------------------------------------
    def step_back(self):
        if not self.history:
            return False
        (
            self.round,
            self.current_player,
            self.tricks_played,
            self.dealer,
            self.played_cards,
            self.players,
            self.last_winner,
            self.current_round,
            self.scores,
            self.previously_played_cards,
        ) = self.history.pop()
        self.game_over = False
        return True

    # Simple accessors --------------------------------------------------
    def get_player_id(self):
        return self.current_player

    def get_num_players(self):
        return self.num_players

    def is_over(self):
        return self.game_over

    def get_payoffs(self):
        return tuple(self.scores)

    def get_legal_actions(self):
        return self.round.get_legal_actions(self.players, self.round.current_player)

    @staticmethod
    def get_num_actions():
        # A 36-card deck yields 36 card actions and 9 bidding actions
        return 45

