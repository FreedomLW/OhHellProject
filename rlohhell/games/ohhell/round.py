# -*- coding: utf-8 -*-
''' Implement Oh Hell Round class
'''

import numpy as np
from rlohhell.games.base import Card
from rlohhell.games.ohhell.utils import TRUMP_SUIT
from rlohhell.utils.utils import rank2int

class OhHellRound:
    ''' Round can call other Classes' functions to keep the game running
    '''
    
    def __init__(self, round_number, num_players, np_random, dealer, last_winner=0, current_player=0):
        ''' Initilize the round class

        Args:
            round_number (int): The round number hence the max number of tricks 
            num_players (int): The number of players
            dealer (object): a custom object to deal cards to players
            last_winner (int): the winner of the last round and hence the starter
            current_player (int): the unique id of the current player
        '''
        self.np_random = np_random
        self.dealer = dealer

        # The list of cards played in the round so far
        self.played_cards = []

        # The card that determines the most powerful suit in the round
        self.trump_card = None

        self.round_number = round_number
        self.num_players = num_players
        self.last_winner = last_winner
        self.current_player = current_player

        # Array tracking the bids
        self.proposed_tricks = [0 for _ in range(self.num_players)]

        # The number of players that have bid so far
        self.players_proposed = 0

        # Track whether a high joker started the current trick
        self.joker_high_lead = False

    @staticmethod
    def _is_joker(card):
        return card.suit == 'S' and card.rank == '7'

    @staticmethod
    def _highest_trump(trump_cards):
        return max(trump_cards, key=lambda c: rank2int(c.rank))

    def _joker_options(self, card):
        low = Card(card.suit, card.rank, joker_mode='low')
        high = Card(card.suit, card.rank, joker_mode='high')
        return [low, high]

    def _unique_actions(self, actions):
        seen = set()
        unique_actions = []
        for card in actions:
            key = (card.suit, card.rank, getattr(card, 'joker_mode', None))
            if key not in seen:
                unique_actions.append(card)
                seen.add(key)
        return unique_actions

    def _actions_with_jokers(self, base_actions, hand):
        actions = list(base_actions)
        for card in hand:
            if self._is_joker(card):
                actions.extend(self._joker_options(card))
        return self._unique_actions(actions)

    def _disallowed_last_bid(self):
        """Return the bid value that would violate the sum rule for the last bidder.

        The classical Oh Hell rule forbids the combined bids from matching the
        number of cards dealt in the round. This helper calculates the single
        value that is off-limits for the final player to bid, or ``None`` if the
        constraint does not apply.
        """

        if self.players_proposed != self.num_players - 1:
            return None

        total_tricks = sum(self.proposed_tricks)
        disallowed_bid = self.round_number - total_tricks
        if 0 <= disallowed_bid <= self.round_number:
            return disallowed_bid

        return None


    def proceed_round(self, players, action):
        ''' Call other Classes's functions to keep one round running

        Args:
            action (str/int): The action(card) or bid choosen by the player
        '''

        legal_actions = self.get_legal_actions(players, self.current_player)
        is_lead = len(self.played_cards) == 0
        if is_lead:
            self.joker_high_lead = False

        # IF the action was a number it is treated a bid otherwise as a Card
        if isinstance(action, int):
            disallowed_bid = self._disallowed_last_bid()
            if disallowed_bid is not None and action == disallowed_bid:
                # Nudge the last bidder onto a legal value rather than raising
                # an exception.  This mirrors common table play where the
                # player is simply not permitted to lock in the forbidden
                # number.
                for alternative in legal_actions:
                    if alternative != disallowed_bid:
                        action = alternative
                        break

            if action not in legal_actions:
                raise Exception('{} is not legal action. Legal actions: {}'.format(action, legal_actions))

            players[self.current_player].proposed_tricks = action
            self.proposed_tricks[self.current_player] = action
            players[self.current_player].has_proposed = True
            self.players_proposed += 1
        else:
            if action not in legal_actions:
                raise Exception('{} is not legal action. Legal actions: {}'.format(action, legal_actions))

            if self._is_joker(action):
                action.joker_mode = getattr(action, 'joker_mode', 'low')
                if is_lead and action.joker_mode == 'high':
                    self.joker_high_lead = True

            players[self.current_player].played_cards.append(action)
            self.played_cards.append(action)
            players[self.current_player].hand.remove(action)

        self.current_player = (self.current_player + 1) % self.num_players 

        return self.current_player

    
    # Gets the avaiable actions of a player usually the current one
    def get_legal_actions(self, players, player_id):
        ''' Returns the list of actions possible for the player
        '''

        # OhHell doesn't allow the total tricks bid to equal the number of card deal to each player
        # this affects the last bidder's actions
        if players[player_id].has_proposed == False:
            full_list = list(range(0, self.round_number+1))
            disallowed_bid = self._disallowed_last_bid()
            if disallowed_bid is not None and disallowed_bid in full_list:
                full_list.remove(disallowed_bid)
            return full_list

        full_list = players[player_id].hand

        # If the player has proposed then available actions are a subset of the player's hand
        if player_id == self.last_winner:
            return self._actions_with_jokers(full_list, full_list)
        else:
            if len(self.played_cards) == 0:
                return self._actions_with_jokers(full_list, full_list)

            if self.joker_high_lead:
                trump_cards = [card for card in full_list if TRUMP_SUIT == card.suit]
                if trump_cards:
                    return [self._highest_trump(trump_cards)]
                return self._actions_with_jokers(full_list, full_list)

            starting_suit = self.played_cards[0].suit
            hand_same_as_starter = [card for card in full_list if starting_suit == card.suit]
            if hand_same_as_starter:
                return self._actions_with_jokers(hand_same_as_starter, full_list)
            else:
                return self._actions_with_jokers(full_list, full_list)

    def get_state(self, players, player_id):
        ''' Encode the state for the player

        Args:
            players (list): A list of the players
            player_id (int): The id of the player

        Returns:
            (dict): The state of the player
        '''

        # The details to be encoded from the round
        state = {}
        state['hand'] = players[player_id].hand
        state['played_cards'] = self.played_cards
        state['proposed_tricks'] = players[player_id].proposed_tricks
        state['tricks_won'] = players[player_id].tricks_won
        state['players_tricks_won'] = [player.tricks_won for player in players]
        state['legal_actions'] = self.get_legal_actions(players, player_id)
        return state

    def is_over(self):
        ''' Check whether the round is over

        Returns:
            (boolean): True if the current round is over
        '''
        if len(self.played_cards) == self.num_players:
            return True
        return False
