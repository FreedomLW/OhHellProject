import os
import json
import numpy as np
from collections import OrderedDict

import rlohhell

from rlohhell.utils.utils import rank2int, int2rank
from rlohhell.games.base import Card

# In this variant diamonds are always trump
TRUMP_SUIT = 'D'

# Read required docs
ROOT_PATH = rlohhell.__path__[0]

# a map of abstract action to its index and a list of abstract action
with open(os.path.join(ROOT_PATH, 'games/ohhell/jsondata/action_space.json'), 'r') as file:
    ACTION_SPACE = json.load(file, object_pairs_hook=OrderedDict)
    ACTION_LIST = list(ACTION_SPACE.keys())


class Hand:
    def __init__(self, cards_left):
        self.cards_left = cards_left  # The set of cards not played yet
        

        self.RANK_TO_STRING = {6: "6", 7: "7", 8: "8", 9: "9",
                               10: "T", 11: "J", 12: "Q", 13: "K", 14: "A"}
        self.STRING_TO_RANK = {v: k for k, v in self.RANK_TO_STRING.items()}
        # Order used when sorting cards, 6 is lowest and Ace highest
        self.RANK_LOOKUP = "6789TJQKA"
        self.SUIT_LOOKUP = "SCDH"


    def _sort_cards(self):
        '''
        Sort all the seven cards ascendingly according to RANK_LOOKUP
        '''
        self.cards_left = sorted(
            self.cards_left, key=lambda card: self.RANK_LOOKUP.index(card[1]))


def determine_winner(played_cards, trump_card):
    '''
    Return the index of the player that wins in that round

    trump_card (Card): A list of just one card. The suit of this card is
        ignored because diamonds are always trump in this variant.
    played_cards (list): A list of cards played in the round so far
    '''

    trump_suit = TRUMP_SUIT
    first_suit = played_cards[0].suit

    if trump_suit is not None:
        trump_cards_played = [rank2int(card.rank) for card in played_cards if trump_suit == card.suit]
    else:
        trump_cards_played = []

    same_as_first_suit = [rank2int(card.rank) for card in played_cards if first_suit == card.suit]

    if trump_cards_played:
        highest = max(trump_cards_played)
        highest = int2rank(highest)
        return played_cards.index(Card(trump_suit, highest))
    else:
        highest = max(same_as_first_suit)
        highest = int2rank(highest)
        return played_cards.index(Card(first_suit, highest))
        
        
def cards2list(cards):
    ''' Get the corresponding string representation of cards

    Args:
        cards (list): list of UnoCards objects

    Returns:
        (string): string representation of cards
    '''
    cards_list = []
    for card in cards:
        cards_list.append(card.get_str())
    return cards_list


def trumps_in_hand(hand, trump_suit):
    ''' Return an array with the trumps from a given list'''
    trump_cards = [ card for card in hand if trump_suit == card.suit ]
    return trump_cards


def get_fixed_trump_card():
    '''Return the fixed trump card for this variant.'''
    return Card(TRUMP_SUIT, 'A')


