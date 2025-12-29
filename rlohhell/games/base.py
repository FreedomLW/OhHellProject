''' Game-related base classes'''
from enum import Enum


class Suit(Enum):
    '''Enumeration of card suits.'''
    SPADES = 'S'
    HEARTS = 'H'
    DIAMONDS = 'D'
    CLUBS = 'C'
    BLACK_JOKER = 'BJ'
    RED_JOKER = 'RJ'


class Rank(Enum):
    '''Enumeration of card ranks for the short deck (6 through Ace).'''
    SIX = '6'
    SEVEN = '7'
    EIGHT = '8'
    NINE = '9'
    TEN = 'T'
    JACK = 'J'
    QUEEN = 'Q'
    KING = 'K'
    ACE = 'A'


class Card:
    '''Card stores the suit and rank of a single card.

    Note:
        The suit variable should be one of ``[S, H, D, C, BJ, RJ]`` meaning
        ``[Spades, Hearts, Diamonds, Clubs, Black Joker, Red Joker]``.
        The rank variable uses a short deck from ``6`` through ``A``.
    '''
    suit = None
    rank = None
    valid_suit = [s.value for s in Suit]
    valid_rank = [r.value for r in Rank]

    def __init__(self, suit, rank, joker_mode=None):
        ''' Initialize the suit and rank of a card

        Args:
            suit: string, suit of the card, should be one of valid_suit
            rank: string, rank of the card, should be one of valid_rank
        '''
        self.suit = suit
        self.rank = rank
        # For special cards like the 7♠ joker we track whether it is played as
        # the highest or lowest card in the trick.
        self.joker_mode = joker_mode

    def __eq__(self, other):
        if isinstance(other, Card):
            return self.rank == other.rank and self.suit == other.suit
        else:
            # don't attempt to compare against unrelated types
            return NotImplemented

    def __hash__(self):
        suit_index = Card.valid_suit.index(self.suit)
        rank_index = Card.valid_rank.index(self.rank)
        return rank_index + 100 * suit_index

    def __str__(self):
        ''' Get string representation of a card.

        Returns:
            string: the combination of rank and suit of a card. Eg: AS, 5H, JD, 3C, ...
        '''
        return self.suit + self.rank

    def get_index(self):
        ''' Get index of a card.

        Returns:
            string: the combination of suit and rank of a card. Eg: 1S, 2H, AD, BJ, RJ...
        '''
        return self.suit+self.rank
