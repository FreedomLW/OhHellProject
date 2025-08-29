from rlohhell.utils.utils import init_short_deck
from rlohhell.games.base import Card

class OhHellDealer:

    def __init__(self, np_random):
        ''' Initialize a ohhell dealer class
        '''
        self.np_random = np_random
        # Odessa poker uses a 36-card deck from 6 through Ace
        self.deck = init_short_deck()
        self.shuffle()

    def shuffle(self):
        ''' Shuffle the deck
        '''
        self.np_random.shuffle(self.deck)

    def flip_trump_card(self):
        """Return the trump card for a new game.

        The original implementation flipped a random card from the deck to
        determine the trump suit. In this variant the trump suit is fixed to
        diamonds, so we simply return a diamond card without altering the deck.

        Returns:
            Card: A :class:`Card` object representing the trump. The rank is
            irrelevant for trump determination, so the lowest rank is used.
        """
        # Always return a diamond card
        return Card('D', '6')

    def deal_cards(self, player, num):
        ''' Deal some cards from deck to one player

        Args:
            player (object): The object of DoudizhuPlayer
            num (int): The number of cards to be dealed
        '''
        for _ in range(num):
            player.hand.append(self.deck.pop())

    
    def deal_card(self):
        ''' Deal one card from the deck

        Returns:
            (Card): The drawn card from the deck
        '''
        return self.deck.pop()
