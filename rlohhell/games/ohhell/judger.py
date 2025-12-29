from rlohhell.games.ohhell.utils import determine_winner

class OhHellJudger:
    ''' The Judger class for Oh Hell!
    '''
    def __init__(self, np_random):
        ''' Initialize a judger class
        '''
        self.np_random = np_random

    def judge_round(self, played_cards, trump_card):
        ''' Returns the position of winner of the round relative to the first card player

        Args:
            played_cards (list): The list of cards played
            trump_card (card): The trump card for the game
        '''

        winner = determine_winner(played_cards, trump_card)

        return winner



    def judge_game(self, players):
        '''Calculate round scores according to Oh Hell rules.

        A player scores ``10 * bid`` points for matching their bid exactly.
        If they win more tricks than they bid they receive a consolation equal
        to the number of tricks taken.  Missing the bid on the low side costs
        ``10 * bid`` points.

        Args:
            players (list): The list of players who play the game
        Returns:
            tuple: Round scores for each player in order
        '''

        round_scores = []

        for player in players:
            bid = player.proposed_tricks
            tricks = player.tricks_won

            if tricks == bid:
                score = 10 * bid
            elif tricks > bid:
                score = tricks
            else:
                score = -10 * bid

            round_scores.append(score)

        return tuple(round_scores)
