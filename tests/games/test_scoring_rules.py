from rlohhell.games.ohhell.judger import OhHellJudger as Judger
from rlohhell.games.ohhell.player import OhHellPlayer as Player


def test_round_scoring_matches_examples():
    rng = None
    players = [Player(i, rng) for i in range(3)]

    players[0].proposed_tricks = 2
    players[0].tricks_won = 2

    players[1].proposed_tricks = 3
    players[1].tricks_won = 1

    players[2].proposed_tricks = 1
    players[2].tricks_won = 2

    scores = Judger(rng).judge_game(players)

    assert scores == (20, -30, 2)
    assert players[0].tricks_won == 2
    assert players[1].tricks_won == 1
    assert players[2].tricks_won == 2
