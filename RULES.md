# RULES.md

## Deck

36 cards: ranks **6, 7, 8, 9, 10, J, Q, K, A** in four suits
(Spades, Hearts, Diamonds, Clubs).

**Joker** is represented as 7 of Spades with a mode flag.
Default mode is **Higher** (highest card in the trick).

## Trump

Trump suit is always **Diamonds**.

## Players

Number of players is a pre-selected constant for the match (default: 4).

## Round sequence

A full game consists of three phases determined by the number of players
and deck size. Let `max = 36 / num_players`:

| Phase | Cards per round | Count |
|-------|-----------------|-------|
| Climb | 1, 2, …, max−1 | max−1 |
| Plateau | max (one per player as dealer) | num_players |
| Descend | max−1, …, 2, 1 | max−1 |

**Total rounds:** `2 × (max − 1) + num_players`

Example (4 players): climb [1..8] + plateau [9,9,9,9] + descend [8..1] = **20 rounds**.

## Bidding

Each round starts with a bidding phase where players announce their trick
prediction in turn order.

**Last-bidder constraint:** the last player to bid cannot make the total of
all bids equal to the number of tricks in the round.

**Exception:** this constraint does **not** apply when the round size is less
than 4 cards.

## Trick play

- The leader may play **any** card.
- Other players must follow the led suit if possible.
- If a player cannot follow the led suit, they must play a trump card if
  they have one.
- If neither the led suit nor trump is available, they may play any card.

### Trick winner

The highest trump card played wins the trick. If no trump was played,
the highest card of the led suit wins. Cards of other suits have no
strength.

## Joker behavior

The Joker is a special card. A player may choose one of two modes when
playing it:

- **Higher mode** (default): the Joker acts as the highest card in the
  requested context and always wins the trick.
- **Simple mode**: the Joker is treated as a plain 7 of Clubs.

When the leader opens with the Joker they declare one of:

- **"Higher of [suit]"** — players who hold the declared suit must play a
  higher card of that suit; others may play any card. The Joker always
  wins.
- **"Simple 7 of Clubs"** — the Joker acts as 7♣ and the trick proceeds
  normally.

## Scoring

At the end of each round every player is scored:

| Condition | Score |
|-----------|-------|
| `tricks == bid` (bid > 0) | `10 × bid` |
| `tricks == bid == 0` | `5` |
| `tricks > bid` | `tricks` (consolation) |
| `tricks < bid` | `−10 × bid` (penalty) |

## Win condition

The player with the **highest cumulative score** across all rounds wins the
game.
