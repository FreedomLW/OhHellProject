# RULES.md

## Core game constants
- Deck size is **36 cards**.
- **Trump suit is always Diamonds**.
- Number of players is a **pre-selected constant** for the match/configuration.

## Bidding rule
- Standard “last bidder cannot make the total bids equal to total tricks” rule applies.
- **Exception:** this rule does **not** apply when the number of cards in the round is less than 4.

## Round structure
Each round has two phases:
1. **Bidding phase** — players announce their trick prediction.
2. **Play phase** — players play one card each trick.

## Trick-play rules
- The leader can play **any** card.
- Other players should follow the led suit when possible.
- If a player cannot follow led suit, they should play trump suit if they have it.
- If neither is available, they may play any card.

## Joker behavior
Joker is a special card and is **not treated as Clubs by default**.
A player may choose Joker mode when playing it:
- **Higher mode**: Joker is used as the highest card in the requested context.
- **Simple mode**: Joker is treated as a simple **7 of Clubs**.

If the leader starts with Joker, they may explicitly declare one of:
- “higher of any suit” (leader names the suit)
- “simple 7 of clubs”

For “higher of any suit”:
- Players who have the declared suit should play a **higher card of that suit**.
- Players who do not have the declared suit may play any other card.
- The Joker card always wins that trick.

That declaration defines how the opening Joker should be interpreted for that trick.

## Notes
- This document reflects required project rules for current PR correction.
- If code differs from these rules, implementation should be aligned in a follow-up change.
