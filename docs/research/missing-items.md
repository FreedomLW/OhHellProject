# Missing / follow-up items

## Code bugs

1. **Round sequence off by one**: `game.py:54` uses `range(1, max_cards + 1)` producing
   climb [1..9] instead of [1..8]. The correct sequence is climb to `max - 1`, then
   plateau at `max`. Fix: change to `range(1, max_cards)`. Related tests in
   `test_ohhell_game.py` will need updating.

2. **Bidding exception not implemented**: `RULES.md` states the last-bidder constraint
   does not apply when round size < 4. Code in `round.py` (`_disallowed_last_bid`)
   always applies the constraint regardless of round size.

## Unfinished work

3. **Agent redesign plan**: implement work packages from
   `docs/planning/agent-redesign-plan.md` — bruteforce tactical agent, reward
   curriculum, bid-target randomization, robust evaluation protocol.

4. **BC pipeline (WP1, WP3–WP5)**: complete remaining work packages from
   `docs/planning/bruteforce-bc-realization-plan.md` — one-round runner, BC trainer,
   evaluation protocol, iterative relabeling loop.

## Low priority

5. Add lint / type-check standards to CI.
