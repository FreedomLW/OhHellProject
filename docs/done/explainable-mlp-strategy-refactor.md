# Explainable MLP strategy refactor

## Summary
- Replaced ad-hoc dynamic placeholder objects in `ExplainableMLPStrategy` with explicit `SimpleNamespace` defaults for player and round state.
- Added a helper for closest legal bid selection to make bid fallback logic explicit and reusable.
- Hardened bidding against empty legal action lists by returning a safe default bid (`0`).
- Removed the old module-level `round_state_placeholder` helper now that defaults are encapsulated inside the strategy.

## Validation
- Added unit tests for:
  - empty legal bids fallback,
  - closest legal bid snapping,
  - illegal neural choice fallback to first legal card.
