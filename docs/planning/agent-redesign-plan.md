# Agent redesign plan (next iteration)

## Goals
- Expand baseline opponent diversity with a **bruteforce tactical agent**.
- Remove policy collapse around bids `{0, 1}` by introducing **bid-target randomization**.
- Split learning into a **two-stage reward curriculum**:
  1. maximize trick-taking ability,
  2. optimize bid accuracy / score.
- Borrow robust evaluation ideas from RHC-AI style tournamenting and scenario batching.

## Proposed work packages
1. **Environment API extension**
   - Add configurable reward mode in env config (`"trick_max"`, `"bid_accuracy"`, `"hybrid"`).
   - Add optional per-episode sampled bid target profile for curriculum runs.
   - Expose richer trick context in observations (lead suit, current winning card, remaining hand suit counts).

2. **Bruteforce tactical agent**
   - Add `rlohhell/heuristics/bruteforce_agent.py` with bounded-depth search over legal cards.
   - Use fast rollout scoring objective:
     - immediate trick win probability,
     - future hand control potential,
     - trump preservation penalty.
   - Keep latency bounded using beam width + memoization by `(hand_signature, trick_state)`.

3. **Random bins setup**
   - During bidding, sample target bins (per seat) from calibrated distribution instead of always anchoring low bids.
   - Add training callback metrics: bid entropy, mean absolute bid error, realized trick distribution.

4. **Reward curriculum**
   - Stage A reward emphasizes trick count accumulation.
   - Stage B interpolates toward canonical Oh Hell scoring.
   - Add schedule knob (`--reward-curriculum`) in `scripts/train_maskable_self_play.py`.

5. **Evaluation protocol improvements**
   - Fixed-seed batch evaluation (CRN) for fair model comparisons.
   - Opponent portfolio evaluation: random / param bot / bruteforce / latest checkpoint.
   - Report robust metrics (mean, p10, p90, exploitability proxy).

## Acceptance criteria
- New env knobs are documented and covered by tests.
- New brute-force agent is pluggable via opponent pool.
- Training script can run curriculum mode end-to-end.
- Evaluation report includes the added robustness metrics.
