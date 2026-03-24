# Thoughts

## 2026-03-23: Research findings

From Judgement trick-taking AI (josh-holder): separate bid model trained on data,
heuristic card play baseline, per-trick feedback crucial. RL (DQN/A3C) plateaued.
Key: "trick-taking games provide rich feedback every card, not just end of round."

From RHC-AI game cup winner: BC→RL pipeline works, but RL needs KL penalty to
stay near BC policy. Entropy 0.00006 was optimal (very low). Without constraints
"any unconstrained learning will collapse in a few steps."

## Implications for our approach

1. Our RL ent_coef=0.01 may be WAY too high. RHC used 0.00006. Try 0.001 or lower.
2. Per-trick reward shaping: instead of score delta at round end, give intermediate
   signal after each trick (+1 if on track for bid, -1 if off track).
3. Our BC model already has strong card play. RL should ONLY refine bidding +
   edge cases. Keep it constrained.

## User-thoughts

1. Why accuracy always around 1.0? It doesn't usally for ML/RL
2. It's a good idea to teach models with previous versions (heuristics and others)? They simply less strong than the current approaches.
3. We should not look at the WR (because it depends on many factors and show bad how "good" model is. It strongly depends on who are we playing with). It is better to look at score. Because this metric is more isolated from the opponents (still depend but much less.) So, I recommended to retry your approach with self-play iteractions. And better to look at the score, not score diff.
4. Also I marked that you need to look at score at first. Not "WR against"
5. Don't write echos and comments for bash commands.

## 2026-03-23: Exp01 analysis + answers to user questions

**Why 100% accuracy**: The dataset is 36K-90K samples, the MLP has far more params
than needed to memorize this. It's pure overfitting — a lookup table in neural net
form. Fixes: way more data (millions), regularization, early stopping, smaller net.
Actually this may be fine for BC — we WANT it to memorize the oracle's moves exactly.
The real test is evaluation performance, not training accuracy.

**User Q2 — teaching with weaker opponents**: Good point. The oracle plays AGAINST
these opponents during data gen. If opponents are too weak (random), the oracle
wins easily and the model learns "obvious" plays. If opponents are stronger (our
baseline models), the oracle needs to be cleverer, teaching harder lessons. BUT
exp01 showed self-play makes it worse. The issue: the model overfits to beating
itself, then loses to different strategies. Better approach: MIXED opponent pool
(heuristic + random + conservative + greedy + models) during data gen.

**5K seeds result**: 0.511 WR slightly better than bc3 (0.496). More data helps
marginally. Self-play iterations made it WORSE (0.312 at iter 1).
Self-play BC is unreliable at scale. Focus on RL fine-tuning instead.

## 2026-03-23: RL chaining works!

**Key discovery**: chaining RL fine-tuning passes with decreasing entropy works.

| Model | Entropy | WR vs baselines | Score |
|-------|---------|----------------|-------|
| BC 5K base | — | 0.511 | 52.0 |
| RL chain v1 (exp02) | 0.001 | 0.556 | 54.8 |
| RL chain v2 (exp03) | 0.0005 | 0.563 | 56.0 |
| RL chain v3 (exp04) | 0.0003 | 0.591 | 59.7 |

Each pass: 500K steps, 16 envs, decreasing entropy. The model keeps finding
marginal improvements by fine-tuning its strategy while staying close to the BC
base. This is exactly what RHC-AI described: "constrained learning near BC policy."

**Next question**: does this plateau? How many more passes can we chain?

## 2026-03-23: Plateau + mixed opponents

**RL chain plateaued at v3** (score 59.7). v4 got 59.1 — no improvement. The
entropy is now so low (0.0002) that the model barely explores.

**Mixed opponent BC data didn't help**: score 52.1 (same as heuristic-only).
After RL: 57.9 (worse than chain's 59.7). The diverse game states didn't
translate to a better model. The pure heuristic BC data + RL chain was better.

**Current ceiling: ~60 score**. To break through, need a fundamentally different
approach. Options:
1. Per-trick reward shaping (code change to env) — richer learning signal
2. Larger network — maybe the MLP can't represent subtler strategies
3. Smarter bidding teacher — the HeuristicStrategy bid formula is crude
4. Much more RL steps (1M+) with the current best as starting point