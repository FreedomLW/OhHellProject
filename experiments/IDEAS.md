# Experiment Ideas

`[ ]` todo  `[~]` in progress  `[x]` done  `[-]` abandoned

## Evaluation

1000 full games vs random mix of baselines (bc3 + rl_v1 + rl_chain_v3).
```
python scripts/evaluate_model.py --model <path> --vs-baselines 1000
```
Focus on **score**, not win rate.

---

- [x] Hybrid BC (heuristic bids + oracle play) → `bc_hybrid_stage3.zip`
- [x] RL fine-tune from BC, ~360K steps → `rl_finetune_v1.zip`
- [x] More BC data (5K seeds) — marginal improvement (52.0 vs 49.5)
- [x] Very low entropy RL chaining — works! 52→55→56→60. → `rl_chain_v3.zip`
- [-] Self-play BC iterations at 5K scale — unreliable, score dropped to 25.8
- [~] **RL chain v4**: one more pass from v3 to check plateau
- [ ] **RL with baseline opponents**: add bc3+rl_v1+rl_v3 to opponent pool
- [ ] **Per-trick reward shaping**: intermediate reward after each trick
- [ ] **Smarter bid teacher**: position-aware, suit void counting
- [ ] **Larger network**: net_arch=[256,256]
- [ ] **DAgger**: play → collect failures → relabel with oracle → retrain
- [ ] **Mixed opponent BC data**: gen data vs mix of heuristic+random+greedy+models
