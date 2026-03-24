# Experiment Results

## Baselines

| Model | Path | Score vs baselines (1000g) |
|-------|------|---------------------------|
| bc_hybrid_stage3 | `models/bc_hybrid_stage3.zip` | 49.5 |
| rl_finetune_v1 | `models/rl_finetune_v1.zip` | 53.2 |
| rl_chain_v3 | `models/rl_chain_v3.zip` | 59.7 |

## Experiments

| ID | Idea | Score vs baselines | Notes |
|----|------|--------------------|-------|
| bc3 (ref) | baseline bc | 49.5 | reference |
| rl_v1 (ref) | baseline rl | 53.2 | reference |
| exp01_5k | 5K seeds, heur opp | 52.0 | more data helps slightly |
| exp01_5k_sp | 5K seeds, self-play 1 | 25.8 | self-play hurt |
| exp01_5k_sp2 | 5K seeds, self-play 2 | 44.9 | recovered but worse |
| exp02 | Low-ent RL (0.001) from 5K BC | 54.8 | |
| exp03 | RL chain v2 (ent=0.0005) | 56.0 | |
| **exp04** | **RL chain v3 (ent=0.0003)** | **59.7** | **current best** |
| exp05 | RL chain v4 (ent=0.0002) from v3 | 59.1 | plateau — no improvement |
| exp06_bc | Mixed opponent BC (5K, 7 opponents) | 52.1 | same as heuristic-only BC |
| exp06_rl | RL from mixed-opp BC (ent=0.001) | 57.9 | worse than chain approach |

## Key findings

1. Oracle bids are poison — use hybrid labels (heuristic bids + oracle play).
2. Self-play BC iterations unreliable at scale — focus on RL fine-tuning.
3. RL chaining with decreasing entropy works: 52→55→56→60 score progression.
4. Focus on score, not win rate (user insight).
