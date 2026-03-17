# Plan: simplify oracle teacher seed usage

Date: 2026-03-17
Owner: Codex agent

## Goal
Make oracle teacher CLI easier by defaulting to one seed and keeping multi-seed support optional.

## Steps
1. Replace default explicit seed list with `--seed` + `--num-seeds`.
2. Keep `--seeds` as optional override for advanced use.
3. Update tests and README examples.
4. Record completion in `docs/done/`.
