# Plan: exhaustive play-phase oracle and hand sampling controls

Date: 2026-03-17
Owner: Codex agent

## Goal
Update oracle training to prioritize maximum tricks (bins) in play phase via explicit action-tree search for the target player, and add hand-sample controls.

## Steps
1. Extend oracle dataset generator with play-phase-only mode and hand-samples-per-seed.
2. Add exhaustive target-player search over legal card plays (max bins objective).
3. Wire new options into curriculum and teacher CLI.
4. Add/update tests and docs.
