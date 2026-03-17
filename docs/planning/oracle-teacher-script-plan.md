# Plan: oracle teacher training script and parameter search

Date: 2026-03-17
Owner: Codex agent

## Goal
Add a script that can teach a student model via the existing oracle bootstrapping loop and also verify/search for optimal oracle training parameters.

## Planned steps
1. Extend curriculum evaluation module with a parameter search helper over iteration and rollout candidates.
2. Add a script in `scripts/` to run fixed training or search mode and print/save JSON reports.
3. Add tests for parameter search helper and script behavior.
4. Record completion in `docs/done/`.
