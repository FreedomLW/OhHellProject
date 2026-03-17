# AGENT.md

## Project overview
OhHellProject is a reinforcement-learning and simulation codebase for the card game **Oh Hell**.
It contains:
- Core game logic (`rlohhell/games/ohhell`)
- RL/Gym environments (`rlohhell/envs`)
- Policies and self-play training scripts (`rlohhell/policies`, `scripts/`)
- Evolutionary training/evaluation pipeline (`rlohhell/evo`)
- Automated tests (`tests/`)

## Agent operating contract (OpenAI Harness style)
When working in this repository, the agent should:
1. Read project docs first (`README.md`, `ARCHITECTURE.md`, `RULES.md`, `docs/README.md`).
2. Keep docs and implementation synchronized:
   - design/plans in `docs/planning/`
   - active/in-progress notes in `docs/wip/`
   - completed decisions/tasks in `docs/done/`
   - ideas and experiments in `docs/research/`
3. Run verification checks after every meaningful change:
   - run tests (`pytest`)
   - ensure required docs exist and are updated
4. Prefer small, explicit commits with clear scope.

## Definition of done for tasks
A task is considered done only when all are true:
- Code and docs match current behavior and architecture.
- Tests pass locally.
- Planning entry was moved/recorded as done in `docs/done/`.
- Any remaining uncertainty is captured in `docs/research/` or a TODO list.

## Important project commands
- Install: `uv pip install -r requirements.txt && uv pip install -e .`
- Run tests: `pytest`
- Train self-play model: `python scripts/train_maskable_self_play.py --help`
- Evolutionary training: `rlohhell-train-evo --help`

## If something is missing
If required documentation, rules, or architecture details are missing or ambiguous:
- note the gap in `docs/research/missing-items.md`
- add a proposed fix path and owner/action suggestion
