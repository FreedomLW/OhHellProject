# Harness packaging + pipeline alignment completed

Date: 2026-03-17

## What was fixed
- Reworked packaging metadata to be fully PEP 621 compliant in `pyproject.toml`.
- Removed conflicting legacy metadata from `setup.py` and left a compatibility shim.
- Added `AGENTS.md` mirror to satisfy harness contracts that reference plural naming.
- Updated CI harness checks to validate both `AGENT.md` and `AGENTS.md`.

## Why it failed before
`pip install -e .` was failing because setuptools attempted to merge partially-defined
`[project]` metadata from `pyproject.toml` with legacy `setup.py` fields, causing
`_MissingDynamic` warnings and a `NoneType` readme handling crash during editable build
requirements resolution.

## Result
Editable install now resolves metadata from one canonical source (`pyproject.toml`) and
avoids mixed-configuration failure mode.
