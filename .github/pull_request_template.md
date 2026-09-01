<!-- Guides: docs/development/contributing.md and docs/development/pull-requests.md -->

## What this changes

<!-- The claim this change makes, and the reason for it. Name the file to read first. -->

## Evidence

<!-- What moved, what stayed identical, and the gate you ran. For example:
     pytest -m "not slow": 5081 passed, 113 skipped. -->

## Checks

- [ ] `ruff check .` and `ruff format --check .` pass
- [ ] `mypy` passes
- [ ] `pytest -m "not slow" -q` passes
- [ ] `python -m tests.prose` findings are fixed, or carry `accepted: <reason>` in `tests/prose-report.md`
- [ ] `nox -s docs` passes, if a document or a docstring changed
- [ ] `python -m build`, strict Twine validation, archive checks, and clean install checks pass
- [ ] regenerated artifacts carry matching manifest hashes and line-feed endings

<!-- CI runs all of these except the prose report. Running them locally first saves a round trip,
     and `nox -s docs` is the slowest one to learn about from a runner. -->
