# Contributing to cleverly

`cleverly` is alpha software for causal studies built on targeted maximum likelihood estimation.
One person maintains it, so a review can take several days. Open an issue before you start a large
change.

The full guides live in the documentation site.

| for | read |
| --- | --- |
| setup, branch names, the checks, documentation and docstring rules | [Contributing](https://esbraun.github.io/cleverly-tmle/development/contributing.html) |
| the commit style, the pull request body, what CI does and does not check | [Pull requests](https://esbraun.github.io/cleverly-tmle/development/pull-requests.html) |
| which tier a change has to satisfy | [Test tiers and gates](https://esbraun.github.io/cleverly-tmle/development/testing-strategy.html) |
| designing and registering a validation study | [Method benchmarking strategy](https://esbraun.github.io/cleverly-tmle/development/method-benchmarking.html) |

The sources are `docs/development/contributing.md` and `docs/development/pull-requests.md`.

## Set up

```bash
git clone https://github.com/esbraun/cleverly-tmle.git
cd cleverly-tmle
uv venv
uv pip install -e ".[dev,docs]"
```

## Run the checks

```bash
ruff check .
ruff format --check .
python -m tests.prose
mypy
pytest -m "not slow" -q
sphinx-build -W --keep-going -b html docs docs/_build/html
```

`nox` with no argument runs the `lint`, `typecheck`, `docs`, and `tests` sessions, which mirror the
CI jobs. `nox -s docs` runs the last command in an isolated environment, which is what CI uses. Run
one test tier at a time, because each tier expects the whole machine.

`python -m tests.prose` reports on the reader-facing prose. It changes nothing and it fails
nothing. Fix each finding, or record `accepted: <reason>` against it in `tests/prose-report.md`.
The fast tier fails only on a finding with no recorded judgment.

## Working agreements

`CLAUDE.md` holds the working agreements that no other file states and no test enforces. Read it
before you change scientific code, documentation prose, or a docstring.

## License

`cleverly` is under the [MIT License](LICENSE). Your contribution to the project is licensed
under the same terms.

One directory differs. `tests/canonical/` is under the
[GNU General Public License v3.0](tests/canonical/LICENSE), because its R runners call
reference packages in the same process. A contribution to that directory is licensed under
the GPL. No published distribution carries it.
