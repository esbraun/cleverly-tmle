"""Task automation for cleverly.

The sessions here mirror the GitHub Actions workflows: ``lint``/``typecheck``/``tests``
run on every push, ``slow`` is the nightly statistical-validation tier.
"""

from __future__ import annotations

import nox

nox.options.default_venv_backend = "uv|virtualenv"
nox.options.sessions = ["lint", "typecheck", "tests"]

PYTHONS = ["3.11", "3.12", "3.13"]


#: Pinned exactly, and the same values as ``pyproject.toml``'s ``dev`` extra and
#: ``.github/workflows/ci.yml``.  All three name the toolchain and all three have to move
#: together: these sessions used to install ``ruff>=0.6`` and ``mypy>=1.11``, so
#: ``nox -s lint`` resolved to whatever was current on PyPI and could pass against a
#: formatter CI rejects -- which is the exact failure the pins exist to prevent.
RUFF = "ruff==0.16.1"
MYPY = "mypy==1.19.1"


@nox.session
def lint(session: nox.Session) -> None:
    session.install(RUFF)
    session.run("ruff", "check", ".")
    session.run("ruff", "format", "--check", ".")


@nox.session
def typecheck(session: nox.Session) -> None:
    session.install("-e", ".[all]", MYPY)
    session.run("mypy", "src/cleverly")


@nox.session(python=PYTHONS)
def tests(session: nox.Session) -> None:
    """Fast tier: everything except the statistical validation runs."""
    session.install("-e", ".[dev]")
    # ``-n auto`` as CI runs it.  The inner ``n_jobs=2`` on the simulation studies is
    # load-balancing the tail *under* xdist (CLAUDE.md sets out the measurement), so a
    # session without it is not the tier that was benchmarked.
    session.run("pytest", "-m", "not slow", "-q", "-n", "auto", *session.posargs)


@nox.session
def slow(session: nox.Session) -> None:
    """Nightly tier: coverage, consistency and type I error studies."""
    session.install("-e", ".[dev]")
    session.run("pytest", "-m", "slow", "-q", "-n", "auto", *session.posargs)


@nox.session
def bench(session: nox.Session) -> None:
    session.install("-e", ".[all]")
    session.run("python", "benchmarks/bench_tmle.py", *session.posargs)


@nox.session(name="bench-drtmle")
def bench_drtmle(session: nox.Session) -> None:
    """Characterise how the doubly-robust alternation exits.  Tens of minutes at defaults."""
    session.install("-e", ".[dev]")
    session.run("python", "benchmarks/bench_drtmle.py", *session.posargs)
