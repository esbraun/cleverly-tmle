"""Task automation for cleverly.

The sessions here mirror the GitHub Actions workflows: ``lint``/``typecheck``/``tests``
run on every push, ``slow`` is the nightly statistical-validation tier.
"""

from __future__ import annotations

import nox

nox.options.default_venv_backend = "uv|virtualenv"
nox.options.sessions = ["lint", "typecheck", "tests"]

PYTHONS = ["3.10", "3.11", "3.12", "3.13"]


@nox.session
def lint(session: nox.Session) -> None:
    session.install("ruff>=0.6")
    session.run("ruff", "check", ".")
    session.run("ruff", "format", "--check", ".")


@nox.session
def typecheck(session: nox.Session) -> None:
    session.install("-e", ".[all]", "mypy>=1.11")
    session.run("mypy", "src/cleverly")


@nox.session(python=PYTHONS)
def tests(session: nox.Session) -> None:
    """Fast tier: everything except the statistical validation runs."""
    session.install("-e", ".[dev]")
    session.run("pytest", "-m", "not slow", "-q", *session.posargs)


@nox.session
def slow(session: nox.Session) -> None:
    """Nightly tier: coverage, consistency and type I error studies."""
    session.install("-e", ".[dev]")
    session.run("pytest", "-m", "slow", "-q", *session.posargs)


@nox.session
def bench(session: nox.Session) -> None:
    session.install("-e", ".[all]")
    session.run("python", "benchmarks/bench_tmle.py", *session.posargs)


@nox.session(name="bench-drtmle")
def bench_drtmle(session: nox.Session) -> None:
    """Characterise how the doubly-robust alternation exits.  Tens of minutes at defaults."""
    session.install("-e", ".[all]")
    session.run("python", "benchmarks/bench_drtmle.py", *session.posargs)
