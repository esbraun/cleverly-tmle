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


def _workers() -> str:
    """Workers for ``-n auto``, sized from the cores this process may actually use.

    ``pytest-xdist`` resolves ``auto`` through ``psutil`` if it is installed and
    ``os.cpu_count()`` otherwise -- ``psutil`` is not a dependency of this project, so it is
    always the second one, and that reports the *host's* cores.  Inside a container with a
    CPU quota, which is every CI runner and the sandbox this repository is developed in,
    that asks for several times the parallelism the job can be given.

    ``tests.parallel`` goes through joblib, and joblib through loky, which reads the quota
    and the affinity mask.  Imported lazily and defensively: a session that runs before the
    test extras are installed should not fail on this, it should fall back to ``auto``'s own
    answer.
    """
    try:
        from tests.parallel import worker_count
    except Exception:  # pragma: no cover - nox runs outside the package's environment
        import os

        return str(os.cpu_count() or 1)
    return str(worker_count())


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
    # ``-n auto`` as CI runs it, with the worker count sized from the cores this process
    # may actually use -- xdist resolves ``auto`` through ``os.cpu_count()`` here (psutil is
    # not a dependency), which is the host count and ignores a container's CPU quota.
    # The inner ``n_jobs=2`` on the simulation studies is load-balancing the tail *under*
    # xdist (CLAUDE.md sets out the measurement), so a session without it is not the tier
    # that was benchmarked.
    session.run(
        "pytest",
        "-m",
        "not slow and not docs",
        "-q",
        "-n",
        "auto",
        *session.posargs,
        env={"PYTEST_XDIST_AUTO_NUM_WORKERS": _workers()},
    )


@nox.session
def slow(session: nox.Session) -> None:
    """Nightly tier: coverage, consistency and type I error studies."""
    session.install("-e", ".[dev]")
    session.run(
        "pytest",
        "-m",
        "slow",
        "-q",
        "-n",
        "auto",
        *session.posargs,
        env={"PYTEST_XDIST_AUTO_NUM_WORKERS": _workers()},
    )


@nox.session
def docs(session: nox.Session) -> None:
    """The guides' own examples, run.

    A separate tier from ``slow`` because it fails for a different reason and wants a
    different response: ``slow`` failing is a statistical result, this failing is a page
    that has stopped being true.  Its cost is the same order, though -- the README's first
    example is a ``default``-library fit at n=2000 -- which is why it is not in ``tests``.
    """
    session.install("-e", ".[dev]")
    session.run(
        "pytest",
        "-m",
        "docs",
        "-q",
        "-n",
        "auto",
        *session.posargs,
        env={"PYTEST_XDIST_AUTO_NUM_WORKERS": _workers()},
    )


@nox.session
def bench(session: nox.Session) -> None:
    """Where the time goes, and whether a compiled kernel would change that.

    ``.[bench]`` rather than ``.[all]``: it adds ``numba``, which the *Compiled kernels*
    section times a jitted Newton loop against numpy with.  The section prints a note and
    skips itself when numba is absent, so this still runs without it -- but then it
    answers one fewer question than it claims to.
    """
    session.install("-e", ".[bench]")
    session.run("python", "benchmarks/bench_tmle.py", *session.posargs)


@nox.session(name="bench-numba")
def bench_numba(session: nox.Session) -> None:
    """Whether numba or explicit parallelism helps *after* the nuisances are fitted.

    Defaults to ``benchmarks/configs/sandbox.json`` -- sizes to 100,000 and core counts to
    four, which is minutes rather than hours.  Pass ``-- --config benchmarks/configs/full.yaml``
    on a machine with cores to spare; that one is the sweep the report is meant to be
    regenerated from, and it needs ``pyyaml``, which this repository deliberately does not
    depend on.

    The correctness half of this investigation is **not** here: it is
    ``tests/unit/test_numba_benchmark.py``, which the ordinary ``tests`` session runs.
    Whether a compiled kernel computes what numpy computes is a correctness question and
    belongs where correctness questions are checked, not behind a benchmark session
    somebody has to remember to run.
    """
    session.install("-e", ".[bench]")
    args = session.posargs or ["--config", "benchmarks/configs/sandbox.json"]
    session.run("python", "-m", "benchmarks.numba.cli", *args)


@nox.session(name="bench-numba-pipelines")
def bench_numba_pipelines(session: nox.Session) -> None:
    """The denominator: how much of a real fit the post-nuisance half is, per flavour.

    Runs at both learner presets, because a ``glm`` share alone is the standard way to
    mislead with this measurement -- it is the cheapest preset available and inflates every
    package-owned share several-fold.
    """
    session.install("-e", ".[bench]")
    args = session.posargs or ["--pipeline-libraries", "glm", "default"]
    session.run("python", "-m", "benchmarks.numba.cli", "--pipelines", *args)
