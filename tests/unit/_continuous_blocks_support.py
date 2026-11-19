"""Fixtures of ``tests/unit/test_continuous_survival_blocks.py`` and their golden numbers.

Every fixture is a fit through the public estimator only, so :func:`golden_records` runs the
same fits on the engine before blocks.  ``tests/unit/_continuous_dense_golden.py`` holds what
it printed there; regenerate it only from that engine (X6 part A's at ``d2b949e5``), with its
``src`` first on the path, from the repository root:
``PYTHONPATH=<checkout of d2b949e5>/src:. python -c "from tests.unit._continuous_blocks_support
import golden_records as g; g()"``.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np

from cleverly.survival import ContinuousSurvivalTMLE, StratifiedNelsonAalen
from cleverly.survival.data import ContinuousSurvivalData


def law(n: int, causes: int, seed: int = 1) -> ContinuousSurvivalData:
    """The scaling assessment's exponential law with ``causes`` competing causes."""
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(n, 2))
    a = rng.binomial(1, 1.0 / (1.0 + np.exp(-0.3 * w[:, 0])))
    times = [
        rng.exponential(1.0 / np.exp(-1.0 + 0.3 * w[:, 0] - 0.4 * a + 0.2 * w[:, 1]))
        for _ in range(causes)
    ]
    censor = rng.exponential(1.0 / 0.15, n)
    t = np.minimum.reduce([*times, censor])
    d = np.zeros(n, dtype=int)
    for j, tj in enumerate(times):
        d[tj == t] = j + 1
    return ContinuousSurvivalData(t, d, a, w, ("t", "d", "a", "w0", "w1"), None, "pandas")


#: name -> (units, causes, estimator settings, walk survival floor or None).
FIXTURES: dict[str, tuple[int, int, dict[str, Any], float | None]] = {
    "one cause, cross-fitted": (800, 1, {}, None),
    "two causes": (700, 2, {"cross_fit": False}, None),
    "three causes": (600, 3, {}, None),
    "stratified Nelson-Aalen": (
        700,
        1,
        {
            "hazard_library": [StratifiedNelsonAalen((0,))],
            "censoring_library": [StratifiedNelsonAalen((0,))],
        },
        None,
    ),
    "factual tilt": (600, 1, {"tilt": "factual"}, None),
    "band": (600, 2, {"simultaneous": True, "n_multiplier": 500}, None),
    "frozen pair": (600, 1, {"cross_fit": False, "step_size": 1.0}, 0.9),
}

#: Fixtures whose floats may differ from the dense walk in the last digits (contract section 3).
TOLERANT: dict[str, tuple[int, int, dict[str, Any], float | None]] = {
    "restricted mean": (700, 1, {"restricted_horizons": (2.0,)}, None),
}


#: Fixtures the golden file does not record: small ones for blocks of one to seven rows, and
#: one whose walk accepts a step with pairs held.
EXTRA: dict[str, tuple[int, int, dict[str, Any], float | None]] = {
    "small, two causes": (240, 2, {"cross_fit": False}, None),
    "small, restricted mean": (240, 1, {"restricted_horizons": (2.0,)}, None),
    "accepted hold": (600, 2, {"cross_fit": False, "step_size": 1.0}, 0.9),
}
ALL = {**FIXTURES, **TOLERANT, **EXTRA}


def estimator(name: str, **extra: Any) -> ContinuousSurvivalTMLE:
    """The fixture's estimator, with ``extra`` settings."""
    _, _, settings, _ = ALL[name]
    # A large first step and a tight cutoff, so every walk accepts two steps and halves.
    merged = {
        "n_folds": 3,
        "learner_folds": 3,
        "random_state": 20261118,
        "step_size": 0.3,
        "tol": 1e-3,
        **settings,
        **extra,
    }
    return ContinuousSurvivalTMLE((1.0, 2.0), **merged)


@contextmanager
def walk_floor(value: float | None) -> Iterator[None]:
    """Patch the walk's survival floor, so a fixture freezes pairs."""
    from cleverly.survival import _targeting

    original = _targeting.WALK_SURVIVAL_FLOOR
    if value is not None:
        _targeting.WALK_SURVIVAL_FLOOR = value  # type: ignore[misc]
    try:
        yield
    finally:
        _targeting.WALK_SURVIVAL_FLOOR = original  # type: ignore[misc]


def fit(name: str, **extra: Any) -> Any:
    """Fit the fixture, warnings silenced."""
    n, causes, _, floor = ALL[name]
    data = law(n, causes)
    with walk_floor(floor), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return estimator(name, **extra).fit_data(data)


def summary(result: Any) -> dict[str, Any]:
    """The numbers the golden file records of one fit."""
    record = result.record
    return {
        "estimates": {
            name: (float(e.psi), float(e.std_error)) for name, e in result.estimates.items()
        },
        "steps": int(record.steps),
        "halvings": int(record.halvings),
        "frozen_share": float(record.frozen_share),
        "off_grid_share": float(record.off_grid_share),
    }


def golden_records() -> None:
    """Print the golden module's dictionary from the installed engine."""
    import subprocess

    head = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    print(f"# generated at {head}")
    print("GOLDEN = {")
    for name in {**FIXTURES, **TOLERANT}:
        print(f"    {name!r}: {summary(fit(name))!r},")
    print("}")
