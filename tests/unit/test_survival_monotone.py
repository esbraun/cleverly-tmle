"""A one-step hazard curve is monotone where the per-horizon curves are not (X14 W1).

The witness sample is ``make_point_survival(n=300, seed=3, hazard_scale={3: 0.1})`` with
the default logistic learners of scikit-learn, which draw no random number.  On it the
``"iterated_mean"`` curve of arm 1 decreases by about 1e-3 between horizons 2 and 3.  The
per-horizon hazard curve also decreases there, by a rounding error: each horizon reads its
own targeted hazards, so no reduction order ties the two levels.  The one-step curve reads
one set of hazards for every horizon, and ``np.diff`` of each of its curves is non-negative
with no tolerance (``cleverly.longitudinal.hazard`` module docstring).

A two-cause sample checks each cause-specific incidence curve and their sum.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.datasets import make_point_survival
from cleverly.longitudinal import LTMLE
from cleverly.longitudinal.hazard import project_curves

COLUMNS = {
    "outcome": [f"Y{k}" for k in range(1, 6)],
    "treatment": "A",
    "baseline": ["W1", "W2"],
    "censoring": [f"C{k}" for k in range(1, 6)],
}
REGIMENS = {"a1": 1, "a0": 0}


def _logistic() -> LogisticRegression:
    return LogisticRegression(max_iter=1000)


def _fit(frame: Any, columns: dict[str, Any], **settings: Any) -> Any:
    learners: dict[str, Any] = {
        "outcome_learner": _logistic(),
        "treatment_learner": _logistic(),
        "censoring_learner": _logistic(),
    }
    if settings.get("survival_model", "iterated_mean") == "iterated_mean":
        learners["pseudo_learner"] = LinearRegression()
    with warnings.catch_warnings():
        # The witness has a node without an event, so a status warning is expected.
        warnings.simplefilter("ignore")
        return LTMLE(REGIMENS, n_folds=1, **learners, **settings).fit(frame, **columns)


@pytest.fixture(scope="module")
def frame() -> Any:
    sample, _ = make_point_survival(n=300, seed=3, layout="wide", hazard_scale={3: 0.1})
    return sample


def _curve(result: Any, label: str, cause: str | None = None, horizons: int = 5) -> np.ndarray:
    if cause is None:
        names = [f"risk_regimen[{label} @ t={t}]" for t in range(1, horizons + 1)]
    else:
        names = [f"cif_regimen[{label}, {cause} @ t={t}]" for t in range(1, horizons + 1)]
    return np.array([result[name].psi for name in names])


def test_w1_the_iterated_mean_curve_decreases(frame: Any) -> None:
    result = _fit(frame, COLUMNS)
    assert np.diff(_curve(result, "a1")).min() < -1e-6


def test_w1_the_per_horizon_curve_decreases_in_floating_point(frame: Any) -> None:
    result = _fit(frame, COLUMNS, survival_model="hazard")
    assert np.diff(_curve(result, "a1")).min() < 0.0


def test_w1_the_one_step_curve_never_decreases(frame: Any) -> None:
    result = _fit(frame, COLUMNS, survival_model="hazard", targeting="one_step")
    assert next(iter(result.fits.values())).hazard.converged
    for label in REGIMENS:
        assert np.all(np.diff(_curve(result, label)) >= 0.0)


def test_w1_the_isotonic_iterated_mean_curve_is_monotone_and_pava(frame: Any) -> None:
    plain = _fit(frame, COLUMNS)
    projected = _fit(frame, COLUMNS, isotonic=True)
    for label in REGIMENS:
        after = _curve(projected, label)
        assert np.all(np.diff(after) >= 0.0)
        np.testing.assert_array_equal(after, project_curves(_curve(plain, label)[None, :])[0])


def test_w1_each_cause_and_their_sum_never_decrease() -> None:
    sample, _ = make_point_survival(n=300, seed=3, layout="wide", causes=2, n_times=4)
    nodes = range(1, 5)
    columns = {
        "outcome": {"relapse": [f"R{k}" for k in nodes], "death": [f"D{k}" for k in nodes]},
        "treatment": "A",
        "baseline": ["W1", "W2"],
        "censoring": [f"C{k}" for k in nodes],
    }
    result = _fit(sample, columns, survival_model="hazard", targeting="one_step")
    for label in REGIMENS:
        relapse = _curve(result, label, "relapse", horizons=4)
        death = _curve(result, label, "death", horizons=4)
        assert np.all(np.diff(relapse) >= 0.0)
        assert np.all(np.diff(death) >= 0.0)
        total = relapse + death
        assert np.all(np.diff(total) >= 0.0)
        assert np.all(total <= 1.0)
