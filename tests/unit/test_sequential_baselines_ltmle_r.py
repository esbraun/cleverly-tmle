"""W9: sequential g-computation and IPW against R ``ltmle`` 1.3-0 on one nonsaturated panel.

``tests/canonical/sequential_baselines/generate_reference.R`` fits two ``ltmle`` calls per
regime on ``data.csv``: ``gcomp=TRUE`` with ``stratify=TRUE`` for the plug-in, and
``stratify=FALSE`` for the IPTW point, which ``CalcIPTW`` normalizes by the weight sum.  Every
regression reads the full history with ``SL.library="glm"``, and ``gbounds=c(0.01, 1)``.  The
exact laws in ``test_sequential_baselines.py`` remain the oracle for the target.  This fixture
checks that the GLM fits and the weights match the reference implementation.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
import pytest

from cleverly.learners import GLM
from cleverly.longitudinal import DynamicRegimen, SequentialGComputation, SequentialIPW

FIXTURE = Path(__file__).parents[1] / "canonical" / "sequential_baselines"
COLUMNS = {
    "outcome": "Y",
    "treatment": ["A1", "A2"],
    "baseline": ["W1", "W2"],
    "time_varying": [[], ["L2"]],
    "censoring": ["C1", "C2"],
}
REGIMENS = {
    "always": 1,
    "never": 0,
    "rule": DynamicRegimen(
        "rule", (lambda h: h["W1"] > 0, lambda h: h["L2"] > 0), rule_kind="known"
    ),
}


def _reference() -> pd.DataFrame:
    return pd.read_csv(FIXTURE / "reference.csv", float_precision="round_trip")


def _fit(estimator: str) -> dict[str, float]:
    frame = pd.read_csv(FIXTURE / "data.csv", float_precision="round_trip")
    learner = GLM(tol=1e-12)
    settings = {"g_bounds": (0.01, 1.0), "simultaneous": False}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if estimator == "gcomp":
            model = SequentialGComputation(
                REGIMENS,
                outcome_learner=learner,
                treatment_learner=learner,
                censoring_learner=learner,
                **settings,
            )
        else:
            model = SequentialIPW(
                REGIMENS,
                treatment_learner=learner,
                censoring_learner=learner,
                normalization="hajek",
                **settings,
            )
        result = model.fit(frame, **COLUMNS)
    return {name: result.psi(f"ey_regimen[{name}]") for name in REGIMENS}


@pytest.mark.parametrize(("estimator", "reference"), [("gcomp", "gcomp"), ("ipw", "iptw")])
def test_each_point_matches_r_ltmle(estimator: str, reference: str) -> None:
    expected = _reference()
    points = _fit(estimator)
    for name, value in points.items():
        row = expected[(expected["regime"] == name) & (expected["estimator"] == reference)]
        assert len(row) == 1, (name, reference)
        assert value == pytest.approx(float(row["estimate"].iloc[0]), abs=1e-7), name
