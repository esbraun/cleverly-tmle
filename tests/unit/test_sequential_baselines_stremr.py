"""Sequential g-computation and IPW against R ``stremr`` on one long panel.

``tests/canonical/stremr/run_stremr.R`` fits ``stremr`` at osofr/stremr ``e075f719`` on
``long.csv``: two decision times, a survival event after each, and no censoring.

| ``stremr`` estimator | cleverly | relation |
| --- | --- | --- |
| ``survNPMSM`` | ``SequentialIPW(normalization="hajek")`` | the same estimator: IPW hazards, then their product |
| ``fit_GCOMP`` with ``stratifyQ_by_rule=TRUE`` | ``SequentialGComputation`` | the same estimator |
| ``directIPW`` | none | a different estimator of the same risk. Its denominator at horizon 2 also counts the units that failed at horizon 1. It equals the Hajek risk at horizon 1 |

Without censoring the two packages factor the mechanism the same way, so the shared estimators
agree to the GLM tolerance.  The exact laws in ``test_sequential_baselines.py`` remain the
oracle for the target.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
import pytest

from cleverly.learners import GLM
from cleverly.longitudinal import LongitudinalData, SequentialGComputation, SequentialIPW

FIXTURE = Path(__file__).parents[1] / "canonical" / "stremr"
REGIMENS = {"always": 1, "never": 0}
SHARED = {"survNPMSM": "ipw", "GCOMP": "gcomp"}

# The local study queue writes ``reference.csv``.  Comparator runs do not gate a method, so
# these tests wait for it.
pytestmark = pytest.mark.skipif(
    not (FIXTURE / "reference.csv").exists(),
    reason="the stremr reference run has not been registered",
)


def _data() -> LongitudinalData:
    long = pd.read_csv(FIXTURE / "long.csv", float_precision="round_trip")
    return LongitudinalData.from_long_panel(
        long.drop(columns=["Lprev", "Aprev", "C"]),
        unit="ID",
        time="t",
        treatment="A",
        baseline=["W"],
        time_varying=["L"],
        event="Y",
    )


def _risks(estimator: str) -> dict[tuple[str, int], float]:
    learner = GLM(tol=1e-12)
    settings = {"g_bounds": (0.01, 1.0), "simultaneous": False}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if estimator == "gcomp":
            model = SequentialGComputation(
                REGIMENS, outcome_learner=learner, treatment_learner=learner, **settings
            )
        else:
            model = SequentialIPW(REGIMENS, treatment_learner=learner, **settings)
        result = model.fit(_data())
    return {
        (plan, horizon): result.psi(f"risk_regimen[{plan} @ t={horizon}]")
        for plan in REGIMENS
        for horizon in (1, 2)
    }


def _reference() -> pd.DataFrame:
    return pd.read_csv(FIXTURE / "reference.csv", float_precision="round_trip")


@pytest.mark.parametrize(("reference", "estimator"), list(SHARED.items()))
def test_each_shared_estimator_matches_stremr(reference: str, estimator: str) -> None:
    expected = _reference()
    for (plan, horizon), value in _risks(estimator).items():
        row = expected[
            (expected["plan"] == plan)
            & (expected["horizon"] == horizon)
            & (expected["estimator"] == reference)
        ]
        assert len(row) == 1, (plan, horizon, reference)
        assert value == pytest.approx(float(row["risk"].iloc[0]), abs=1e-6), (plan, horizon)


def test_direct_ipw_is_the_hajek_risk_at_the_first_horizon_only() -> None:
    expected = _reference()
    hajek = _risks("ipw")
    direct = expected[expected["estimator"] == "directIPW"].set_index(["plan", "horizon"])
    for plan in REGIMENS:
        assert float(direct.loc[(plan, 1), "risk"]) == pytest.approx(hajek[(plan, 1)], abs=1e-6)
        # The denominator at horizon 2 counts the units that failed at horizon 1, so the two
        # estimators of one risk differ in a finite sample.
        assert abs(float(direct.loc[(plan, 2), "risk"]) - hajek[(plan, 2)]) > 1e-4


def test_no_bound_binds() -> None:
    """``stremr`` truncates no weight here, so ``g_bounds`` must not bind in cleverly either."""
    expected = _reference()
    assert float(expected["max_weight"].max()) < 100.0
