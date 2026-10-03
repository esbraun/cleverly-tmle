"""The post-fit surfaces of the stratified incremental, MSM, natural-course and DR-TMLE fits.

Each composition fits baseline strata, and each surface that reads a fit must read the stratum
blocks too: ``retarget`` reproduces the fit, the truncation curve and ``score_check`` run, a
round trip keeps the nested stratum record, a bootstrap replicate refits every stratum, a
cross-fitted pooled fit and a clustered fit report every stratum.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.estimators import DRTMLE, TMLE
from cleverly.estimators.serialize import dumps, loads
from cleverly.interventions import Incremental
from cleverly.msm import MSM
from cleverly.validation import score_check
from tests.unit._stratified_alternating_support import GRID, frame


def logistic() -> LogisticRegression:
    return LogisticRegression(max_iter=1000)


#: Each composition: its estimator settings and its fit roles.  The roles name binary
#: outcomes, so a cross-fitted fit needs no declared outcome bounds.
COMPOSITIONS: dict[str, tuple[Any, dict[str, Any]]] = {
    "incremental": (
        lambda **kw: TMLE(incremental=[Incremental(2.0), Incremental(0.5)], **kw),
        {},
    ),
    "logit MSM": (lambda **kw: TMLE(msm=MSM.linear(modifiers=("W",), link="logit"), **kw), {}),
    "continuous logit MSM": (
        lambda **kw: TMLE(msm=MSM.linear(doses=GRID, link="logit"), density_bins=6, **kw),
        {"outcome": "Yd", "treatment": "D", "treatment_kind": "continuous"},
    ),
    "DR-TMLE": (
        lambda **kw: DRTMLE(
            estimands=("ey", "ate"),
            reduced_outcome_learner=LinearRegression(),
            reduced_treatment_learner=logistic(),
            **kw,
        ),
        {},
    ),
}


def fit(name: str, rows: pd.DataFrame, **settings: Any) -> Any:
    build, roles = COMPOSITIONS[name]
    settings = {
        "outcome_learner": logistic(),
        "treatment_learner": logistic(),
        "cross_fit": False,
        "simultaneous": False,
        "random_state": 0,
        **settings,
    }
    columns = {"outcome": "Y", "treatment": "A", "covariates": ["W", "V"], **roles}
    cluster = settings.pop("cluster", None)
    if cluster is not None:
        columns["id"] = cluster
    return build(**settings).fit(rows, strata=["V"], **columns).single()


def stratum_names(result: Any) -> list[str]:
    return [name for name in result.estimates if "[V=" in name]


@pytest.fixture(scope="module")
def rows() -> pd.DataFrame:
    return frame(600).assign(cluster=lambda f: np.arange(len(f)) // 3)


@pytest.fixture(scope="module", params=list(COMPOSITIONS))
def fitted(request: pytest.FixtureRequest, rows: pd.DataFrame) -> tuple[str, Any]:
    return request.param, fit(request.param, rows)


def test_every_stratum_is_reported(fitted: tuple[str, Any]) -> None:
    _, result = fitted
    assert len(stratum_names(result)) >= 3


def test_retarget_reproduces_the_fit(fitted: tuple[str, Any]) -> None:
    _, result = fitted
    estimates, _ = result.estimator.retarget(
        result.data, result.nuisance, estimands=result.config.estimands
    )
    for name in stratum_names(result):
        assert estimates[name].psi == pytest.approx(result[name].psi, abs=1e-10)


def test_the_truncation_curve_and_the_score_check_run(fitted: tuple[str, Any]) -> None:
    name, result = fitted
    if name != "incremental":
        # An incremental fit refuses an explicit propensity bound by design.
        curve = result.diagnostics.truncation_curve([0.02])
        assert curve is not None
    check = score_check(result)
    assert all(row.passed for row in check.rows if row.kind == "fluctuation")


def test_a_round_trip_keeps_every_stratum(fitted: tuple[str, Any]) -> None:
    _, result = fitted
    back = loads(dumps(result))
    for name in stratum_names(result):
        assert back[name].psi == result[name].psi
    nested = result.fluctuations[next(iter(result.fluctuations))].stratified
    restored = back.fluctuations[next(iter(back.fluctuations))].stratified
    assert (nested is None) == (restored is None)
    if nested is not None:
        np.testing.assert_array_equal(nested.epsilon, restored.epsilon)


@pytest.mark.parametrize("name", list(COMPOSITIONS))
def test_a_bootstrap_refits_every_stratum(name: str, rows: pd.DataFrame) -> None:
    result = fit(name, rows, n_bootstrap=2)
    assert result.bootstrap is not None
    assert set(stratum_names(result)) <= set(result.bootstrap.draws)


@pytest.mark.parametrize("name", list(COMPOSITIONS))
def test_a_cross_fitted_pooled_fit_reports_every_stratum(name: str, rows: pd.DataFrame) -> None:
    result = fit(name, rows, cross_fit=True, n_folds=2)
    assert len(stratum_names(result)) >= 3
    assert all(np.isfinite(result[n].std_error) for n in stratum_names(result))


@pytest.mark.parametrize("name", list(COMPOSITIONS))
def test_a_clustered_fit_reports_every_stratum(name: str, rows: pd.DataFrame) -> None:
    result = fit(name, rows, cluster="cluster")
    assert len(stratum_names(result)) >= 3
    assert all(np.isfinite(result[n].std_error) for n in stratum_names(result))


def test_the_in_sample_natural_course_reports_every_stratum(rows: pd.DataFrame) -> None:
    result = (
        TMLE(
            estimands=("ey_obs", "par"),
            outcome_learner=logistic(),
            treatment_learner=logistic(),
            cross_fit=False,
            simultaneous=False,
        )
        .fit(
            rows, outcome="Yobs", treatment="A", covariates=["W", "V"], delta="Delta", strata=["V"]
        )
        .single()
    )
    back = loads(dumps(result))
    for name in stratum_names(result):
        assert back[name].psi == result[name].psi
    assert all(row.passed for row in score_check(result).rows if row.kind == "fluctuation")
