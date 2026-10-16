"""Refusals of a repeated outcome, each pinned to its message.

Each declaration refusal raises in ``LongitudinalData.from_frame``, before any learner.
Each result refusal names what a repeated fit reports instead: a mean at each horizon,
not a cumulative risk.  WR runs the result refusals on every estimator kind that fits a
repeated outcome.
"""

from __future__ import annotations

import re
import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.exceptions import DataError
from cleverly.longitudinal import IPSI, LTMLE, LongitudinalData, Repeated

from .test_repeated_outcome import panel

ROLES: dict[str, Any] = {
    "treatment": ["A1", "A2", "A3"],
    "baseline": ["W"],
    "time_varying": [["L1"], ["L2"], ["L3"]],
    "censoring": ["C1", "C2", "C3"],
}
OUTCOME = Repeated(["Y1", "Y2", "Y3"])


KINDS = (LTMLE, IPSI)


def _estimator(kind: type[Any] = LTMLE, **overrides: Any) -> Any:
    settings: dict[str, Any] = {
        "reference": "never",
        "outcome_learner": LogisticRegression(),
        "pseudo_learner": LinearRegression(),
        "treatment_learner": LogisticRegression(),
        "censoring_learner": LogisticRegression(),
        "n_folds": 1,
        "simultaneous": False,
    }
    settings.update(overrides)
    return kind({"never": 0, "always": 1}, **settings)


@pytest.fixture(scope="module", params=KINDS, ids=[kind.__name__ for kind in KINDS])
def result(request: pytest.FixtureRequest) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _estimator(request.param).fit(panel(), outcome=OUTCOME, **ROLES)


@pytest.fixture(scope="module", params=KINDS, ids=[kind.__name__ for kind in KINDS])
def continuous_result(request: pytest.FixtureRequest) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _estimator(
            request.param, outcome_learner=LinearRegression(), q_bounds=(-1.0, 6.0)
        ).fit(panel(continuous=True), outcome=OUTCOME, **ROLES)


# ---------------------------------------------------------------- the declaration


def test_one_column_name_is_not_a_repeated_outcome() -> None:
    with pytest.raises(DataError, match="one column name is an end-of-study outcome"):
        Repeated("Y1")  # type: ignore[arg-type]


def test_an_empty_declaration_is_refused() -> None:
    with pytest.raises(DataError, match="Repeated\\(\\) is empty"):
        Repeated([])


def test_a_repeated_column_is_refused() -> None:
    with pytest.raises(DataError, match="Repeated\\(\\) repeats a column"):
        Repeated(["Y1", "Y1", "Y3"])


def test_the_column_count_must_be_the_node_count() -> None:
    with pytest.raises(DataError, match="names 2 column\\(s\\) but there are 3 treatment node"):
        LongitudinalData.from_frame(panel(), outcome=Repeated(["Y1", "Y2"]), **ROLES)


def test_the_outcome_is_not_also_a_declared_covariate() -> None:
    roles = {**ROLES, "time_varying": [["L1"], ["Y1", "L2"], ["L3"]]}
    with pytest.raises(DataError, match="time_varying= names the outcome column\\(s\\) \\['Y1'\\]"):
        LongitudinalData.from_frame(panel(), outcome=OUTCOME, **roles)


def test_a_value_after_censoring_is_refused() -> None:
    """``Y_t`` on a row censored at ``t`` is data recorded after censoring."""
    frame = panel()
    censored = np.flatnonzero((frame["C2"] == 0).to_numpy())
    frame.loc[censored[0], "Y2"] = 1.0
    with pytest.raises(DataError, match="Y2"):
        LongitudinalData.from_frame(frame, outcome=OUTCOME, **ROLES)


def test_a_missing_value_on_an_uncensored_row_names_censoring() -> None:
    """``Y_t`` missing on a row still observed at ``t`` is a censoring node, not a gap."""
    frame = panel()
    observed = np.flatnonzero((frame["C3"] == 1).to_numpy())
    frame.loc[observed[0], "Y3"] = np.nan
    with pytest.raises(DataError, match="encode it as censoring at that time point"):
        LongitudinalData.from_frame(frame, outcome=OUTCOME, **ROLES)


def test_a_missing_earlier_value_is_refused_too() -> None:
    """``Y_1`` is also the first covariate of node 2, yet gets the outcome's censoring wording."""
    frame = panel()
    observed = np.flatnonzero((frame["C1"] == 1).to_numpy())
    frame.loc[observed[0], "Y1"] = np.nan
    with pytest.raises(DataError, match=r"no Y1\. .*encode it as censoring"):
        LongitudinalData.from_frame(frame, outcome=OUTCOME, **ROLES)


@pytest.mark.parametrize("kind", KINDS, ids=lambda kind: kind.__name__)
def test_horizons_on_an_end_of_study_fit_names_both_curve_declarations(kind: type[Any]) -> None:
    with pytest.raises(ValueError, match=re.escape("outcome=Repeated([...])")):
        _estimator(kind, horizons=[1]).fit(panel(), outcome="Y3", **ROLES)


# ---------------------------------------------------------------- the result


@pytest.mark.parametrize(
    "call",
    [
        lambda r: r.rmst("always", 3),
        lambda r: r.rmtl("always", 3),
        lambda r: r.ratio("ey_regimen[always @ t=3]", "ey_regimen[never @ t=3]", view="survival"),
        lambda r: r.curve(scale="risk"),
        lambda r: r.curve(scale="survival"),
    ],
    ids=["rmst", "rmtl", "ratio_survival_view", "curve_risk", "curve_survival"],
)
def test_a_survival_functional_is_refused(result: Any, call: Any) -> None:
    with pytest.raises(ValueError, match="the mean of a repeated outcome at each horizon"):
        call(result)


def test_incidence_total_is_refused(result: Any) -> None:
    with pytest.raises(
        ValueError, match="the mean of a repeated outcome at each horizon"
    ) as caught:
        result.incidence_total()
    assert "competing risks" not in str(caught.value)


def test_the_mean_view_of_an_end_of_study_fit_names_the_missing_curve() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = _estimator().fit(panel(), outcome="Y3", **ROLES)
    with pytest.raises(ValueError, match="one end-of-study outcome") as caught:
        fit.curve(scale="mean")
    assert "this fit reports a cumulative risk" not in str(caught.value)


def test_a_repeated_container_has_no_event_time(result: Any) -> None:
    with pytest.raises(DataError, match="repeated outcome has no event time") as caught:
        result.data.to_time_to_event()
    assert "end-of-study" not in str(caught.value)


def test_a_rule_description_names_the_earlier_outcome() -> None:
    from .test_repeated_outcome import RULE, regimens

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = LTMLE(
            regimens({"never": (0, 0, 0), **RULE}, 3, held=False),
            reference="never",
            outcome_learner=LogisticRegression(),
            pseudo_learner=LinearRegression(),
            treatment_learner=LogisticRegression(),
            censoring_learner=LogisticRegression(),
            n_folds=1,
            simultaneous=False,
        ).fit(panel(), outcome=OUTCOME, **ROLES)
    lines = fit.config.describe(contrast=False)
    assert any("Y_{t-1}" in line for line in lines)


def test_the_curve_is_a_curve_of_means(result: Any) -> None:
    """W16: ``curve()`` reports ``ey_regimen`` rows, one per plan and horizon."""
    frame = result.curve()
    assert isinstance(frame, pd.DataFrame)
    means = [name for name in frame["estimand"] if name.startswith("ey_regimen[")]
    assert means == [f"ey_regimen[{d} @ t={h}]" for d in ("never", "always") for h in (1, 2, 3)]
    assert set(frame["view"]) == {"mean"}
    assert not any(name.startswith("risk_regimen[") for name in frame["estimand"])


def test_a_survival_fit_refuses_the_mean_view() -> None:
    from .. import discrete_law_survival as survival

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = _estimator().fit(
            survival.frame(),
            outcome=["Y1", "Y2"],
            treatment=["A1", "A2"],
            baseline=["W"],
            time_varying=[[], ["L2"]],
            censoring=["C1", "C2"],
        )
    with pytest.raises(ValueError, match="scale='mean' is the view of a repeated outcome"):
        fit.curve(scale="mean")


def test_an_odds_ratio_on_a_continuous_repeated_outcome_is_refused(
    continuous_result: Any,
) -> None:
    with pytest.raises(ValueError, match="odds"):
        continuous_result.ratio("ey_regimen[always @ t=3]", "ey_regimen[never @ t=3]", kind="or")


def test_a_ratio_of_means_is_reported(result: Any) -> None:
    ratio = result.ratio("ey_regimen[always @ t=2]", "ey_regimen[never @ t=2]")
    assert ratio.name == "rr_regimen[always vs never @ t=2]"
    assert ratio.psi == pytest.approx(
        result.psi("ey_regimen[always @ t=2]") / result.psi("ey_regimen[never @ t=2]")
    )


def test_the_settings_report_names_the_outcome_kind(result: Any) -> None:
    assert result.config.outcome_kind == "repeated"
    assert "outcome: repeated" in result.summary()
