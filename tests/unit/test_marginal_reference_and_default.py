"""Fix-15: the marginal sensitivity bounds read the fit's reference and its estimand.

Three defects, each with a witness and a mutation control:

* the set interval took a normal quantile beside a fit whose clustered interval reads a
  Student t reference with ``J - 2`` degrees of freedom;
* a call without ``estimand`` bounded the ATE of an ATT, ATC or ratio fit, a parameter the
  fit did not report;
* the ``truncation_curve`` row of a ``SurvivalDRTMLE`` or ``SurvivalCTMLE`` fit read
  available while the call refused.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest
from scipy import stats
from sklearn.linear_model import LogisticRegression

from cleverly import ATC, ATT, CausalStudy, OddsRatio, PointTreatment, RiskRatio
from cleverly.assessment import AssessmentStatus
from cleverly.datasets import make_binary_outcome, make_clustered
from cleverly.estimators import TMLE
from cleverly.exceptions import CapabilityError
from cleverly.sensitivity import marginal, marginal_sensitivity_bounds
from tests.conftest import linear_in_sample


def _clustered(n: int, cluster_size: int) -> Any:
    frame, _ = make_clustered(n=n, cluster_size=cluster_size, seed=7)
    return TMLE(**linear_in_sample()).fit(frame, outcome="Y", treatment="A", id="cluster").single()


# ------------------------------------------------------------------ the reference distribution


def test_the_limits_read_the_fit_s_few_cluster_t_reference() -> None:
    fit = _clustered(360, 30)  # 12 clusters: t with 10 df
    assert fit.estimates["ate"].reference_df == 10
    bounds = marginal_sensitivity_bounds(fit, 1.0, estimand="ate")
    row = bounds.rows[0]
    assert bounds.reference_df == 10
    t = stats.t.ppf(0.975, 10)
    np.testing.assert_allclose(row.lower - row.lower_limit, t * row.lower_std_error, rtol=1e-10)
    np.testing.assert_allclose(row.upper_limit - row.upper, t * row.upper_std_error, rtol=1e-10)
    # At gamma = 1 the set interval is as wide as the fit's own t interval on the same
    # standard error scale; the normal quantile made it narrower.
    low, high = fit.estimates["ate"].ci
    fit_half = (high - low) / 2.0
    bound_half = (row.upper_limit - row.lower_limit) / 2.0
    assert bound_half / fit_half == pytest.approx(1.0, abs=0.05)
    assert bound_half / fit_half > stats.norm.ppf(0.975) / t + 0.05


def test_a_normal_quantile_is_the_mutation_the_witness_sees(monkeypatch: Any) -> None:
    fit = _clustered(360, 30)
    monkeypatch.setattr(marginal, "_reference_df", lambda result, target: None)
    row = marginal_sensitivity_bounds(fit, 1.0, estimand="ate").rows[0]
    t = stats.t.ppf(0.975, 10)
    assert not np.isclose(row.lower - row.lower_limit, t * row.lower_std_error, rtol=1e-6)


def test_the_normal_reference_holds_at_the_threshold_and_without_clusters() -> None:
    fit = _clustered(400, 10)  # 40 clusters: the normal reference
    assert fit.estimates["ate"].reference_df is None
    bounds = marginal_sensitivity_bounds(fit, 1.0, estimand="ate")
    row = bounds.rows[0]
    assert bounds.reference_df is None
    z = stats.norm.ppf(0.975)
    np.testing.assert_allclose(row.upper_limit - row.upper, z * row.upper_std_error, rtol=1e-10)


def test_the_summary_names_a_t_reference() -> None:
    bounds = marginal_sensitivity_bounds(_clustered(360, 30), 1.0, estimand="ate")
    assert "t, 10 df" in str(bounds.summary(detail="full"))


# ------------------------------------------------------------------------ the default estimand


def _binary_study() -> CausalStudy:
    frame, _ = make_binary_outcome(n=600, seed=2)
    covariates = [name for name in frame.columns if name.startswith("W")]
    return CausalStudy(
        frame, design=PointTreatment(outcome="Y", treatment="A", adjustment=covariates)
    )


def _fit(estimand: Any) -> Any:
    return (
        _binary_study()
        .identify(estimand)
        .estimate(
            outcome_learner=LogisticRegression(max_iter=1000),
            treatment_learner=LogisticRegression(max_iter=1000),
            cross_fit=False,
            random_state=0,
        )
    )


@pytest.mark.parametrize(
    ("estimand", "name"), [(ATT(), "att"), (ATC(), "atc"), (RiskRatio(), "rr"), (OddsRatio(), "or")]
)
def test_a_bare_call_bounds_the_parameter_the_fit_reported(estimand: Any, name: str) -> None:
    fit = _fit(estimand)
    assert list(fit.estimates) == [name]
    assert marginal_sensitivity_bounds(fit, 1.0).estimand == name
    assert fit.sensitivity.marginal_bounds(gamma=1.0).estimand == name


def test_the_att_default_is_not_the_ate() -> None:
    """Nonzero witness: on this law the two bounds differ at gamma = 1."""
    fit = _fit(ATT())
    att = marginal_sensitivity_bounds(fit, 1.0).rows[0]
    ate = marginal_sensitivity_bounds(fit, 1.0, estimand="ate").rows[0]
    assert abs(att.lower - ate.lower) > 0.005
    np.testing.assert_allclose(att.lower, fit.estimates["att"].psi, atol=0.02)


def test_the_combined_report_bounds_the_reported_parameter() -> None:
    fit = _fit(ATT())
    report = fit.assess(include_retargets=True, include_refits=True, random_state=0)
    item = next(item for item in report.sensitivity.items if item.name == "marginal_bounds")
    assert "att" in (item.detail or "") or "att" in str(item.report)


# ------------------------------------------------------------- survival truncation rows


def _survival(method: Any) -> Any:
    from cleverly import RegimeContrast, TimeToEvent
    from tests.unit.test_survival_drtmle import sample_frame

    study = CausalStudy(
        sample_frame(600, 21),
        design=TimeToEvent(time="time", event="event", treatment="A", baseline=["W1", "W2"]),
    )
    effect = study.identify(RegimeContrast(regimens={"a1": 1, "a0": 0}, reference="a0"))
    learner = LogisticRegression(max_iter=1000)
    settings: dict[str, Any] = {"outcome_learner": learner, "cross_fit": False, "random_state": 0}
    settings["method"] = method
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return effect.estimate(**settings)


def test_the_survival_drtmle_and_ctmle_truncation_rows_carry_their_call_s_refusal() -> None:
    from cleverly.longitudinal.estimator import longitudinal_truncation_curve
    from cleverly.methods import CollaborativeTMLEMethod, DRTMLEMethod

    for method in (DRTMLEMethod(), CollaborativeTMLEMethod(selection_folds=3)):
        fit = _survival(method)
        row = fit.diagnostics.capability("truncation_curve")
        assert not row.available
        assert row.status is AssessmentStatus.UNAVAILABLE
        assert row.refusal.roadmap == "X62"
        with pytest.raises(CapabilityError) as raised:
            longitudinal_truncation_curve(fit, [0.05])
        assert raised.value.refusal == row.refusal


def test_a_hazard_ltmle_truncation_row_stays_available() -> None:
    """Witness: the rule refuses the two methods, not every survival fit."""
    from tests.unit._capability_sweep_support import KINDS

    fit = KINDS["ltmle_hazard"].build()
    assert fit.diagnostics.capability("truncation_curve").available
