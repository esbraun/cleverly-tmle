"""The reviewed semantic callback for ``docs/examples/plots``.

``tests/unit/test_documentation_runtime.py`` runs the notebook at its documented size and passes
the namespace to :func:`check`.  The slow tier also runs the simulated-confounding cell and calls
:func:`check_refits`.  Each assertion is a claim the prose makes about this draw, with a
tolerance wide enough to survive a library release, and not a statistical guarantee.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from tests.unit.tutorial_semantics import covers, figure_title

#: The refit cells the rule derives from this notebook today, in notebook order.  A change here is
#: a change in what the fast gate skips, so it shows in the diff of the pull request that causes it.
REFIT_CELLS = ("simulated-confounding",)

#: Decimals the prose writes that no output prints, each with how it follows from one.
UNPRINTED_DECIMALS: dict[str, str] = {}


def _check_estimate(namespace: dict[str, Any]) -> None:
    """The fit covers the true effect, and the weighting-only estimate has no interval."""
    truth = namespace["truth"]["ate"]
    assert truth == 1.0
    result, weighting = namespace["result"], namespace["weighting"]
    assert covers(result["ate"], truth)
    assert 0.9 < result["ate"].psi < 1.3
    assert result.inference_status == "influence_curve"
    # The second row of the forest plot has a point and no interval, because the fit holds none.
    assert weighting.inference_status == "estimated_mechanism_plugin"
    estimates = namespace["estimates"].set_index("method")
    assert estimates.loc["weighting only", ["ci_lower", "ci_upper"]].isna().all()
    assert estimates.loc["TMLE", ["ci_lower", "ci_upper"]].notna().all()
    assert weighting["ate"].support.label == "experimental"
    assert figure_title(namespace["forest_fig"]) == "Estimates and intervals"


def _check_overlap_and_balance(namespace: dict[str, Any]) -> None:
    """Offered patients have high chances, the capped share is small, and weighting balances."""
    chances = namespace["chances"]
    assert chances.loc["5%", "offered"] > 0.1, "few offered patients have a low chance"
    assert chances.loc["50%", "not offered"] < 0.4 < 0.6 < chances.loc["50%", "offered"]
    support = namespace["support"]
    assert 0.02 < support.truncated["fraction"] < 0.05
    low, high = support.bounds
    assert abs(low - 0.018) < 0.002 and abs(high - 0.982) < 0.002
    assert figure_title(namespace["overlap_fig"]) == "Predicted probability of treatment, by group"
    differences = namespace["differences"].set_index("covariate")
    before, after = differences["smd_unweighted"], differences["smd_weighted"]
    assert after["discharge_risk"] < 0.1 < before["discharge_risk"], "the main difference is gone"
    assert after["discharge_risk"] < 0.1 < after["prior_utilization"] < before["prior_utilization"]
    assert before["medication_burden"] < 0.1 and after["medication_burden"] < 0.1
    assert figure_title(namespace["balance_fig"]) == "Covariate balance before and after weighting"


def _check_truncation(namespace: dict[str, Any]) -> None:
    """The estimate drifts by about a tenth across the caps and stays well above zero."""
    frame = namespace["truncation"].to_frame()
    assert frame["bound"].is_monotonic_increasing
    assert frame["psi"].is_monotonic_decreasing, "a tighter cap lowers this estimate"
    assert 0.05 < frame["psi"].iloc[0] - frame["psi"].iloc[-1] < 0.2
    assert (frame["ci_lower"] > 0.5).all()
    fitted = frame[frame["is_fitted_bound"]]
    assert len(fitted) == 1
    assert abs(fitted["psi"].iloc[0] - namespace["result"]["ate"].psi) < 1e-6
    assert (
        abs(fitted["truncated_fraction"].iloc[0] - namespace["support"].truncated["fraction"])
        < 0.002
    )
    assert frame["truncated_fraction"].iloc[-1] > 0.3
    title = figure_title(namespace["truncation_fig"])
    assert title == "Estimate across probability limits"


def _check_policy(namespace: dict[str, Any]) -> None:
    """The effect falls below zero at half the odds and rises with the multiplier."""
    table = namespace["policy_table"].set_index("estimand")
    half, double, four = (table["psi"].iloc[i] for i in range(3))
    assert -0.2 < half < -0.05 < 0.05 < double < 0.2
    assert double < four < 0.3
    assert (table["ci_lower"] * table["ci_upper"] > 0).all(), "no interval crosses zero"
    assert figure_title(namespace["policy_fig"]) == "Effect by policy"


def _check_evalue(namespace: dict[str, Any]) -> None:
    """The drawn curve passes through the E-value on the diagonal, and a wrong curve would not."""
    evalue, partner = namespace["evalue"], namespace["partner_strength"]
    ratio, limit_ratio = evalue.risk_ratio, evalue.risk_ratio_ci[0]
    assert 1.4 < ratio < 1.7
    assert 1.0 < limit_ratio < ratio
    # E-value = RR + sqrt(RR (RR - 1)), where the two links are equal.
    assert abs(evalue.point - (ratio + np.sqrt(ratio * (ratio - 1.0)))) < 1e-9
    assert abs(evalue.limit - (limit_ratio + np.sqrt(limit_ratio * (limit_ratio - 1.0)))) < 1e-9
    for ratio_i, value in ((ratio, evalue.point), (limit_ratio, evalue.limit)):
        assert abs(partner(ratio_i, np.array([value]))[0] - value) < 1e-9
        # A pair of links below the curve gives a bias factor under the ratio, so it does not
        # explain the estimate away. The factor is x * y / (x + y - 1).
        x, y = value + 0.5, partner(ratio_i, np.array([value + 0.5]))[0] - 0.2
        assert x * y / (x + y - 1.0) < ratio_i
        assert x * (y + 0.4) / (x + y + 0.4 - 1.0) > ratio_i
    assert evalue.limit < evalue.point
    assert namespace["evalue_fig"].axes[0].get_title(loc="left") == (
        "How strong would a missing confounder have to be?"
    )


def _check_gamma(namespace: dict[str, Any]) -> None:
    """The range is one point at gamma 1, widens with gamma, and stays above zero up to 3."""
    frame = namespace["gamma_bounds"].to_frame()
    first, last = frame.iloc[0], frame.iloc[-1]
    assert first["gamma"] == 1.0 and abs(first["upper"] - first["lower"]) < 1e-9
    width = frame["upper"] - frame["lower"]
    assert width.is_monotonic_increasing
    assert last["gamma"] == 3.0 and 0.0 < last["lower_limit"] < last["lower"] < 0.5
    assert last["upper"] > 1.5
    assert figure_title(namespace["gamma_fig"]) == "Bounds under hidden confounding"


def _check_survey(namespace: dict[str, Any]) -> None:
    """About a fifth of the scores are missing, and the tilt lowers the estimate to zero."""
    survey = namespace["survey"]
    share = 1.0 - float(survey["responded"].mean())
    assert 0.2 < share < 0.25
    result = namespace["survey_result"]
    assert covers(result["ate"], namespace["survey_truth"]["ate"])
    frame = namespace["tilt"].to_frame()
    assert frame["psi"].is_monotonic_decreasing
    assert abs(frame["psi"].iloc[0] - result["ate"].psi) < 1e-9, "gamma 0 is the main analysis"
    # The grid reaches 3.0, so the estimate crosses zero at the tipping value between 2.0 and 3.0.
    assert frame["gamma"].iloc[-1] == 3.0 and frame["psi"].iloc[-1] < 0.0
    tipping = namespace["tipping"]
    assert 2.0 < tipping < 3.0
    assert (frame["psi"][frame["gamma"] < tipping] > 0.0).all()
    assert (frame["psi"][frame["gamma"] > tipping] < 0.0).all()
    assert figure_title(namespace["tilt_fig"]) == "Estimate across missingness assumptions"


def _check_zipper(namespace: dict[str, Any]) -> None:
    """Fewer than six in ten intervals cover the truth, and every miss is below it."""
    study = namespace["coverage_study"]
    records = namespace["records"]
    truth = namespace["truth_ate"]
    assert len(records) == 100
    inside = (records["low"] <= truth) & (truth <= records["high"])
    assert (inside == records["covered"]).all(), "each recorded interval agrees with `covered`"
    assert 0.3 < inside.mean() < 0.6
    assert (records["low"] > truth).sum() == 0
    assert (records["high"] < truth).sum() == (~inside).sum() > 30
    assert abs(study.summaries["ate"].truth - truth) < 1e-12
    assert figure_title(namespace["zipper_fig"]) == "Interval coverage"


def check(namespace: dict[str, Any]) -> None:
    """The plots tutorial's claims hold at the displayed sample size."""
    _check_estimate(namespace)
    _check_overlap_and_balance(namespace)
    _check_truncation(namespace)
    _check_policy(namespace)
    _check_evalue(namespace)
    _check_gamma(namespace)
    _check_survey(namespace)
    _check_zipper(namespace)


def check_refits(namespace: dict[str, Any]) -> None:
    """The invented confounder shrinks the estimate without reversing it."""
    frame = namespace["cells"].set_index(["treatment_strength", "outcome_strength"])
    estimate = frame["estimate"]
    assert abs(estimate[(0.0, 0.0)] - namespace["result"]["ate"].psi) < 1e-6
    assert (estimate > 0.0).all(), "no cell crosses zero"
    flips = estimate.xs(0.0, level="outcome_strength")
    assert flips.is_monotonic_decreasing and flips.iloc[-1] < 0.5 * flips.iloc[0]
    calibration = namespace["surface"].calibration_frame().set_index("role")
    assert 0.2 < calibration.loc["treatment", "strength"] < 0.4
    title = figure_title(namespace["surface_fig"])
    assert title == "Estimate under a simulated common cause"
