"""The reviewed semantic callback for ``docs/examples/conditional-effects``.

``tests/unit/test_documentation_runtime.py`` runs the notebook at its documented size and passes
the namespace to :func:`check`.  The slow tier also runs the refit cells and calls
:func:`check_refits`.  Each assertion is a claim the prose makes about this draw, not a
statistical guarantee.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from tests.documents import ROOT
from tests.unit.tutorial_semantics import (
    EXAMPLES,
    assert_protocol_recorded,
    covers,
    figure_title,
)

NOTEBOOK = EXAMPLES / "conditional-effects.ipynb"
_PROBE = "reviews/notebook-review/probes/conditional-effects"

#: The refit cells the rule derives from this notebook today, in notebook order.  A change here is
#: a change in what the fast gate skips, so it shows in the diff of the pull request that causes it.
REFIT_CELLS = ("refute", "simulated-confounding", "stability")

#: Decimals the prose writes that no output prints, each with how it follows from one.
UNPRINTED_DECIMALS = {
    "0.945": f"pointwise coverage at age 20 of the projection's truth, 200 draws: {_PROBE}/summary.log",
    "0.955": f"band coverage of the projection's truth, 200 draws: {_PROBE}/summary.log",
}

#: Decimals a reading quotes from a code cell outside its own section, by reading cell, then
#: literal, then the code cell that prints it (tests.unit.tutorial_semantics.narration_scopes).
NARRATION_SOURCES: dict[str, dict[str, str]] = {}


def _summary() -> str:
    return (ROOT / _PROBE / "summary.log").read_text(encoding="utf-8")


def _check_confounding(namespace: dict[str, Any]) -> None:
    """The simple difference overstates the effect in every age band."""
    table = namespace["table"]
    gap = table["simple difference"] - table["true effect"]
    assert (gap > 0.05).all()
    assert table["true effect"].iloc[-1] < 0.02 < 0.05 < table["simple difference"].iloc[-1]


def _check_protocol(namespace: dict[str, Any]) -> None:
    """Every coupon fit carries the printed study description."""
    assert_protocol_recorded(
        NOTEBOOK,
        "protocol",
        namespace["protocol"],
        namespace["curve"],
        namespace["first_curve"],
        namespace["scored"],
        namespace["projection"],
        namespace["learned"],
    )
    trial_protocol = namespace["trial_protocol"]
    for result in (namespace["ratio_curve"], namespace["line"]):
        assert result.provenance.protocol_fingerprint == trial_protocol.fingerprint


def _check_curve(namespace: dict[str, Any]) -> None:
    """The supported EP curve runs high to 60, by up to 6 points at 20, and is near 0 from 70."""
    curve = namespace["curve"]
    ages = namespace["ages"]
    truth = namespace["true_uplift"](ages)
    estimate = curve.predict({"age": ages})
    young = ages <= 60
    assert np.all(estimate[young] - truth[young] > -0.01), "high, or about right, to 60"
    assert 0.055 < estimate[0] - truth[0] < 0.065, "up to 6 points high at 20"
    assert np.max(estimate[young] - truth[young]) == estimate[0] - truth[0]
    assert np.all(np.abs(estimate[ages >= 70]) < 0.02), "near zero from 70"
    assert estimate[-1] > estimate[-2], "the small rise after 70"
    assert curve.fitted_method == "ep"
    assert curve.status == "supported"
    assert curve.inference_status == "no_interval"
    assert curve.sieve.mode == "schedule"
    assert curve.sieve.chosen == 6 == namespace["waves_for"](8_000)
    # The declared schedule of the ep-cate study, max(1, 3 + floor(log2(n / 1000))).
    waves_for = namespace["waves_for"]
    assert [waves_for(n) for n in (500, 1_000, 1_999, 2_000, 4_000)] == [2, 3, 3, 4, 5]
    assert waves_for(100) == 1
    chances = curve.diagnostics.support().propensity_quantiles["overall"]
    assert chances[0.0] > 0.1 and chances[1.0] < 0.95, "the overlap reading"


def _check_assessment(namespace: dict[str, Any]) -> None:
    """The check reads near 1; the outcome models' own effect scores a little better."""
    calibration = namespace["calibration"]
    assert 0.85 < calibration.slope < 1.0
    assert abs(calibration.intercept) < 0.02
    bins = calibration.to_frame()
    gap = bins["pseudo_outcome_contrast"] - bins["mean_prediction"]
    assert np.max(np.abs(gap)) < 0.035, "the groups sit within 3 points of their stand-ins"
    assert figure_title(namespace["calibration_fig"])
    risks = namespace["risks"]
    assert risks["t_learner"] < risks["curve"] < risks["constant"] < 0
    assert risks["constant"] - risks["curve"] > 0.005, "clearly lower than a constant"
    assert risks["curve"] - risks["t_learner"] < 0.002, "a little lower still"
    assert abs(risks["curve"] - risks["dr_learner"]) < 0.001, "about the same as the DR-learner"
    models = namespace["models"]
    assert abs(models.loc["propensity", "calibration_slope"] - 1) < 0.05
    assert abs(models.loc["outcome", "calibration_slope"] - 1) < 0.05
    assert namespace["solved"]


def _check_targeting(namespace: dict[str, Any]) -> None:
    """The rule earns nearly the best possible gain and stops late, at 57 against 51."""
    gain = np.asarray(namespace["gain"])
    target = np.asarray(namespace["target"])
    rule, best, everyone = (
        np.mean(gain * target),
        np.mean(gain * (gain > 0)),
        np.mean(gain),
    )
    assert rule > 0.95 * best
    assert rule > 1.7 * everyone, "nearly twice a coupon for everyone"
    frame = namespace["frame"]
    assert 56 < frame.loc[target, "age"].max() < 58, "up to about 57"
    true_uplift, cost = namespace["true_uplift"], namespace["COST"]
    assert true_uplift(50.5) > cost > true_uplift(51.5)
    assert namespace["break_even"] == 51


def _check_score(namespace: dict[str, Any]) -> None:
    """On half the rows the cutoff comes out late, the net gain is positive, and it covers."""
    first_curve = namespace["first_curve"]
    assert first_curve.sieve.chosen == 5 == namespace["waves_for"](4_000)
    assert first_curve.status == "supported"
    cutoff = namespace["cutoff"]
    assert cutoff == 60
    # The cutoff rule: the last age before the curve first falls to the cost.
    grid, cost = namespace["grid"], namespace["COST"]
    prediction = first_curve.predict({"age": grid})
    assert prediction[grid == cutoff][0] > cost >= prediction[grid == cutoff + 1][0]
    assert np.all(prediction[grid <= cutoff] > cost)
    rise, coupon_cost, true_net = namespace["rise"], namespace["coupon_cost"], namespace["true_net"]
    assert rise["ci_lower"] - coupon_cost > 0, "the interval excludes breaking even"
    assert covers((rise["ci_lower"] - coupon_cost, rise["ci_upper"] - coupon_cost), true_net)
    assert rise["support"] == "experimental"
    summary = _summary()
    assert "net-gain interval covers the true net gain: 0.935" in summary
    assert "5% / 50% / 95%: [42.0, 50.0, 60.0]" in summary, "this draw's cutoff is late"


def _check_projection(namespace: dict[str, Any]) -> None:
    """Every coefficient interval covers the shape's truth; the pointwise one at 20 misses it."""
    by_age, shape = namespace["by_age"], namespace["shape"]
    terms = namespace["terms_on_grid"]
    assert terms.shape[1] == 4, "evaluate() already holds the constant column"
    true_coef = namespace["true_coef"]
    # Independent of the fit: the coefficients of a least-squares fit on the age grid.
    assert np.allclose(true_coef, [0.1096, 0.0871, 0.0095, -0.0043], atol=5e-4)
    table = namespace["projection"].to_frame()
    assert (table["support"] == "supported").all()
    for j in range(4):
        assert covers((table["ci_lower"].iloc[j], table["ci_upper"].iloc[j]), true_coef[j])
    assert namespace["test"].p_value < 1e-4
    at_ages = namespace["at_ages"]
    band = namespace["shape_curve"].band
    ages = namespace["band_ages"]
    shape_truth = shape.evaluate(pd.DataFrame({"age": ages}), by_age) @ true_coef
    misses = ~((at_ages["ci_lower"] <= shape_truth) & (shape_truth <= at_ages["ci_upper"]))
    assert list(np.flatnonzero(misses)) == [0], "only the interval at age 20 misses"
    assert np.all((band.lower <= shape_truth) & (shape_truth <= band.upper)), "the band holds"
    summary = _summary()
    assert "pointwise interval at age 20 covers: 0.945" in summary
    assert "band covers all seven ages: 0.955" in summary
    assert 0 < namespace["slope_robustness"]["rv"] < 1
    bounds = namespace["slope_bounds"]
    slope = table.set_index("estimand").loc["cate_projection[cos1(age)]", "psi"]
    assert bounds.lower < slope < bounds.upper


def _check_spend(namespace: dict[str, Any]) -> None:
    """The spend term's interval contains 0; leaving age out hurts, leaving spend out does not."""
    row = namespace["spend_row"]
    assert covers((row["ci_lower"], row["ci_upper"]), 0.0)
    assert row["ci_upper"] - row["ci_lower"] < 0.2
    stability = namespace["stability_by_column"].to_frame().set_index("dropped")["change"]
    assert stability["age"] > 0.005 and abs(stability["spend"]) < 0.001


def _check_held_out(namespace: dict[str, Any]) -> None:
    """The held-out ranking beats random and spend; the score's slope covers 1."""
    ranking = namespace["ranking"]
    assert ranking.route == "external"
    assert ranking.status == "supported"
    assert ranking.n_eval == len(namespace["second"]) == 4_000
    at_40 = namespace["at_40"]
    assert at_40["estimate"] > 0.4 * ranking.ate
    autoc = ranking.area("autoc")
    assert autoc["lower"] > 0, "the ranking beats random"
    assert figure_title(namespace["ranking_fig"]) == "Gain from treating by predicted effect first"
    frame = namespace["score_check"].to_frame().set_index("estimand")
    assert covers(tuple(frame.loc["calibration[priority]", ["ci_lower", "ci_upper"]]), 1.0)
    risk = frame.loc["risk_difference"]
    assert risk["psi"] < 0 < risk["ci_upper"] < 0.002, "an interval that just reaches 0"
    assert namespace["score_check"].status == "supported"
    areas = namespace["areas"].set_index("score")
    assert areas.loc["priority - spend", "lower"] > 0
    assert covers(tuple(areas.loc["spend", ["lower", "upper"]]), 0.0)
    assert namespace["versus_spend"].status == "experimental"


def _check_learned_rule(namespace: dict[str, Any]) -> None:
    """The learned rules' interval is above 0, narrower than the hand split's, experimental."""
    row = namespace["gain_row"]
    assert row["ci_lower"] > 0
    rise = namespace["rise"]
    assert row["ci_upper"] - row["ci_lower"] < rise["ci_upper"] - rise["ci_lower"]
    assert row["support"] == "experimental"
    assert row["psi"] > namespace["true_best"], "the estimate exceeds the best possible value"


def _check_sensitivity(namespace: dict[str, Any]) -> None:
    """The net gain falls at a weaker cause than the rise; the bound reads the cost."""
    to_zero, to_cost = namespace["to_zero"], namespace["to_cost"]
    assert 0 < to_cost["rv"] < to_zero["rv"] < 1
    assert to_cost["rva"] < to_zero["rva"]
    # Witness: the cost-null value is the one the bound gives at the cost, not at zero.
    sensitivity, cost = namespace["sensitivity"], namespace["coupon_cost"]
    assert cost > 0.04
    assert sensitivity.robustness_value(null_hypothesis=cost)["rv"] == to_cost["rv"]
    bounds = namespace["bounds"]
    assert bounds.lower - cost > 0, "the net gain stays positive at 0.03"


def _check_trial(namespace: dict[str, Any]) -> None:
    """The risk-ratio curve falls from about 1 to about half; the line covers the true slope."""
    levels = namespace["levels"]
    ratio = np.exp(namespace["ratio_curve"].predict({"severity": levels}))
    assert np.max(np.abs(ratio - np.exp(-0.8 * levels))) < 0.15
    assert ratio[0] > 0.9 and 0.4 < ratio[-1] < 0.6
    assert namespace["ratio_curve"].sieve.mode == "fixed"
    assert namespace["ratio_curve"].status == "experimental", "a fixed size stays experimental"
    slope = namespace["slope"]
    assert covers((slope["ci_lower"], slope["ci_upper"]), -0.8)
    assert slope["ci_upper"] < 0
    assert slope["support"] == "supported"


def check(namespace: dict[str, Any]) -> None:
    """The conditional-effects tutorial's claims hold at the displayed sample size."""
    _check_confounding(namespace)
    _check_protocol(namespace)
    _check_curve(namespace)
    assert figure_title(namespace["curve_fig"]) == "Conditional effect by age"
    _check_assessment(namespace)
    _check_targeting(namespace)
    _check_score(namespace)
    _check_sensitivity(namespace)
    _check_projection(namespace)
    assert figure_title(namespace["shape_fig"])
    _check_spend(namespace)
    _check_held_out(namespace)
    _check_learned_rule(namespace)
    _check_trial(namespace)


def check_refits(namespace: dict[str, Any]) -> None:
    """The refutation, the stress test and the split-sensitivity readings."""
    tests = namespace["refutation"].to_frame().set_index("test")
    rise = namespace["rise"]["psi"]
    assert abs(tests.loc["placebo", "refuted_mean"]) < 0.01, "the placebo rise is near zero"
    assert abs(tests.loc["subset", "refuted_mean"] - rise) < 0.01
    assert abs(tests.loc["random_common_cause", "refuted_mean"] - rise) < 0.005
    assert (tests["verdict"] == "descriptive").all()
    assert 0.09 < namespace["rule_support"].min_support_propensity < 0.11
    cells = namespace["surface"].cells
    assert max(abs(cell.induced_treatment_association) for cell in cells) < 0.06
    stability = namespace["stability"]
    assert stability.relative_sd < 0.01
    assert not stability.warning
