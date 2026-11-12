"""The reviewed semantic callback for ``docs/examples/conditional-effects``.

``tests/unit/test_documentation_runtime.py`` runs the notebook at its documented size and passes
the namespace to :func:`check`.  The slow tier also runs the stability cell and calls
:func:`check_refits`.  Each assertion is a claim the prose makes about this draw, not a
statistical guarantee.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from tests.unit.tutorial_semantics import figure_title

#: The refit cells the rule derives from this notebook today, in notebook order.  A change here is
#: a change in what the fast gate skips, so it shows in the diff of the pull request that causes it.
REFIT_CELLS = ("stability",)

#: Decimals the prose writes that no output prints, each with how it follows from one.
UNPRINTED_DECIMALS = {
    "1.5": "the curve over the true effect at age 51 (about 1.47); _check_targeting recomputes it",
    "1.6": "the curve over the true effect at age 56 (about 1.64); _check_targeting recomputes it",
    "0.1": "the robust standard error of the calibration slope (about 0.12); _check_assessment "
    "recomputes it",
}


def _check_confounding(namespace: dict[str, Any]) -> None:
    """The simple difference overstates the effect in every age band."""
    table = namespace["table"]
    gap = table["simple difference"] - table["true effect"]
    assert (gap > 0.05).all()
    assert table["true effect"].iloc[-1] < 0.02 < 0.05 < table["simple difference"].iloc[-1]


def _check_curve(namespace: dict[str, Any]) -> None:
    """The EP curve falls too steeply, a few points high to 60, and uses one wave."""
    curve = namespace["curve"]
    ages = namespace["ages"]
    truth = namespace["true_uplift"](ages)
    estimate = curve.predict({"age": ages})
    assert np.all(np.diff(estimate) < 0), "the curve only falls"
    young = ages <= 60
    assert np.all((estimate[young] - truth[young] > 0.02) & (estimate[young] - truth[young] < 0.04))
    assert abs(estimate[ages == 70][0]) < 0.01 and estimate[-1] < 0, "near zero after 70"
    assert curve.fitted_method == "ep"
    assert curve.status == "experimental"
    assert curve.inference_status == "no_interval"
    assert curve.sieve.mode == "cv"
    assert curve.sieve.chosen == 1
    chances = curve.diagnostics.support().propensity_quantiles["overall"]
    assert chances[0.0] > 0.1 and chances[1.0] < 0.95, "the overlap reading"


def _check_assessment(namespace: dict[str, Any]) -> None:
    """The check reads near 1 but is too coarse to see the curve's real, too-steep error."""
    calibration = namespace["calibration"]
    assert 0.85 < calibration.slope < 1.05
    assert abs(calibration.intercept) < 0.02
    bins = calibration.to_frame()
    gap = bins["pseudo_outcome_contrast"] - bins["mean_prediction"]
    assert np.max(np.abs(gap)) < 0.02, "the groups sit within 2 points of their stand-ins"
    # Accuracy review round 4, M1: the stand-ins are too noisy at this size to see the error.
    diagnostics = namespace["curve"].diagnostics
    predictions, _ = diagnostics._out_of_fold(calibration.n_folds, calibration.random_state)
    *_, d_t, d_r = diagnostics._pieces()[3]
    stand_in = d_t - d_r
    design = np.column_stack([np.ones_like(predictions), predictions])
    beta = np.linalg.lstsq(design, stand_in, rcond=None)[0]
    residual = stand_in - design @ beta
    bread = np.linalg.inv(design.T @ design)
    slope_se = np.sqrt((bread @ (design.T * residual**2) @ design @ bread)[1, 1])
    assert 0.07 < slope_se < 0.15, "the slope is uncertain by about 0.1"
    groups = np.digitize(predictions, np.quantile(predictions, [0.2, 0.4, 0.6, 0.8]))
    group_se = [stand_in[groups == g].std(ddof=1) / np.sqrt(np.sum(groups == g)) for g in range(5)]
    assert all(0.02 < se < 0.04 for se in group_se), "each group uncertain by about 3 points"
    # The curve is in fact about 20% too steep from 20 to 80.
    ages, truth = namespace["ages"], namespace["true_uplift"](namespace["ages"])
    estimate = namespace["curve"].predict({"age": ages})
    steepness = (estimate[0] - estimate[-1]) / (truth[0] - truth[-1])
    assert 1.15 < steepness < 1.25, steepness
    risks = namespace["risks"]
    assert risks["curve"] < risks["t_learner"] < risks["constant"] < 0
    assert risks["t_learner"] - risks["curve"] > 0.005, "clearly lower"
    assert abs(risks["curve"] - risks["dr_learner"]) < 0.001, "about the same as the DR-learner"


def _check_targeting(namespace: dict[str, Any]) -> None:
    """The rule earns nearly the best possible gain and stops a little late."""
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
    assert 55 < frame.loc[target, "age"].max() < 57, "under about 56"
    # The true gain beats the cost only up to about age 51.
    true_uplift, cost = namespace["true_uplift"], namespace["COST"]
    assert true_uplift(50.5) > cost > true_uplift(51.5)
    assert namespace["break_even"] == 51
    curve = namespace["curve"]
    ratio = curve.predict({"age": np.array([51.0, 56.0])}) / true_uplift(np.array([51.0, 56.0]))
    assert np.allclose(ratio, [1.5, 1.6], atol=0.05), ratio


def _check_trial(namespace: dict[str, Any]) -> None:
    """The risk-ratio curve falls from about 1 to about half, close to the truth."""
    levels = namespace["levels"]
    ratio = np.exp(namespace["ratio_curve"].predict({"severity": levels}))
    assert np.max(np.abs(ratio - np.exp(-0.8 * levels))) < 0.15
    assert ratio[0] > 0.9 and 0.4 < ratio[-1] < 0.6
    assert namespace["ratio_curve"].sieve.mode == "fixed"
    assert namespace["ratio_curve"].status == "experimental", "a fixed size stays experimental"
    assert namespace["ratio_curve"].inference_status == "no_interval"


def _check_score(namespace: dict[str, Any]) -> None:
    """On half the rows the cutoff stays near 56, the net gain is positive, and it covers."""
    assert namespace["first_curve"].sieve.chosen == 1
    assert 50 < namespace["cutoff"] < 58
    rise, coupon_cost, true_net = (
        namespace["rise"],
        namespace["coupon_cost"],
        namespace["true_net"],
    )
    assert rise["ci_lower"] - coupon_cost > 0, "the interval excludes breaking even"
    assert rise["ci_lower"] - coupon_cost < true_net < rise["ci_upper"] - coupon_cost
    assert rise["support"] == "experimental"


def _check_ranking(namespace: dict[str, Any]) -> None:
    """On held-out rows the curve's ranking gains more than random and the interval excludes 0."""
    ranking = namespace["ranking"]
    assert ranking.route == "external"
    assert ranking.status == "supported"
    assert ranking.n_eval == len(namespace["second"]) == 4_000
    at_40 = {row["q"]: row for row in ranking.rows("gain")}[0.4]
    assert abs(at_40["estimate"] - 0.102) < 0.002
    assert abs(0.4 * ranking.ate - 0.056) < 0.002
    autoc = ranking.area("autoc")
    assert abs(autoc["estimate"] - 0.089) < 0.002
    assert autoc["lower"] > 0, "the ranking beats random"
    assert 0.055 < autoc["lower"] < 0.057 and 0.121 < autoc["upper"] < 0.123


def check(namespace: dict[str, Any]) -> None:
    """The conditional-effects tutorial's claims hold at the displayed sample size."""
    _check_confounding(namespace)
    _check_curve(namespace)
    assert figure_title(namespace["curve_fig"]) == "Conditional effect by age"
    _check_assessment(namespace)
    _check_targeting(namespace)
    _check_score(namespace)
    _check_ranking(namespace)
    assert figure_title(namespace["ranking_fig"]) == "Gain from treating by predicted effect first"
    _check_trial(namespace)


def check_refits(namespace: dict[str, Any]) -> None:
    """The split-sensitivity reading is well below the warning level."""
    stability = namespace["stability"]
    assert stability.relative_sd < 0.3
    assert not stability.warning
