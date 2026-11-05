"""The reviewed semantic callback for ``docs/examples/conditional-effects``.

``tests/unit/test_documentation_runtime.py`` runs the notebook at its documented size and passes
the namespace to :func:`check`.  The slow tier also runs the stability cell and calls
:func:`check_refits`.  Each assertion is a claim the prose makes about this draw, not a
statistical guarantee.
"""

from __future__ import annotations

from typing import Any

import numpy as np

#: The refit cells the rule derives from this notebook today, in notebook order.  A change here is
#: a change in what the fast gate skips, so it shows in the diff of the pull request that causes it.
REFIT_CELLS = ("stability",)

#: Decimals the prose writes that no output prints, each with how it follows from one.
UNPRINTED_DECIMALS = {
    "0.05": "the calibration line at a prediction of 0.03: intercept 0.028 + slope 0.76 x 0.03",
    "0.22": "the calibration line at a prediction of 0.25: intercept 0.028 + slope 0.76 x 0.25",
}


def _check_confounding(namespace: dict[str, Any]) -> None:
    """The simple difference overstates the effect in every age band, most in the oldest."""
    table = namespace["table"]
    gap = table["simple difference"] - table["true effect"]
    assert (gap > 0.05).all()
    assert table["true effect"].iloc[-1] < 0.02 < 0.1 < table["simple difference"].iloc[-1]


def _check_curve(namespace: dict[str, Any]) -> None:
    """The EP curve follows the truth, is experimental, and uses one function of age."""
    curve = namespace["curve"]
    ages = namespace["ages"]
    truth = namespace["true_uplift"](ages)
    estimate = curve.predict({"age": ages})
    assert np.max(np.abs(estimate - truth)) < 0.04
    assert np.all(np.diff(estimate) < 0), "the reading says the curve only falls"
    assert curve.fitted_method == "ep"
    assert curve.status == "experimental"
    assert curve.inference_status == "no_interval"
    assert curve.sieve.mode == "cv"
    assert curve.sieve.chosen == 1
    assert 1.8 < estimate[4] / truth[4] < 2.3, "the reading says about double at 60"
    chances = curve.diagnostics.support().propensity_quantiles["overall"]
    assert chances[0.0] > 0.1 and chances[1.0] < 0.9, "the overlap reading"


def _check_assessment(namespace: dict[str, Any]) -> None:
    """The slope is below 1, and the curve has the lowest out-of-fold error."""
    calibration = namespace["calibration"]
    assert 0.5 < calibration.slope < 1.0
    risks = namespace["risks"]
    assert risks["curve"] < risks["t_learner"] < risks["constant"] < 0
    assert abs(risks["curve"] - risks["dr_learner"]) < 0.001, "about the same as the DR-learner"
    bins = calibration.to_frame()
    middle = bins.iloc[2]
    assert middle["pseudo_outcome_contrast"] < 0.5 * middle["mean_prediction"], "a noisy bin"
    line = calibration.intercept + calibration.slope * np.array([0.03, 0.25])
    assert np.allclose(line, [0.05, 0.22], atol=0.005), "the narrated line values"


def _check_targeting(namespace: dict[str, Any]) -> None:
    """The rule earns nearly the best possible gain and far more than a coupon for all."""
    gain = np.asarray(namespace["gain"])
    target = np.asarray(namespace["target"])
    rule, best, everyone = (
        np.mean(gain * target),
        np.mean(gain * (gain > 0)),
        np.mean(gain),
    )
    assert rule > 0.9 * best
    assert rule > 1.5 * everyone
    frame = namespace["frame"]
    assert frame.loc[target, "age"].max() < 60
    # The true gain beats the cost only up to about age 51.
    true_uplift, cost = namespace["true_uplift"], namespace["COST"]
    assert true_uplift(50.5) > cost > true_uplift(51.5)
    assert frame.loc[target, "age"].max() > 55, "the rule stops too late"
    near_cutoff = namespace["curve"].predict({"age": np.array([58.0])})[0] / true_uplift(58.0)
    assert 1.7 < near_cutoff < 2.3, "up to about twice the true effect at the cutoff"


def _check_trial(namespace: dict[str, Any]) -> None:
    """The risk-ratio curve falls from about 1 to about the true 0.45."""
    levels = namespace["levels"]
    ratio = np.exp(namespace["ratio_curve"].predict({"severity": levels}))
    assert np.max(np.abs(ratio - np.exp(-0.8 * levels))) < 0.15
    assert ratio[0] > 0.9 and ratio[-1] < 0.6
    assert 0.7 < namespace["trial_calibration"].slope < 1.3
    assert namespace["ratio_curve"].sieve.mode == "fixed"


def check(namespace: dict[str, Any]) -> None:
    """The conditional-effects tutorial's claims hold at the displayed sample size."""
    _check_confounding(namespace)
    _check_curve(namespace)
    _check_assessment(namespace)
    _check_targeting(namespace)
    _check_trial(namespace)


def check_refits(namespace: dict[str, Any]) -> None:
    """The split-sensitivity reading is well below the warning level."""
    stability = namespace["stability"]
    assert stability.relative_sd < 0.3
    assert not stability.warning
