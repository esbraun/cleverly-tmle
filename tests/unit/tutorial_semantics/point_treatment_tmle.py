"""The reviewed semantic callback for ``docs/examples/point-treatment-tmle``.

``tests/unit/test_documentation_runtime.py`` runs the tutorial at its documented size and
passes the resulting namespace to :func:`check`.  The tutorial is a notebook, so its narrated
decimals are also compared against its stored outputs, and every decimal it writes is printed.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pytest

from cleverly.datasets import navigation_protocol
from cleverly.datasets.synthetic import nonlinear_bounded_dgp
from tests.documents import ROOT
from tests.unit.tutorial_semantics import (
    EXAMPLES,
    assert_protocol_recorded,
    covers,
)

NOTEBOOK = EXAMPLES / "point-treatment-tmle.ipynb"


def _untreated_selection_bias(n: int = 400_000) -> float:
    """``E[Y^0 | A=1] - E[Y^0 | A=0]`` in the law behind ``navigation_data``, by Monte Carlo."""
    law = nonlinear_bounded_dgp()
    latent = np.random.default_rng(0).standard_normal((n, law.n_latent))
    g = law.propensity(latent)
    untreated = law.outcome_mean(latent, 0.0, None)
    return float(np.average(untreated, weights=g) - np.average(untreated, weights=1.0 - g))


def _law_moments(n: int = 400_000) -> dict[str, float]:
    """``nu^2 = E[1 / {g (1 - g)}]`` and the share of true ``g`` below 0.1, by Monte Carlo."""
    law = nonlinear_bounded_dgp()
    g = law.propensity(np.random.default_rng(1).standard_normal((n, law.n_latent)))
    return {"nu2": float(np.mean(1.0 / (g * (1.0 - g)))), "below_0.1": float(np.mean(g < 0.1))}


_PROBE = "reviews/notebook-review/probes/point-treatment-tmle-final"
#: Decimals the Step 10 readings quote from the committed refutation probe logs.
PROBE_LITERALS = (
    "0.3074",
    "0.0740",
    "1.1121",
    "0.0039922",
    "0.0879",
    "-0.0101",
    "0.1859",
    "0.0349",
    "-0.0631",
    "0.1329",
    "1.0349",
    "0.9660",
    "1.1088",
    "0.9179",
    "0.9652",
    "0.9450",
    "0.85",
    "1.15",
    "0.88",
    "0.99",
)
_REFUTE_PROBE = "reviews/notebook-review/probes/point-treatment-tmle-refuters"

#: Decimals the readings write that no stored output prints, each with its source.
UNPRINTED_DECIMALS = {
    "0.98": f"mean SE over empirical SD of the Step 6 fit, 200 draws: {_PROBE}/summary.log",
    "1.9": f"share (0.0188) of the law's true propensity below 0.1: {_PROBE}/truth.log",
    "4.83": f"the law's nu^2 = E[1/(g(1-g))], 4.8286 by Monte Carlo: {_PROBE}/truth.log",
    "-0.018": f"mean error of the linear-Q plug-in over 200 draws: {_PROBE}/plugin.log",
    "-0.0022": f"mean error of TMLE with linear Q and the booster's g: {_PROBE}/summary.log",
    "-0.0006": f"mean error of TMLE with the booster's Q and logistic g: {_PROBE}/summary.log",
    "-0.0218": f"mean error of TMLE with both nuisances linear: {_PROBE}/summary.log",
    "0.431": f"RV of the shown draw at the law's nu^2, by summarize.py: {_PROBE}/summary.log",
    "0.0578": "the size of the largest printed move, -0.0578, at outcome strength 2.0",
    **dict.fromkeys(
        PROBE_LITERALS,
        f"a value of the Step 10 review probe logs, {_REFUTE_PROBE}/standard.log and generated.log",
    ),
}


def _assert_committed_numbers() -> None:
    """Each probe number of ``UNPRINTED_DECIMALS`` matches its committed file."""
    probe = ROOT / _PROBE
    summary = (probe / "summary.log").read_text(encoding="utf-8")
    truth = (probe / "truth.log").read_text(encoding="utf-8")
    plugin = (probe / "plugin.log").read_text(encoding="utf-8")

    def number(text: str, pattern: str) -> float:
        match = re.search(pattern, text)
        assert match, pattern
        return float(match.group(1))

    assert round(number(summary, r"SE/SD (\S+)"), 2) == 0.98
    assert "coverage 0.945 (MC SE 0.016; 189 of 200)" in summary
    assert "lin_q_flex_g  coverage 0.975 (MC SE 0.011; 195 of 200)" in summary
    assert "flex_q_lin_g  coverage 0.950 (MC SE 0.015; 190 of 200)" in summary
    assert "both_linear   coverage 0.150 (MC SE 0.025; 30 of 200)" in summary
    assert round(number(summary, r"lin_q_flex_g .*bias (\S+)"), 4) == -0.0022
    assert round(number(summary, r"flex_q_lin_g .*bias (\S+)"), 4) == -0.0006
    assert round(number(summary, r"both_linear .*bias (\S+)"), 4) == -0.0218
    assert "95% limits of the bounds exclude zero: 1.000 (MC SE 0.000; 200 of 200)" in summary
    assert round(number(summary, r"law +nu\^2 [\d.]+: .*rv (\S+)"), 3) == 0.431
    assert round(100 * number(truth, r"share g < 0.10 +(\S+)"), 1) == 1.9
    assert round(number(truth, r"nu2 +(\S+)"), 2) == 4.83
    assert number(truth, r"min g +(\S+)") >= 0.05
    assert number(truth, r"max g +(\S+)") <= 0.95
    assert round(number(plugin, r"bias (\S+)"), 3) == -0.018
    assert "seeds 6000 to 6199, 200 draws" in plugin


def _check_probe_logs() -> None:
    """Each decimal of ``PROBE_LITERALS`` is in a committed probe log, at its printed form."""
    probe = ROOT / _REFUTE_PROBE
    text = (probe / "standard.log").read_text(encoding="utf-8")
    text += (probe / "generated.log").read_text(encoding="utf-8")
    for literal in PROBE_LITERALS:
        assert literal in text, f"{literal} is not in the Step 10 probe logs"
    assert "simulated_outcome: 400 draws, level L2, verdict passed" in text
    assert "plasmode: 400 draws, level L2, verdict passed" in text


def _check_refutation(namespace: dict[str, Any]) -> None:
    """Step 10: the refutation tests and the generated outcomes read as the prose says."""
    estimate = namespace["estimate"]
    refutation = namespace["refutation"]
    tests = {test.name: test for test in refutation.tests}
    assert list(tests) == ["placebo", "random_common_cause", "subset"]
    for test in tests.values():
        # No registry row covers this configuration, so every test is descriptive at L0.
        assert (test.rung, test.verdict, test.passed) == ("L0", "descriptive", None)
        assert len(test.values) == 20
        assert test.original == estimate.psi
    assert refutation.passed is None
    assert (
        "This fit differs on learner HistGradientBoostingRegressor("
        in tests["placebo"].scope_sentence
    )

    # Placebo: the exact null is 0, and the mean placebo estimate sits far below the effect.
    placebo = tests["placebo"]
    assert placebo.expectation == "~ 0"
    assert placebo.statistic_pvalue > 0.05
    assert abs(placebo.mean) < 0.01 * abs(estimate.psi)
    # A permuted offer carries no effect, so a pipeline that kept it would sit near the estimate.
    assert abs(placebo.mean - estimate.psi) > 100.0 * abs(placebo.mean)

    # Random common cause: the shift is a fraction of a standard error.
    assert abs(tests["random_common_cause"].statistic) < 0.1

    # Subset: the reference is SE * sqrt((n/m - 1) * n/(n - 1)) with m = 2100 of n = 3000.
    n, m = len(namespace["frame"]), 2100
    reference = estimate.std_error * np.sqrt((n / m - 1.0) * n / (n - 1.0))
    printed = float(tests["subset"].expectation.removeprefix("scatter ~ "))
    assert abs(printed - reference) < 1e-5
    # Nonzero witness: the retired reference SE * sqrt(n/m) is almost twice as large.
    assert estimate.std_error * np.sqrt(n / m) > 1.5 * reference
    assert 0.5 < tests["subset"].statistic < 1.5

    # Generated outcomes: the declared law's truth, written out from its equations.
    declared = namespace["declared"]
    assert (declared.intercept, declared.effect, declared.adjustment_scale) == (-0.5, 0.4, 0.5)
    covariates = np.asarray(namespace["frame"][namespace["covariates"]], dtype=float)
    standardized = (covariates - covariates.mean(axis=0)) / covariates.std(axis=0)
    shift = declared.adjustment_scale * standardized.sum(axis=1) / np.sqrt(covariates.shape[1])

    def expit(x: Any) -> Any:
        return 1.0 / (1.0 + np.exp(-x))

    truth = float(
        np.mean(
            expit(declared.intercept + declared.effect + shift) - expit(declared.intercept + shift)
        )
    )
    assert 0.09 < truth < 0.095
    generated = namespace["generated_tests"]
    assert list(generated) == ["simulated_outcome", "plasmode"]
    for test in generated.values():
        bias = test.readings[0]
        assert test.n_replicates == 20 and test.rung == "L1" and test.verdict == "descriptive"
        assert bias.name == "bias" and bias.band == (-0.2, 0.2)
        # 20 draws give an interval wider than the band, which is why no verdict prints.
        assert bias.upper - bias.lower > 2 * bias.band[1]
        assert bias.lower < 0.0 < bias.upper
    assert generated["simulated_outcome"].declared_effect == pytest.approx(truth, abs=1e-10)
    # The plasmode truth is a sum over the same support points, with its own draws of A.
    assert generated["plasmode"].declared_effect == pytest.approx(truth, abs=1e-3)


def _check_evalue(namespace: dict[str, Any]) -> None:
    """Step 11: the E-value reads as the prose says."""
    evalue = namespace["evalue"]
    estimate = namespace["estimate"]
    assert evalue.approximate
    assert evalue.point > evalue.limit > 1.0
    # The standardized-difference route: the risk ratio is above 1 for a positive effect.
    assert evalue.risk_ratio > 1.0
    outcome = np.asarray(namespace["frame"]["transition_score"], dtype=float)
    assert 0.23 < outcome.std() < 0.24
    assert estimate.psi > 0.0


def _check_odds_bounds(namespace: dict[str, Any]) -> None:
    """The bounds on the odds of an offer read as the prose says."""
    bounds = namespace["odds_bounds"]
    treated = namespace["with_offer"].rows
    untreated = namespace["without_offer"].rows
    assert [row.gamma for row in bounds.rows] == [1.0, 2.0, 3.0, 4.0]
    # "At gamma = 1 each range is a single value ... 0.167, matches the estimate".
    for first in (bounds.rows[0], treated[0], untreated[0]):
        assert first.lower == pytest.approx(first.upper, abs=1e-12)
    assert bounds.rows[0].lower == pytest.approx(namespace["estimate"].psi, abs=5e-4)
    # "The ranges then widen": each row nests the one before it.
    for before, after in zip(bounds.rows, bounds.rows[1:], strict=False):
        assert after.lower < before.lower and after.upper > before.upper
    # "where that room comes from": the effect range is the difference of the group ranges,
    # a nonzero witness that each end pairs the opposite ends of the two arms.
    for row, one, zero in zip(bounds.rows, treated, untreated, strict=True):
        assert row.lower == pytest.approx(one.lower - zero.upper, abs=1e-12)
        assert row.upper == pytest.approx(one.upper - zero.lower, abs=1e-12)
        assert row.upper - row.lower > (one.upper - one.lower) + 1e-12 or row.gamma == 1.0
    # "if everyone were offered ... as low as 0.476": a bound on the mean under offering to
    # all, which sits well below the offered group's observed mean, so it is not that mean.
    frame = namespace["frame"]
    offered = frame["transition_navigation"] == 1
    assert treated[3].lower < float(frame["transition_score"][offered].mean()) - 0.05
    # "stay above zero through gamma = 3 and fall to -0.0032 at gamma = 4".
    assert bounds.rows[2].lower_limit > 0.0 > bounds.rows[3].lower_limit
    assert bounds.limit_breakdown == 4.0
    assert bounds.breakdown is None
    # "no simulation study backs these limits for this fit. The study behind them used linear
    # models and at least 8,000 patients, while this fit uses boosted models and 3,000".
    assert bounds.limit_descriptive
    sentence = bounds.limit_reading.sentence
    assert "This fit differs on learner HistGradientBoostingRegressor(" in sentence
    assert "n = 3000 (below 8000)" in sentence
    assert "'LinearRegression()'" in sentence


def _check_surface(namespace: dict[str, Any]) -> None:
    """Step 11: the stress surface reads as the prose says."""
    estimate = namespace["estimate"]

    cells = namespace["cells"].set_index(["treatment_strength", "outcome_strength"])
    anchor = cells.loc[(0.0, 0.0)]
    assert anchor["estimate"] == pytest.approx(estimate.psi, abs=1e-10)
    # Every perturbed cell lowers the estimate, and the movement grows along each axis.
    assert (cells["displacement"].drop((0.0, 0.0)) < 0.0).all()
    for treatment in cells.index.get_level_values(0).unique():
        row = cells.loc[treatment, "displacement"].to_numpy()
        assert (np.diff(row) < 0.0).all(), treatment
    for outcome_strength in cells.index.get_level_values(1).unique():
        column = cells.xs(outcome_strength, level=1)["displacement"].to_numpy()
        assert (np.diff(column) < 0.0).all(), outcome_strength
    # "The latent variable is almost uncorrelated with the offer": the induced association is
    # below 0.05 in every cell, so the treatment axis moves the estimate by misclassification.
    assert cells["induced_treatment_association"].abs().max() < 0.05
    assert cells.loc[(0.0, 2.0), "induced_outcome_association"] < -0.8
    calibration = namespace["calibration"].set_index(["covariate", "role"])["strength"]
    assert (
        calibration[("discharge_risk", "treatment")]
        > calibration[("medication_burden", "treatment")]
    )
    assert max(cells.index.get_level_values(0)) >= calibration[("discharge_risk", "treatment")]


def check(namespace: dict[str, Any]) -> None:
    """The ATE tutorial's graph, estimand, failure mode, diagnostics, and sensitivity claims hold."""
    _assert_committed_numbers()
    # The full summary prints the protocol's fields; the short default leaves them out.
    assert "Discharge-home order" not in namespace["effect"].summary()
    # A full summary wraps its lines, so each check reads the words in order.
    summary = " ".join(str(namespace["effect"].summary(detail="full")).split())
    assert "Discharge-home order" in summary
    assert "scores death before day 30 as the worst transition score" in summary
    assert "Analyze the offer regardless of completed contacts" in summary
    assert dict(namespace["result"].summary().header)["Out-of-fold predictions"] != "no"
    assert list(namespace["frame"].columns[:2]) == ["transition_score", "transition_navigation"]

    # Step 2: "all four covariates enter g" and "enter Q".  Moving one covariate at a time
    # moves both functions, so each covariate is a confounder of the law.  A covariate that
    # entered neither would leave both unchanged.
    law = nonlinear_bounded_dgp()
    base = np.random.default_rng(2).standard_normal((2_000, law.n_latent))
    for column in range(law.n_latent):
        moved = base.copy()
        moved[:, column] += 0.5
        assert np.abs(law.propensity(moved) - law.propensity(base)).max() > 1e-3, column
        assert (
            np.abs(law.outcome_mean(moved, 0.0, None) - law.outcome_mean(base, 0.0, None)).max()
            > 1e-3
        ), column
    # "a true propensity between 0.05 and 0.95".
    wide = law.propensity(np.random.default_rng(3).standard_normal((200_000, law.n_latent)) * 3.0)
    assert wide.min() >= 0.05
    assert wide.max() <= 0.95

    # Step 3: "The offered patients score ... higher", and the arms differ on discharge risk first.
    by_arm = namespace["by_arm"]
    truth = namespace["truth"]
    unadjusted = namespace["unadjusted"]
    assert unadjusted > truth["ate"]
    assert by_arm.loc[0.0, "discharge_risk"] < 0.0 < by_arm.loc[1.0, "discharge_risk"]
    assert by_arm.loc[1.0, "discharge_risk"] > by_arm.loc[0.0, "discharge_risk"] + 0.3
    # "Two mechanisms of this law raise the unadjusted difference": the ATT exceeds the ATE,
    # and the offered patients would score higher under usual support alone.  Both have the
    # same sign in this law.  The margins are the retired Gaussian law's, scaled by the ratio
    # of the two ATEs (0.1629 against 1.750).
    assert truth["att"] > truth["ate"] + 0.01
    assert truth["att"] > truth["ate"] > truth["atc"]
    assert _untreated_selection_bias() > 0.01

    # Step 4: the protocol step prints the record, and the fit carries its digest.
    assert namespace["protocol"] == navigation_protocol()
    assert_protocol_recorded(NOTEBOOK, "protocol", namespace["protocol"], namespace["result"])

    # Step 6: the headline interval covers the truth on this draw.  The repeated-draw numbers
    # the reading quotes (189 of 200, SE/SD 0.98) come from the committed probe.  The declared
    # q_bounds are the score's own support: the scaler is the identity, and the outcome range
    # is the nonzero witness that the support holds the data rather than being padded around it.
    ate = truth["ate"]
    assert covers(namespace["estimate"], ate)
    scaler = namespace["result"].nuisance.scaler
    assert (scaler.lower, scaler.upper) == (0.0, 1.0)
    assert scaler.is_identity
    outcome = np.asarray(namespace["frame"]["transition_score"], dtype=float)
    assert scaler.lower < outcome.min() < outcome.max() < scaler.upper

    # Step 7: "that average equals the regression coefficient on the offer".
    linear_q = namespace["linear_q"]
    plug_in = namespace["plug_in"]
    assert linear_q.feature_names_in_[0] == "transition_navigation"
    assert abs(plug_in - float(linear_q.coef_[0])) < 1e-10
    # The plug-in misses by more than TMLE with the same Q class and the booster's g, and that
    # fit and the flexible-Q fit cover the truth.  The both-linear fit misses by more than the
    # plug-in and its interval excludes the truth.  The probes give the repeated-draw numbers.
    errors = {label: abs(point.psi - ate) for label, point in namespace["dr_points"].items()}
    assert abs(plug_in - ate) > 3.0 * errors["linear Q, flexible g"]
    one_flexible = max(errors["flexible Q, linear g"], errors["linear Q, flexible g"])
    assert errors["both linear"] > one_flexible
    assert errors["both linear"] > abs(plug_in - ate)
    for label in ("flexible Q, linear g", "linear Q, flexible g"):
        assert covers(namespace["dr_points"][label], ate), label
    assert not covers(namespace["dr_points"]["both linear"], ate)

    # Step 8: no row needs attention, the score-equation check passed, and the shallow booster's
    # calibration slope is near 1.  The verdict the reading quotes is the report's own.
    assessment = namespace["assessment"]
    assert tuple(item.name for item in assessment.attention) == ()
    ledger = assessment.to_frame()
    assert set(ledger.loc[ledger["check"] == "score_equations", "status"]) == {"passed"}
    assert "nuisance fits look reasonable" in namespace["nuisance"].summary()
    (propensity,) = (m for m in namespace["nuisance"].models if m.name == "propensity")
    assert abs(propensity.metrics["calibration_slope"] - 1.0) < 0.1
    # "no unit sits at the truncation bound".
    support = namespace["support"]
    assert support.truncated["fraction"] == 0.0
    assert support.truncated["most_extreme"] > 0.05
    # "It puts a share of 0.0027 of the rows below 0.1 ... In the law, about 1.9%": the fitted
    # tail is nonzero and thinner than the law's.
    fitted_below = support.tail_mass[0.1]["below"]
    moments = _law_moments()
    assert 0.0 < fitted_below < 0.5 * moments["below_0.1"]
    assert 0.018 < moments["below_0.1"] < 0.020

    # Step 9: the default bound clips no row on this law, and the two tight bounds do.  The
    # fitted bound comes from 5 / (sqrt(n) log n), and a retarget there reproduces Step 6.
    curve = namespace["curve"]
    n = len(namespace["frame"])
    fitted_bound = 5.0 / (np.sqrt(n) * np.log(n))
    nearest = curve.loc[(curve["bound"] - fitted_bound).abs().idxmin()]
    assert abs(nearest["bound"] - fitted_bound) < 1e-12
    assert abs(nearest["psi"] - namespace["estimate"].psi) < 1e-10
    loose = curve[curve["bound"] <= 0.05 + 1e-12]
    assert len(loose) >= 5
    assert (loose["truncated_fraction"] == 0.0).all()
    assert (loose["psi"] == namespace["estimate"].psi).all()
    tight = {
        round(float(row["bound"]), 4): row for _, row in curve.iterrows() if row["bound"] > 0.05
    }
    assert set(tight) == {0.1, 0.2}
    assert tight[0.1]["truncated_fraction"] > 0.0
    assert tight[0.2]["truncated_fraction"] > tight[0.1]["truncated_fraction"]
    # Nonzero witness: "the estimate moves to 0.1674" at the tightest bound.
    assert tight[0.2]["psi"] - namespace["estimate"].psi > 5e-4

    # Step 10: the refutation readings quote committed probe logs.  The refits themselves are
    # checked in check_refits.
    _check_probe_logs()

    # Step 11: the default nu^2 estimator runs, so the robustness value carries its limit.
    # The robustness value is derived from the fit, so the fast gate recomputes it here.
    result = namespace["result"]
    robustness = result.sensitivity.robustness_value()
    assert "nu2_estimator" not in robustness
    assert 0.0 < robustness["rva"] < robustness["rv"] < 1.0
    elements = result.sensitivity.elements(estimand="ate")
    assert elements.nu2_estimator == "doubly_robust"
    # "the law's value is about 4.83 ... The bounds are then slightly narrow": a positive
    # estimate under the finite population value. "slightly" fails loudly above 10%.
    assert 0.0 < elements.nu2 < moments["nu2"]
    assert elements.nu2 > 0.9 * moments["nu2"]
    assert abs(moments["nu2"] - 4.83) < 0.02
    # "equal strengths ... would move the point estimate to zero", and the 95% limit at rva.
    at_rv = result.sensitivity.omitted_confounding(cf_y=robustness["rv"], cf_d=robustness["rv"])
    assert abs(at_rv.lower) < 1e-3
    at_rva = result.sensitivity.omitted_confounding(cf_y=robustness["rva"], cf_d=robustness["rva"])
    assert abs(at_rva.ci_lower) < 1e-3
    _check_evalue(namespace)
    _check_odds_bounds(namespace)


def check_refits(namespace: dict[str, Any]) -> None:
    """Steps 10 and 11: the refutation refits, the benchmark refits, and the stress surface."""
    _check_refutation(namespace)
    moments = _law_moments()
    result = namespace["result"]
    robustness = namespace["robustness"]
    assert robustness == result.sensitivity.robustness_value()
    elements = result.sensitivity.elements(estimand="ate")
    at_rv = result.sensitivity.omitted_confounding(cf_y=robustness["rv"], cf_d=robustness["rv"])
    # "The residual outcome variance ... more than doubles", so the implied cf_y is a clip.
    # The bound is finite at cf_y = 1 (RM47), so the clipped benchmark still calibrates one.
    strong = namespace["strong"]
    assert strong.covariates == ("discharge_risk",)
    assert strong.sigma2_short > 2.0 * strong.sigma2_long
    assert strong.cf_y == 1.0 < strong.gain_y
    strong_bounds = namespace["strong_bounds"]
    assert (strong_bounds.cf_y, strong_bounds.cf_d, strong_bounds.rho) == (1.0, strong.cf_d, 1.0)
    # "The bound is finite there, because the treatment strength `cf_d` stays at 0.0282", and
    # "both one-sided limits exclude zero" at the discharge_risk strengths too.
    assert 0.0 < strong.cf_d < 0.05
    assert 0.0 < strong_bounds.ci_lower < strong_bounds.lower < strong_bounds.psi
    assert strong_bounds.bias > namespace["bounds"].bias
    # "does not cover a confounder that changes the offer more": a larger cf_d at cf_y = 1
    # moves the bound further, which is the nonzero witness of that sentence.
    wider = result.sensitivity.omitted_confounding(cf_y=1.0, cf_d=2.0 * strong.cf_d, rho=1.0)
    assert wider.bias > strong_bounds.bias
    # "would need a weight strength of about 0.24 to pull the lower limit to zero, more than three
    # times that of any recorded characteristic": the bisection lands on the crossing, and the
    # limit is positive just below it and negative just above it.
    tipping = namespace["tipping_cf_d"]
    assert round(tipping, 2) == 0.24
    recorded = ("age", "medication_burden", "prior_utilization", "discharge_risk")
    weight_strengths = [result.sensitivity.benchmark(covariates=(name,)).cf_d for name in recorded]
    assert 3.0 * max(weight_strengths) < tipping < 4.0 * max(weight_strengths)
    crossing = result.sensitivity.omitted_confounding(cf_y=1.0, cf_d=tipping)
    assert abs(crossing.ci_lower) < 1e-6
    assert result.sensitivity.omitted_confounding(cf_y=1.0, cf_d=tipping - 0.01).ci_lower > 0.0
    assert result.sensitivity.omitted_confounding(cf_y=1.0, cf_d=tipping + 0.01).ci_lower < 0.0
    # "Both lie below the robustness value", and the gains go to the bound unchanged, at rho = 1.
    benchmark = namespace["benchmark"]
    assert benchmark.covariates == ("medication_burden",)
    assert 0.0 < benchmark.cf_d < benchmark.cf_y < robustness["rv"]
    bounds = namespace["bounds"]
    assert (bounds.cf_y, bounds.cf_d, bounds.rho) == (benchmark.cf_y, benchmark.cf_d, 1.0)
    assert bounds.confounding_strength < at_rv.confounding_strength
    # "Both limits exclude zero on this draw".
    assert 0.0 < bounds.ci_lower < bounds.lower < bounds.psi < bounds.upper < bounds.ci_upper
    assert "the sign of the effect survives" in bounds.summary()
    # "the bias bound would be about 5% larger" with the law's nu^2: the bias scale is
    # proportional to sqrt(nu^2).
    assert 1.03 < np.sqrt(moments["nu2"] / elements.nu2) < 1.07
    _check_surface(namespace)
