"""The reviewed semantic callback for ``docs/examples/survey-nonresponse``.

``tests/unit/test_documentation_runtime.py`` runs the tutorial at its documented size and
passes the resulting namespace to :func:`check`.  The tutorial is a notebook, so its narrated
decimals are also compared against its stored outputs, and every decimal it writes is printed.
"""

from __future__ import annotations

import csv
from typing import Any

import numpy as np
import pytest
from scipy.special import logit
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression

from cleverly import CapabilityError, CrossFitting
from cleverly._summary import vocabulary
from cleverly.datasets import missing_outcome_bounded_dgp, navigation_protocol
from cleverly.sensitivity import ConfounderStrengthGrid, missingness_tilt, simulated_confounding
from cleverly.sensitivity.omitted_variable import OMITTED_VARIABLE_OPERATIONS
from cleverly.validation import BoundedAdjustmentOutcome, GaussianAdjustmentOutcome
from cleverly.validation.refute import refute_refusal
from tests.unit.tutorial_semantics import (
    EXAMPLES,
    assert_protocol_recorded,
    changed_fields,
    covers,
)

NOTEBOOK = EXAMPLES / "survey-nonresponse.ipynb"

_PROBE = "reviews/notebook-review/probes/sn-bounded"

#: Decimals the readings write that no stored output prints, each with its source.
UNPRINTED_DECIMALS = {
    "0.173": "complete-case target E[tau(W) | Delta = 1] of the law (0.172772), by quadrature: "
    f"{_PROBE}/summary.log",
    "0.960": f"complete-case intervals that contain 0.172772 (480 of 500): {_PROBE}/summary.log",
    "0.00043": f"mean complete-case estimate minus 0.172772, 500 draws: {_PROBE}/summary.log",
    "0.00023": f"Monte Carlo SE of the mean complete-case estimate: {_PROBE}/summary.log",
    "3.17": f"5th percentile of the interval tipping gamma (3.1721): {_PROBE}/summary.log",
    "5.48": f"95th percentile of the interval tipping gamma (5.4783): {_PROBE}/summary.log",
}

_REFUTE_PROBE = "reviews/notebook-review/probes/sn-refute/summary.log"

#: Decimals of the 400-draw readings the Details quote, which no cell prints.  Each appears in the log.
_REFUTE_LOG_DECIMALS = ("-0.134", "-0.232", "-0.03598", "0.004196")
UNPRINTED_DECIMALS.update(
    dict.fromkeys(_REFUTE_LOG_DECIMALS, f"a value of the 400-draw review probe, {_REFUTE_PROBE}")
)
UNPRINTED_DECIMALS.update(
    {
        "9.4": f"the longest wall time of a probe run, 566 s, in {_REFUTE_PROBE}",
        "0.00056": f"the truth 0.10287 minus the simulated_outcome mean 0.10231, {_REFUTE_PROBE}",
    }
)


def _quadrature_targets() -> tuple[float, float]:
    """The ATE and the complete-case target ``E[tau(W) | Delta = 1]`` of the law."""
    nodes, weights = np.polynomial.hermite_e.hermegauss(40)
    weights = weights / weights.sum()
    grid = np.stack(np.meshgrid(nodes, nodes, nodes, indexing="ij"), -1).reshape(-1, 3)
    mass = np.prod(np.stack(np.meshgrid(weights, weights, weights, indexing="ij"), -1), -1)
    mass = mass.reshape(-1)
    law = missing_outcome_bounded_dgp()
    g = law.propensity(grid)
    respond = g * law.missingness(grid, 1.0) + (1 - g) * law.missingness(grid, 0.0)
    effect = law.outcome_mean(grid, 1.0, None) - law.outcome_mean(grid, 0.0, None)
    ate = float((effect * mass).sum())
    return ate, float((effect * respond * mass).sum() / (respond * mass).sum())


def _read(path: str) -> list[dict[str, str]]:
    with (EXAMPLES.parents[1] / path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def check(namespace: dict[str, Any]) -> None:
    """Witness the identification, graph, comparison, diagnostic, and sensitivity claims.

    The committed review probe also backs each quoted number.
    """
    law = missing_outcome_bounded_dgp()
    truth = namespace["truth"]["ate"]
    ate, cc_target = _quadrature_targets()
    # The library truth and an independent quadrature agree, and the complete-case target
    # differs from the ATE by a visible amount.
    assert ate == pytest.approx(truth, abs=1e-5)
    assert cc_target > ate + 0.02

    # The review probe: each quoted value recomputed from its committed rows.
    probe = _read(f"{_PROBE}/sweep.csv")
    assert len(probe) == 500
    ate_hits = sum(row["covers"] == "True" for row in probe)
    cc = np.array([float(row["cc_psi"]) for row in probe])
    assert ate_hits == 478
    assert sum(row["cc_covers_ate"] == "True" for row in probe) == 0
    assert sum(row["cc_covers_target"] == "True" for row in probe) == 480
    assert f"{cc.mean() - cc_target:.5f}" == "0.00043"
    assert f"{cc.std(ddof=1) / np.sqrt(500):.5f}" == "0.00023"
    tips = np.array([float(row["tip_ci"]) for row in probe if row["tip_ci"]])
    assert len(tips) == 485
    assert sum(bool(row["tip_point"]) for row in probe) == 358
    assert (f"{np.percentile(tips, 5):.2f}", f"{np.percentile(tips, 95):.2f}") == ("3.17", "5.48")

    effect = namespace["effect"]
    fitted = namespace["full"]
    expression = effect.functional.expression
    assumptions = " ".join(effect.identification.assumptions).lower()
    nuisances = effect.identification.required_nuisances
    summary = effect.summary(detail="full")

    assert "responded=1" in expression
    assert "missingness at random" in assumptions
    assert "response positivity" in assumptions
    assert "missingness_mechanism" in nuisances
    assert fitted.nuisance.missingness is not None
    assert expression in summary.section("Assumptions").lines[0]

    # Step 6: a cross-fitted, stacked fit with boosted trees for Q, logistic g and pi, and the
    # known range of the score as q_bounds.
    method = namespace["method"]
    fit_summary = fitted.summary()
    assert dict(fit_summary.header)["Out-of-fold predictions"] == "5 folds"
    assert fitted.config.estimator_name == "stacked CV-TMLE (Levy)"
    assert method.cross_fitting == CrossFitting(n_folds=5)
    assert method.targeting.q_bounds == (0.0, 1.0)
    assert isinstance(method.models.outcome_learner, HistGradientBoostingRegressor)
    assert method.models.outcome_learner.early_stopping is True
    assert isinstance(method.models.treatment_learner, LogisticRegression)
    assert isinstance(method.models.missingness_learner, LogisticRegression)

    # The protocol step prints the record, and the fit carries its digest.  The reading names
    # the three fields this page changes; the composite death rule of the program stays.
    program = navigation_protocol()
    protocol = namespace["protocol"]
    assert changed_fields(protocol, program) == {
        "target_population",
        "outcome",
        "assumption_rationale",
    }
    assert "composite strategy" in protocol.intercurrent_event_handling[2]
    assert "between 0 and 1" in protocol.outcome
    rationale = protocol.assumption_rationale
    assert rationale[0] == program.assumption_rationale[0]
    assert rationale[2:] == program.assumption_rationale[1:]
    assert "missingness at random" in rationale[1]
    assert_protocol_recorded(NOTEBOOK, "protocol", protocol, fitted)

    # "This synthetic law draws no deaths", and the score lies in (0, 1).
    frame = namespace["frame"]
    assert set(frame.columns) == {
        "transition_score",
        "transition_navigation",
        "discharge_risk",
        "age",
        "prior_utilization",
        "responded",
    }
    observed = frame["transition_score"].dropna()
    assert float(observed.min()) > 0.0 and float(observed.max()) < 1.0
    # "About a quarter of patients never return the survey."
    assert 0.70 < float(frame["responded"].mean()) < 0.80
    assert int(frame["transition_score"].isna().sum()) == int((frame["responded"] == 0).sum())

    # Step 3: respondents have lower discharge risk (W1 -> Delta) and more prior utilization
    # (W3 -> Delta).
    by_response = namespace["by_response"]
    assert by_response.loc[1.0, "discharge_risk"] < 0.0 < by_response.loc[0.0, "discharge_risk"]
    assert by_response.loc[1.0, "prior_utilization"] > by_response.loc[0.0, "prior_utilization"]

    # The causal graph lists every arrow of the law and no other.  Each present arrow moves
    # its child by a visible amount, and each absent arrow moves it by nothing.  Columns are
    # (discharge_risk, age, prior_utilization).
    base = np.zeros((1, 3))

    def moves(function: Any, column: int) -> float:
        shifted = base.copy()
        shifted[0, column] = 1.0
        return abs(float(function(shifted)[0] - function(base)[0]))

    def outcome(w: Any) -> Any:
        return law.outcome_mean(w, 0.0, None)

    def response(w: Any) -> Any:
        return law.missingness(w, 0.0)

    assert [moves(law.propensity, column) > 0.05 for column in range(3)] == [True, True, False]
    assert moves(law.propensity, 2) == 0.0
    assert all(moves(outcome, column) > 0.02 for column in range(3))
    assert [moves(response, column) > 0.02 for column in range(3)] == [True, False, True]
    assert moves(response, 1) == 0.0
    # A -> Delta, and the effect falls as discharge risk rises (W1 modifies A -> Y).
    assert float(law.missingness(base, 1.0)[0]) > float(law.missingness(base, 0.0)[0]) + 0.05
    high_risk = np.array([[1.0, 0.0, 0.0]])

    def effect_at(w: Any) -> float:
        return float(law.outcome_mean(w, 1.0, None)[0] - law.outcome_mean(w, 0.0, None)[0])

    assert effect_at(high_risk) < effect_at(base) - 0.05
    # Step 6: "in this synthetic law the two logistic models have the true form" (their
    # logits are exactly linear in A and W), and the outcome surface is not linear.
    rng = np.random.default_rng(0)
    w = rng.normal(size=(500, 3))
    ones = np.ones(len(w))

    def linear_residual(design: Any, target: Any) -> float:
        coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
        return float(np.max(np.abs(design @ coefficients - target)))

    stacked = np.column_stack([np.r_[ones, ones], np.r_[ones * 0, ones], np.vstack([w, w])])
    means = np.r_[law.outcome_mean(w, 0.0, None), law.outcome_mean(w, 1.0, None)]
    assert linear_residual(stacked, means) > 0.05
    assert linear_residual(np.column_stack([ones, w]), logit(law.propensity(w))) < 1e-9
    response_logits = np.r_[logit(law.missingness(w, 0.0)), logit(law.missingness(w, 1.0))]
    assert linear_residual(stacked, response_logits) < 1e-9
    # "The adjustment set must contain W3", and Step 5 declares the response role.
    assert "prior_utilization" in namespace["study"].design.adjustment
    assert namespace["study"].design.missingness == "responded"

    # Step 6 and Step 7 relations on this draw.
    complete_case = namespace["complete_case"]["ate"]
    full = fitted["ate"]
    respondents = namespace["respondents"]
    assert covers(full, truth)
    assert not covers(complete_case, truth)
    assert complete_case.psi > truth
    assert len(respondents) == int(frame["responded"].sum())
    # The complete-case target for these respondents comes from the law's outcome means, and
    # the complete-case interval contains it.
    respondent_effect = namespace["respondent_effect"]
    w_resp = respondents[["discharge_risk", "age", "prior_utilization"]].to_numpy()
    assert respondent_effect == pytest.approx(
        float(np.mean(law.outcome_mean(w_resp, 1.0, None) - law.outcome_mean(w_resp, 0.0, None))),
        rel=1e-12,
    )
    assert covers(complete_case, respondent_effect)
    assert f"{namespace['distance']:.1f}" == "0.2"
    assert f"{cc_target:.3f}" == "0.173"

    # The support and nuisance reports give what the reading quotes.
    assert tuple(namespace["assessment"].attention) == ()
    support = namespace["support"]
    assert round(support.mechanisms["P(A=a,Delta=1|W)"]["min"], 4) == 0.0081
    assert round(support.mechanisms["P(Delta=1|A,W)"]["min"], 3) == 0.038
    assert support.mechanisms["P(A=a,Delta=1|W)"]["clipped_fraction"] == 0.0
    assert support.truncated["fraction"] == 0.0
    # "The largest weight on a respondent is 41.2": one over the chance of the offer they got
    # times the chance of answering under it, over respondents only.  Nonzero witness: the
    # report's clever-covariate maximum also counts non-responders and is about twice as large.
    weight = namespace["respondent_weight"]
    assert len(weight) == int(frame["responded"].sum())
    assert f"{weight.max():.1f}" == "41.2"
    assert max(support.clever_covariate_max.values()) > 1.5 * weight.max()
    # "only 3 of 3,051 respondents carry a weight above 20".
    assert namespace["heavy"] == int((weight > 20).sum()) == 3
    # The fit caps the chances at the printed bounds, and nothing reached them.
    assert round(support.bounds[0], 4) == 0.0095 and support.nuisance_bound == 0.01
    assert "nuisance fits look reasonable" in namespace["nuisance"].summary()
    assert "out-of-fold predictions" in namespace["nuisance"].summary()
    # The omitted-variable bound admits a response mechanism, so its rows run.  The benchmark
    # refits, so the default assessment defers it.
    ledger = namespace["assessment"].to_frame().set_index(["surface", "check"])
    for row in OMITTED_VARIABLE_OPERATIONS:
        status = "deferred" if row == "benchmark" else "completed"
        assert str(ledger.loc[("sensitivity", row), "status"]) == status
    # The E-value standardizes by a missing-at-random plug-in of sd(Y), which fits one more
    # outcome regression, so the default assessment defers it.
    assert str(ledger.loc[("sensitivity", "evalue"), "status"]) == "deferred"
    for row in ("missingness", "tipping_gamma"):
        assert str(ledger.loc[("sensitivity", row), "status"]) == "completed"

    # Step 10: the estimate never reaches 0 under the navigation-arm tilt, and the interval does
    # at 4.921.  Nonzero witness: just before it, the lower limit is still positive.
    assert namespace["tipping_gamma"] is None
    ci_tipping = namespace["ci_tipping_gamma"]
    assert f"{ci_tipping:.3f}" == "4.921"
    arm_gamma = {0: 0.0, 1: -1.0}
    at_tip = missingness_tilt(fitted, [ci_tipping], estimands=["ate"], arm_gamma=arm_gamma)
    before = missingness_tilt(fitted, [ci_tipping - 0.5], estimands=["ate"], arm_gamma=arm_gamma)
    assert abs(float(at_tip["ci_lower"].iloc[0])) < 1e-4
    assert float(before["ci_lower"].iloc[0]) > 0.0
    # The mean shift divides each move by the mean non-response probability under navigation.
    # Nonzero witness: navigation raises response, so the other arm's column would differ.
    missingness = fitted.nuisance.bounded_missingness(0.01)
    navigation_column = fitted.nuisance.arms.index(1)
    nonresponse = namespace["nonresponse"]
    assert nonresponse == pytest.approx(float(np.mean(1.0 - missingness[:, navigation_column])))
    assert nonresponse < float(np.mean(1.0 - missingness[:, 1 - navigation_column])) - 0.05
    # Gamma = 30 has reached the floor: a larger tilt moves the estimate by nothing visible.
    floor = missingness_tilt(fitted, [30.0, 60.0], estimands=["ate"], arm_gamma=arm_gamma)
    assert abs(float(floor["psi"].iloc[0] - floor["psi"].iloc[1])) < 1e-6
    assert float(floor["psi"].iloc[0]) > 0.0
    assert (
        f"{namespace['mar_mean']:.3f}",
        f"{namespace['ci_mean_shift']:.3f}",
    ) == ("0.727", "0.688")
    assert namespace["ci_mean_shift"] < namespace["mar_mean"]

    # The curve's gamma=0 row is the MAR estimate, and its interval reuses the MAR standard error.
    curve = namespace["missingness_curve"]
    mar = curve[curve["gamma"] == 0.0]
    assert float(mar["psi"].iloc[0]) == pytest.approx(full.psi, rel=1e-9)
    assert (curve["std_err"] == full.std_error).all()
    assert curve["psi"].is_monotonic_decreasing
    assert (curve["psi"] > 0.0).all()
    # The printed mean drop is the move of the estimate over the mean non-response probability,
    # and it approaches the missing-at-random mean as gamma grows toward the floor.
    printed = namespace["curve"]
    expected_drop = (full.psi - curve["psi"].to_numpy()) / nonresponse
    assert np.allclose(printed["mean drop"].to_numpy(), expected_drop)
    assert float(printed["mean drop"].iloc[-1]) < namespace["mar_mean"]


def check_refits(namespace: dict[str, Any]) -> None:
    """The refit cells: the refutation, the generated outcomes, and the E-value and bounds."""
    _check_refutation(namespace)
    _check_generated_outcomes(namespace)
    _check_evalue_and_bounds(namespace)


def _check_refutation(namespace: dict[str, Any]) -> None:
    """Step 9: the three refits are descriptive, and each check the page skips refuses."""
    fitted = namespace["full"]
    refutation = namespace["refutation"]
    original = fitted["ate"].psi
    tests = {test.name: test for test in refutation.tests}
    assert set(tests) == {"placebo", "subset", "random_common_cause"}
    for test in tests.values():
        # "No check prints a verdict": no registry row covers a boosted fit with missing outcomes.
        assert test.passed is None and test.rung == "L0"
        assert len(test.values) == 20 and test.original == pytest.approx(original)
    assert refutation.passed is None
    report = " ".join(refutation.summary(detail="full").section(vocabulary.SECTION_COMPARED).lines)
    # The full summary names the nearest registered row, a TMLE row with GLM nuisances, and the
    # learner on which this fit differs.  Nonzero witness: the first registered row is a
    # longitudinal one.
    assert "The nearest fits studied: method TMLE," in report and "LTMLE" not in report
    assert "Where this fit differs, it has a learner outside those studied" in report
    # "The verdict column reads descriptive".
    assert set(refutation.to_frame()["verdict"]) == {"descriptive"}
    # The placebo reading: "+0.0009203 ... spread 0.006121 ... t +0.6724 (p = 0.509) ... 0.9123".
    placebo = tests["placebo"]
    values = np.asarray(placebo.values)
    assert round(float(values.mean()), 7) == 0.0009203
    assert round(float(values.std(ddof=1)), 6) == 0.006121
    assert round(float(placebo.statistic), 4) == 0.6724
    assert round(float(placebo.statistic_pvalue), 3) == 0.509
    assert round(float(np.mean(placebo.standard_errors) / values.std(ddof=1)), 4) == 0.9123
    # A placebo that kept the offers would return the original estimate, far from the null.
    assert abs(values.mean()) < 0.1 * original
    # The subset reading: "0.0034151 against 0.0034171 ... ratio 0.99941 (p = 0.917)".
    subset = tests["subset"]
    scatter = float(np.std(subset.values, ddof=1))
    n = len(namespace["frame"])
    reference = float(fitted["ate"].std_error * np.sqrt((n / 2800 - 1) * n / (n - 1)))
    assert round(scatter, 7) == 0.0034151 and round(reference, 7) == 0.0034171
    assert round(scatter / reference, 5) == 0.99941
    assert round(float(subset.statistic_pvalue), 3) == 0.917
    # The random-common-cause reading: "+0.04526 standard errors, spread 0.08908".
    shift = (np.asarray(tests["random_common_cause"].values) - original) / fitted["ate"].std_error
    assert round(float(shift.mean()), 5) == 0.04526
    assert round(float(shift.std(ddof=1)), 5) == 0.08908
    assert 0.0 < abs(shift.mean()) < 1.0

    # The negative control the page skips: no recorded pre-offer score, and a control equal to an
    # adjustment column is refused.
    refusal = refute_refusal(
        fitted,
        estimand="ate",
        tests=("negative_control_outcome",),
        negative_control_outcome=namespace["frame"]["age"].to_numpy(),
        negative_control_kind="pre_treatment",
    )
    assert refusal is not None and "equals the adjustment column 'age'" in refusal.message()
    # The replay the page skips: the response reads the offer, so a cell that moves the offer is
    # refused and names F12.  Nonzero witness: the law's response differs by arm.
    law = missing_outcome_bounded_dgp()
    base = np.zeros((1, 3))
    assert float(law.missingness(base, 1.0)[0]) > float(law.missingness(base, 0.0)[0]) + 0.05
    grid = ConfounderStrengthGrid(treatment=(0.0, 0.1), outcome=(0.0, 0.1))
    with pytest.raises(CapabilityError, match="observation_reads") as excinfo:
        simulated_confounding(fitted, grid=grid)
    assert "F12" in str(excinfo.value)


def _check_generated_outcomes(namespace: dict[str, Any]) -> None:
    """Step 9: the generated outcomes run at level L1 on the bounded process."""
    fitted = namespace["full"]
    independent, adjusted = namespace["independent"], namespace["adjusted"]
    assert isinstance(independent, BoundedAdjustmentOutcome)
    assert (independent.effect, independent.adjustment_scale) == (0.0, 0.0)
    assert adjusted.effect != 0.0 and adjusted.adjustment_scale != 0.0
    probe = (EXAMPLES.parents[1] / _REFUTE_PROBE).read_text(encoding="utf-8")
    for literal in _REFUTE_LOG_DECIMALS:
        assert literal in probe, literal
    assert probe.count("400 draws") >= 3 and probe.count("L2") >= 3
    # Only simulated_outcome leaves its band; its mean estimate lies 0.00056 below the truth.
    assert (
        probe.count("bias") >= 3
        and probe.count("inside [-0.2, 0.2]") == 2
        and "outside [-0.2, 0.2]" in probe
    )
    assert round(0.10287 - 0.10231, 5) == 0.00056
    tests = namespace["generated_tests"]
    # "with 10 every line prints level L1, the numbers without a pass or fail".
    assert set(tests) == {"dummy_outcome", "simulated_outcome", "plasmode"}
    for check in tests.values():
        assert check.n_replicates == 10 and len(check.values) == 10
        assert check.rung == "L1" and check.passed is None
        bias = check.readings[0]
        assert bias.name == "bias" and bias.upper - bias.lower > 0.2 - (-0.2)
    # Truth: 0 for the dummy and one shared positive value for the other two, which differs from
    # the recorded estimate.  The held-design simulated outcome withholds the ratio and coverage.
    assert tests["dummy_outcome"].declared_effect == 0.0
    shared = tests["simulated_outcome"].declared_effect
    assert shared == tests["plasmode"].declared_effect and 0.0 < shared < fitted["ate"].psi - 0.03
    assert [reading.name for reading in tests["simulated_outcome"].readings] == ["bias"]
    assert len(tests["plasmode"].readings) == 3
    # A bare Gaussian declaration refuses the fit that declares q_bounds, as the page says.
    refusal = refute_refusal(
        fitted,
        estimand="ate",
        tests=("simulated_outcome",),
        simulated_outcome=GaussianAdjustmentOutcome(),
    )
    assert refusal is not None and "q_bounds" in refusal.message()


def _check_evalue_and_bounds(namespace: dict[str, Any]) -> None:
    """Step 10: the E-value and the omitted-variable bound, read on the fit."""
    fitted = namespace["full"]
    evalue = namespace["evalue"]
    assert evalue.approximate and evalue.limit_descriptive
    assert round(evalue.risk_ratio, 3) == 1.706
    assert tuple(round(x, 3) for x in evalue.risk_ratio_ci) == (1.644, 1.77)
    assert (round(evalue.point, 4), round(evalue.limit, 4)) == (2.8039, 2.6735)
    # The E-value is the closed form of VanderWeele and Ding at the risk ratio.
    rr = evalue.risk_ratio
    assert evalue.point == pytest.approx(rr + np.sqrt(rr * (rr - 1.0)))
    assert "missing-at-random plug-in of the population sd(Y) = 0.2504" in str(evalue.summary())
    rv = namespace["rv"]
    assert (round(rv["rv"], 4), round(rv["rva"], 4)) == (0.3881, 0.3612)
    assert rv["limit_licensed"] is False
    assert "differs on outcome family 'gaussian'" in rv["limit_sentence"]
    # The benchmarks: the table of the reading, recomputed from the fit.
    expected = {
        "discharge_risk": (1.0, 0.361, (-0.0759, 0.3716), (-0.0947, 0.3906)),
        "age": (0.872, 0.032, (0.0974, 0.1982), (0.0881, 0.2077)),
        "prior_utilization": (0.128, 0.0, (0.1478, 0.1478), (0.1392, 0.1564)),
    }
    for covariate, (cf_y, cf_d, bound_limits, ci_limits) in expected.items():
        record = fitted.sensitivity.benchmark(covariates=(covariate,))
        bound = fitted.sensitivity.omitted_confounding(benchmark=record)
        assert (round(record.cf_y, 3), round(record.cf_d, 3)) == (cf_y, cf_d)
        assert (round(bound.lower, 4), round(bound.upper, 4)) == bound_limits
        assert (round(bound.ci_lower, 4), round(bound.ci_upper, 4)) == ci_limits
        assert bound.limit_descriptive
    # "Dropping discharge_risk more than doubles the residual variance": a gain of 1.272, so cf_y
    # clips at 1 and the bound includes 0.  Dropping prior_utilization raises nu^2, so cf_d clips.
    risk = fitted.sensitivity.benchmark(covariates=("discharge_risk",))
    assert risk.sigma2_short > 2.0 * risk.sigma2_long and round(risk.gain_y, 3) == 1.272
    assert risk.cf_y == 1.0
    assert fitted.sensitivity.omitted_confounding(benchmark=risk).lower < 0.0
    utilization = fitted.sensitivity.benchmark(covariates=("prior_utilization",))
    assert utilization.nu2_short > utilization.nu2_long
    assert utilization.gain_d < 0.0 and utilization.cf_d == 0.0
    # The bound is zero-width at cf_d = 0, and the estimate sits at its centre.  A mutation that
    # added the response gain would move it.
    flat = fitted.sensitivity.omitted_confounding(benchmark=utilization)
    assert flat.lower == pytest.approx(fitted["ate"].psi, abs=1e-9)
    # The bisection: at cf_y = 1 the lower limit reaches zero at the printed weight-side
    # strength, about half that of discharge_risk and more than five times that of age.
    tipping = namespace["tipping_cf_d"]
    assert f"{tipping:.2f}" == "0.17"
    assert abs(fitted.sensitivity.omitted_confounding(cf_y=1.0, cf_d=tipping).ci_lower) < 1e-6
    assert fitted.sensitivity.omitted_confounding(cf_y=1.0, cf_d=tipping - 0.01).ci_lower > 0.0
    strengths = {name: record.cf_d for name, (record, _) in namespace["benchmarks"].items()}
    assert strengths["prior_utilization"] < strengths["age"] < tipping < strengths["discharge_risk"]
    assert 0.4 < tipping / strengths["discharge_risk"] < 0.6
    assert 5.0 < tipping / strengths["age"] < 6.0
