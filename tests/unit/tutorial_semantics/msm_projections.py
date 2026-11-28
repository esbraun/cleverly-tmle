"""The reviewed semantic callback for ``docs/examples/msm-projections``.

``tests/unit/test_documentation_runtime.py`` runs the tutorial at its documented size and
passes the resulting namespace to :func:`check`.  The tutorial is a notebook, so its narrated
decimals are also compared against its stored outputs, and every decimal it writes is printed.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from sklearn.dummy import DummyRegressor

from cleverly import MSMProjection
from cleverly.datasets import navigation_protocol
from cleverly.datasets.synthetic import multi_arm_dgp
from cleverly.msm import MSM
from cleverly.sensitivity.omitted_variable import OMITTED_VARIABLE_OPERATIONS
from cleverly.validation import refute
from tests.documents import ROOT
from tests.unit.tutorial_semantics import (
    EXAMPLES,
    assert_protocol_recorded,
    changed_fields,
    covers,
    figure_title,
    stored_output,
)

#: The refit cells the rule derives from this notebook today, in notebook order.  A change here is
#: a change in what the fast gate skips, so it shows in the diff of the pull request that causes it.
REFIT_CELLS = (
    "sensitivity",
    "refute",
    "generated",
)


NOTEBOOK = EXAMPLES / "msm-projections.ipynb"

_PROBE = "reviews/notebook-review/probes/msm-projections-final"

#: Decimals the readings write that no stored output prints, each with its source.
UNPRINTED_DECIMALS = {
    "0.01": f"the true-g threshold, share below it 0.0112 for medium or high: "
    f"{_PROBE}/shortfall.log",
    "0.006": f"the simulated_outcome bias at 400 refits: {_PROBE}/generated_r400.log",
}


#: Decimals a reading quotes from a code cell outside its own section, by reading cell, then
#: literal, then the code cell that prints it (tests.unit.tutorial_semantics.narration_scopes).
NARRATION_SOURCES: dict[str, dict[str, str]] = {
    "trend-reading": {"0.2657": "failure-mode"},
    "assessment-heading": {"0.9886": "trend"},
}


def _log_value(text: str, pattern: str) -> str:
    """Return the first group of ``pattern`` in a committed probe log."""
    match = re.search(pattern, text)
    assert match, f"the probe log no longer contains {pattern!r}"
    return match.group(1)


def _assert_unprinted_sources() -> None:
    """Each decimal in UNPRINTED_DECIMALS equals the committed file that holds it."""
    sweep = (ROOT / _PROBE / "sweep.log").read_text(encoding="utf-8")
    shortfall = (ROOT / _PROBE / "shortfall.log").read_text(encoding="utf-8")
    # "covered the true slope about 95% of the time (3,782 of 4,000)", on a recorded commit.
    assert re.search(r"^run under: cleverly \S+, commit [0-9a-f]{12} \(clean\)", sweep)
    assert re.search(r"slope_cover: 3782/4000 = 0\.9455 ", sweep)
    # The 400-refit run of the generated-outcome checks: 24 seconds, every test level L2 and
    # passed, and the simulated_outcome bias 0.006.
    r400 = (ROOT / _PROBE / "generated_r400.log").read_text(encoding="utf-8")
    assert re.search(r"^run under: cleverly \S+, commit [0-9a-f]{12} \(clean\)", r400)
    assert _log_value(r400, r"at 400 refits each: ([0-9]+) s") == "24"
    for name in ("dummy_outcome", "simulated_outcome", "plasmode"):
        assert f"{name}: 400 draws, level L2, verdict passed" in r400
    simulated = r400.split("simulated_outcome:")[1]
    assert _log_value(simulated, r"bias (-?[0-9.]+) in") == "0.006"
    # "About one patient in a hundred has a true probability below 0.01."
    share = float(_log_value(shortfall, r"g\(high\) below 0\.01: ([0-9.]+)"))
    assert 0.005 < share < 0.02


def _weighted_projection(design: np.ndarray, weights: np.ndarray, means: np.ndarray) -> np.ndarray:
    """Solve the weighted normal equations of the three-point projection."""
    return np.linalg.solve(design.T @ (weights[:, None] * design), design.T @ (weights * means))


def check(namespace: dict[str, Any]) -> None:
    """The projection tutorial's protocol, codings, failure mode, weights, and control hold."""
    _assert_unprinted_sources()
    effect = namespace["arm_means_effect"]
    fitted = namespace["arm_result"]
    study, method = namespace["study"], namespace["method"]
    levels = tuple(study.data.treatment_levels)
    positivity = " ".join(
        assumption
        for assumption in effect.identification.assumptions
        if "positivity" in assumption.lower()
    )
    summary = str(effect.summary(detail="full"))

    assert len(levels) == 3, "the tutorial no longer witnesses multi-arm identification"
    assert fitted.nuisance.propensity.values.shape[1] == len(levels)
    for level in levels:
        assert str(level) in positivity
    # The models a fit needs are a program's record; the summary prints no internal name.
    assert "nuisance" not in summary.lower()
    # The full summary states the multi-arm positivity statement verbatim.
    lines = effect.summary(detail="full").section("Assumptions").lines
    (stated,) = [line for line in lines if line.startswith("- positivity")]
    for level in levels:
        assert repr(level) in stated or str(level) in stated
    assert "P(A = 1 | W)" not in summary
    assert "both counterfactual means" not in summary

    # The protocol step prints the record, and both fits carry its digest.
    # The reading names the fields this page changes in the program protocol.
    assert changed_fields(namespace["protocol"], navigation_protocol()) == {
        "outcome",
        "time_zero",
        "treatment_strategies",
        "treatment_versions",
        "intercurrent_event_handling",
        "assumption_rationale",
    }
    # "cleverly needs a known score range" for out-of-fold predictions: the page's own reason
    # for fitting in sample, and the fit reports it. Death uses the hypothetical strategy.
    protocol = namespace["protocol"]
    assert "no documented range" in protocol.outcome
    assert "standardized" not in protocol.outcome
    assert "hypothetical strategy" in protocol.intercurrent_event_handling[2]
    assert not method.cross_fitting.enabled
    assert dict(fitted.summary().header)["Out-of-fold predictions"] == "no"
    assert_protocol_recorded(
        NOTEBOOK, "protocol", namespace["protocol"], fitted, namespace["trend_result"]
    )

    population = namespace["population"]
    assert population[0] < population[1] < population[2]
    assert population[2] - population[1] > population[1] - population[0]
    # "The observed means put medium above high", while the true means reverse that order.
    # Both confounders push the same way: medium has the higher risk and the lower age.
    observed = namespace["observed"]
    assert observed.loc["medium", "transition_score"] > observed.loc["high", "transition_score"]
    assert observed.loc["medium", "discharge_risk"] > observed.loc["high", "discharge_risk"]
    assert observed.loc["high", "age"] > observed.loc["medium", "age"]
    # "Each interval contains its true mean on this draw."
    for arm, target in zip(namespace["ARMS"], population, strict=True):
        assert covers(fitted[f"ey[{arm}]"], target)
    # "The labels have no order of their own. Sorted as text, they are high, low, and medium."
    assert sorted(namespace["ARMS"]) == ["high", "low", "medium"]

    # "The identification assumptions line lists the four causal assumptions and adds two."
    trend_assumptions = namespace["trend_effect"].identification.assumptions
    assert len(trend_assumptions) == len(effect.identification.assumptions) + 2

    projection = namespace["projection"]
    trend = namespace["trend_result"]
    names = ("msm[(intercept)]", "msm[assigned contacts]")
    for name, target in zip(names, projection, strict=True):
        assert covers(trend[name], target)
    # Under the uniform weight the slope is the fixed contrast of the three means, exactly.
    assert namespace["contrast_slope"] == pytest.approx(projection[1], rel=1e-12, abs=1e-12)
    # The share weight moves the population slope: the weight is part of the estimand.
    design = namespace["design"]
    shares = namespace["shares"]
    np.testing.assert_allclose(
        namespace["share_projection"],
        _weighted_projection(design, shares, population),
        rtol=1e-12,
    )
    assert abs(namespace["share_projection"][1] - projection[1]) > 0.005

    # A uniform weight is a term that vanishes, so witness the library's weight path directly.
    # (a) ``weights=None`` is the explicit all-ones weight, not the arm shares.
    terms = namespace["trend"].terms
    contacts_design = namespace["contacts_design"]
    explicit_uniform = study.identify(
        MSMProjection(
            MSM(
                design=contacts_design,
                terms=terms,
                weights=lambda arm, data: np.ones(len(data)),
                weights_kind="known",
                design_kind="known",
            )
        )
    ).estimate(method=method)
    for name in names:
        assert explicit_uniform[name].psi == pytest.approx(trend[name].psi, rel=1e-12, abs=1e-12)
    # (b) The share-weight section's fixed 1:10:1 weight moves the fitted slope to its own projection.
    # This fit's correctly specified Q witnesses the weighted Gram matrix.
    fixed = namespace["FIXED_WEIGHT"]
    assert fixed == {"low": 1.0, "medium": 10.0, "high": 1.0}
    weighted = namespace["weighted_result"]
    fixed_target = _weighted_projection(
        design, np.array([fixed[arm] for arm in namespace["ARMS"]]), population
    )
    np.testing.assert_allclose(namespace["fixed_projection"], fixed_target, rtol=1e-12)
    assert f"{fixed_target[1]:.4f}" == "0.2400"
    # "On this draw, each interval contains its own population slope and not the other."
    uniform_slope = trend["msm[assigned contacts]"]
    weighted_slope = weighted["msm[assigned contacts]"]
    assert covers(weighted_slope, fixed_target[1])
    assert not covers(uniform_slope, fixed_target[1])
    assert not covers(weighted_slope, projection[1])
    # The share-weight illustration changes the target. One slope's marginal SE does not
    # measure uncertainty in the difference between two fitted slopes.
    shift = abs(namespace["share_projection"][1] - projection[1])
    assert f"{shift:.4f}" == "0.0065"
    # (c) Deliberately misspecify Q while retaining the correctly specified multinomial g.
    # Targeting must now do the adjustment. Dropping h from its clever covariate moves the
    # slope back across the uniform target and outside the fixed-weight target's interval.
    misspecified_q = replace(
        method,
        models=replace(method.models, outcome_learner=DummyRegressor()),
    )
    weighted_misspecified_q = study.identify(
        MSMProjection(
            MSM(
                design=contacts_design,
                terms=terms,
                weights=lambda arm, data: np.full(len(data), fixed[arm]),
                weights_kind="known",
                design_kind="known",
            )
        )
    ).estimate(method=misspecified_q)
    weighted_slope = weighted_misspecified_q["msm[assigned contacts]"]
    assert covers(weighted_slope, fixed_target[1])
    # Its skew-widened interval (0.196 to 0.267 on this draw) reaches just past the uniform
    # target, 0.2657, so the witness reads the estimate: it sits nearer its own target.
    assert abs(weighted_slope.psi - fixed_target[1]) < abs(weighted_slope.psi - projection[1]) / 3

    saturated = namespace["saturated_result"]
    mappings = {
        "msm[(intercept)]": namespace["arm_result"]["ey[low]"],
        "msm[medium vs low]": namespace["arm_contrasts"]["ate[medium vs low]"],
        "msm[high vs low]": namespace["arm_contrasts"]["ate[high vs low]"],
    }
    for name, counterpart in mappings.items():
        coefficient = saturated[name]
        assert coefficient.psi == pytest.approx(counterpart.psi, rel=1e-12, abs=1e-12)
        np.testing.assert_allclose(
            coefficient.influence_curve,
            counterpart.influence_curve,
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(coefficient.ci, counterpart.ci, rtol=1e-12, atol=1e-12)

    line = design @ projection
    residual = line - population
    assert abs(residual[1]) > 0.15, "the contact mapping no longer witnesses misspecification"
    assert 0.17 < abs(residual[1]) < 0.21, "the prose says the line misses medium by about 0.19"
    assert (
        figure_title(namespace["trend_fig"])
        == "How does the average score change with assigned contacts?"
    )
    line_minus_mean = namespace["in_fits"]["line minus mean"]
    assert line_minus_mean.iloc[0] > 0 > line_minus_mean.iloc[1]
    # The miss at medium is a fixed contrast of the three arm means, read through ``contrast``.
    # "The interval excludes zero" and "the interval contains the population miss".
    medium_misfit = namespace["medium_misfit"]
    means = np.array([fitted[f"ey[{arm}]"].psi for arm in namespace["ARMS"]])
    hat = design @ np.linalg.pinv(design)
    assert medium_misfit.psi == pytest.approx((hat @ means - means)[1], rel=1e-10, abs=1e-12)
    assert medium_misfit.ci[1] < 0.0
    assert covers(medium_misfit, residual[1])
    # The Details: the contrast estimate (-0.2084) differs from the fitted line's miss (-0.2092).
    step7_miss = namespace["estimated_line"][1] - means[1]
    assert f"{abs(medium_misfit.psi - step7_miss):.4f}" == "0.0008"

    # Code the schedules: two known codings of the same means answer two questions. Each interval
    # contains its own population slope on this draw, and not the other coding's.
    step_values = np.array([namespace["CADENCE_STEP"][arm] for arm in namespace["ARMS"]])
    np.testing.assert_array_equal(step_values, [0.0, 1.0, 2.0])
    step_design = np.column_stack([np.ones(3), step_values])
    step_target = np.linalg.solve(step_design.T @ step_design, step_design.T @ population)
    assert f"{step_target[1]:.4f}" == "0.7200"
    step_slope = namespace["step_result"]["msm[cadence step]"]
    assert covers(step_slope, step_target[1])
    assert covers(trend["msm[assigned contacts]"], projection[1])
    assert not covers(step_slope, projection[1])
    # The per-step slope is half the high-vs-low difference under the uniform weight.
    high_vs_low = namespace["saturated_result"]["msm[high vs low]"].psi
    assert f"{high_vs_low / 2:.3f}" == "0.734"
    assert abs(step_slope.psi - high_vs_low / 2) < 5e-4
    assert step_target[1] == pytest.approx(population[2] / 2 - population[0] / 2)
    # "Neither line passes through the three means."
    assert np.max(np.abs(step_design @ step_target - population)) > 0.05

    # The assessment names the support warning, and the slope barely moves along the curve
    # even where the largest bound clips most of the rows.
    assessment = namespace["assessment"]
    attention = {item.name for item in assessment.attention}
    assert attention == {"support"}
    assert namespace["scores"].passed
    # The fit reports two coefficients, so each omitted-variable row waits on an estimand
    # (X82 part G2c), which is why the sensitivity cell names the slope.
    ledger = assessment.to_frame().set_index(["surface", "check"])["status"]
    for operation in OMITTED_VARIABLE_OPERATIONS:
        assert str(ledger.loc[("sensitivity", operation)]) == "deferred"
    assert str(ledger.loc[("validation", "support")]) == "warning"
    assert str(ledger.loc[("diagnostics", "refute")]) == "deferred"
    # A stream can split across stored outputs, so compare with whitespace collapsed.
    printed = " ".join(stored_output(NOTEBOOK, "assessment").split())
    assert "checks that need attention: ('support',)" in printed
    support = namespace["support"]
    # "0.0120 of the patients hit the cap, and cleverly flags any share above 1%."
    assert support.truncated["fraction"] > 0.01
    assert support.severity == "strain"
    ratios = {arm: row["ratio"] for arm, row in support.effective_sample_size.items()}
    assert min(ratios, key=ratios.get) == "high"
    # Not narrated: the largest correction-step weight is the high contact count over a fitted g[high].
    lower, upper = support.bounds
    treatment = namespace["frame"]["cadence"].to_numpy()
    propensity = np.clip(np.asarray(trend.nuisance.propensity.values), lower, upper)
    contact_over_g = max(
        float(
            np.max(namespace["CONTACTS_30D"][arm] / propensity[treatment == arm, levels.index(arm)])
        )
        for arm in namespace["ARMS"]
    )
    assert support.clever_covariate_max["msm"] == pytest.approx(contact_over_g, rel=1e-9)
    high_column = propensity[treatment == "high", levels.index("high")]
    assert support.clever_covariate_max["msm"] == pytest.approx(6.0 / high_column.min(), rel=1e-9)
    slope_curve = namespace["slope_curve"]
    assert f"{slope_curve['psi'].min():.4f}" == "0.2654"
    assert f"{slope_curve['psi'].max():.4f}" == "0.2700"
    assert slope_curve["truncated_fraction"].iloc[-1] > 0.5
    assert f"{slope_curve['delta_from_fitted'].abs().max():.4f}" == "0.0033"
    assert slope_curve["delta_from_fitted"].abs().max() < 0.005

    # The slope has no E-value: the E-value reads an arm contrast.
    trend = namespace["trend_result"]
    with pytest.raises(Exception, match="arm contrast"):
        trend.sensitivity.evalue(estimand="msm[assigned contacts]")
    assert round(trend["msm[assigned contacts]"].std_error, 4) == 0.0086
    _check_go_deeper_lessons(namespace)


def _check_go_deeper_lessons(namespace: dict[str, Any]) -> None:
    """The arm-contrast bounds, the model checks, and the three lessons after the trend."""
    study, method, frame = namespace["study"], namespace["method"], namespace["frame"]
    contacts = namespace["CONTACTS_30D"]

    # Check the fit: the concentration line reads the slope's equation, whose weight is the
    # contact count over the chance of the schedule ("these patients weigh most").
    load = namespace["support"].group_score_load["msm"]
    assert load["equation"] == "h_msm1"
    assert f"{load['total_ratio']:.3f}" == "0.344"
    models = namespace["models"]
    aucs = {arm: models.loc[f"propensity[{arm}]", "auc"] for arm in namespace["ARMS"]}
    assert aucs["low"] < min(aucs["medium"], aucs["high"])
    assert 0.6 < models.loc["outcome", "r2"] < 0.7

    # The contrasts' bound: in this sample the first step is the most fragile result.
    arm_contrasts = namespace["arm_contrasts"]
    slope_rv = namespace["trend_result"].sensitivity.robustness_value(estimand=namespace["SLOPE"])
    values = {
        key: arm_contrasts.sensitivity.robustness_value(estimand=key)["rv"]
        for key in ("ate[medium vs low]", "ate[high vs low]")
    }
    assert values["ate[medium vs low]"] < slope_rv["rv"] < values["ate[high vs low]"]
    assert f"{values['ate[medium vs low]']:.3f}" == "0.204"
    # Witness: equal strengths just below the value leave the point bound above zero, and just
    # above it cross zero.
    rv = values["ate[medium vs low]"]
    bound = arm_contrasts.sensitivity.omitted_confounding
    assert bound(cf_y=rv - 0.01, cf_d=rv - 0.01, estimand="ate[medium vs low]").lower > 0
    assert bound(cf_y=rv + 0.01, cf_d=rv + 0.01, estimand="ate[medium vs low]").lower < 0

    # Effect modifier. The truth (risk 1, product 0) follows from the law: the gaussian outcome
    # mean adds W1 with coefficient 1 and has no arm-by-covariate term, and the covariates are
    # independent standard normals. A large draw of the law checks that independently.
    dgp = multi_arm_dgp()
    covariates = np.random.default_rng(20261126).normal(size=(400_000, 3))
    rows, targets = [], []
    for arm in namespace["ARMS"]:
        index, count = dgp.labels.index(arm), contacts[arm]
        rows.append(
            np.column_stack(
                [
                    np.ones(len(covariates)),
                    np.full(len(covariates), count),
                    covariates[:, 0],
                    count * covariates[:, 0],
                ]
            )
        )
        targets.append(dgp.outcome_mean(covariates, index))
    law_coef = np.linalg.lstsq(np.vstack(rows), np.concatenate(targets), rcond=None)[0]
    np.testing.assert_allclose(law_coef[2:], [1.0, 0.0], atol=0.01)
    modifier = namespace["modifier_result"]
    assert covers(modifier["msm[contacts x risk]"], 0.0)
    assert covers(modifier["msm[discharge risk]"], 1.0)
    # Fit-level witness: a score with a true product term of 0.1 is recovered and told from 0.
    shifted = frame.copy()
    arms = shifted["cadence"].map(contacts).to_numpy()
    shifted["transition_score"] = (
        shifted["transition_score"] + 0.1 * arms * shifted["discharge_risk"]
    )
    witness = type(study)(shifted, design=study.design, protocol=namespace["protocol"])
    product = witness.identify(MSMProjection(namespace["modifier_trend"])).estimate(method=method)[
        "msm[contacts x risk]"
    ]
    assert covers(product, 0.1) and not covers(product, 0.0)

    # Yes-or-no outcome: the printed truth solves the quasi-binomial score equation of the logit
    # projection, and differs from least squares on the probability scale.
    design = namespace["design"]
    chances = namespace["binary_population"]
    line = namespace["odds_line"]
    fitted = 1.0 / (1.0 + np.exp(-(design @ line)))
    np.testing.assert_allclose(design.T @ (chances - fitted), 0.0, atol=1e-10)
    from scipy.optimize import least_squares

    probability_scale = least_squares(
        lambda b: chances - 1.0 / (1.0 + np.exp(-(design @ b))), x0=line
    ).x
    assert np.max(np.abs(probability_scale - line)) > 1e-6
    assert f"{np.exp(line[1]):.3f}" == "1.180"
    assert covers(namespace["odds_result"]["msm[assigned contacts]"], line[1])
    assert changed_fields(namespace["binary_protocol"], namespace["protocol"]) == {
        "outcome",
        "intercurrent_event_handling",
        "assumption_rationale",
    }
    assert "hypothetical strategy" in namespace["binary_protocol"].intercurrent_event_handling[2]
    binary_protocol = namespace["binary_protocol"]
    assert "score" not in " ".join(binary_protocol.assumption_rationale)
    assert "score" not in binary_protocol.intercurrent_event_handling[0]
    assert_protocol_recorded(
        NOTEBOOK, "logit", namespace["binary_protocol"], namespace["odds_result"]
    )

    # "No study checks the intervals of a three-schedule fit like this page's ... So
    # the schedule averages, their differences from `low`, the trend ... all print
    # `experimental`."
    for fit in (namespace["arm_result"], namespace["arm_contrasts"], namespace["trend_result"]):
        assert set(fit.to_frame()["support"]) == {"experimental"}

    # Treatment plans: the true line is the least-squares line through the three plan means,
    # and both coefficients carry the supported label.
    plan_means = namespace["plan_means"]
    plan_design = namespace["plan_design"]
    np.testing.assert_allclose(
        namespace["true_plans_line"],
        np.linalg.solve(plan_design.T @ plan_design, plan_design.T @ plan_means),
        rtol=1e-12,
    )
    table = namespace["plans_table"]
    assert set(table["support"]) == {"supported"}
    # "because the cross-fitted longitudinal projection study covers this fit"
    for name in table.index:
        support = namespace["plans_result"][name].support
        assert support.evidence == ("cross-fitted-longitudinal-msm",)
    for name, target in zip(table.index, namespace["true_plans_line"], strict=True):
        assert covers(namespace["plans_result"][name], target)


def check_refits(namespace: dict[str, Any]) -> None:
    """The refutation, the omitted-variable bounds, and the generated outcomes."""
    assert namespace["SLOPE"] == "msm[assigned contacts]"
    _check_refutation(namespace)
    _check_sensitivity(namespace)


def _check_sensitivity(namespace: dict[str, Any]) -> None:
    """The robustness values, the benchmark of every recorded covariate, and the tipping point."""
    # The bound reads the slope of the trend fit.
    robustness = namespace["robustness"]
    slope = namespace["SLOPE"]
    # "a factor weaker than 0.394 on both sides cannot pull the slope to zero (0.366 for its
    # lower 95% limit)". Witness: at equal strengths just above rv the point bound crosses zero.
    assert 0.0 < robustness["rva"] < robustness["rv"] < 1.0
    assert (f"{robustness['rv']:.3f}", f"{robustness['rva']:.3f}") == ("0.394", "0.366")
    sensitivity = namespace["trend_result"].sensitivity
    rv = robustness["rv"]
    assert sensitivity.omitted_confounding(cf_y=rv - 0.01, cf_d=rv - 0.01, estimand=slope).lower > 0
    assert sensitivity.omitted_confounding(cf_y=rv + 0.01, cf_d=rv + 0.01, estimand=slope).lower < 0
    # "False means no simulation study checks the slope's limits for this kind of fit."
    assert robustness["limit_licensed"] is False
    assert "estimand kind" in robustness["limit_sentence"]
    # Every recorded covariate is benchmarked. "Discharge risk and age each reach 0.57 on the
    # weight side. With a rho of size 1, a missing confounder as strong as either would leave a
    # range of slopes that reaches below zero." "Support need reaches 0 on the weight side."
    benchmarks = namespace["benchmarks"]
    adjustment = namespace["study"].design.adjustment
    assert tuple(benchmarks) == tuple(adjustment)
    for name in ("discharge_risk", "age"):
        record, worst, aligned = benchmarks[name]
        assert f"{record.cf_d:.2f}" == "0.57"
        assert worst.lower < 0.0 < worst.upper
        # "Lined up as discharge risk and age are (rho -0.34 and -0.44), the range stays above
        # zero." Witness: the same benchmark at rho = 1 crosses zero (checked above).
        assert abs(record.rho) < 0.9
        assert 0.0 < aligned.lower < aligned.upper
    record, worst, aligned = benchmarks["baseline_support_need"]
    assert record.cf_d == pytest.approx(0.0, abs=1e-3) and worst.ci_lower > 0.0
    # "The benchmark caps its cf_y at 1.000" for discharge_risk, whose gain exceeds 1, and
    # "Dropping discharge risk moves the slope from 0.269 to 0.148".
    record = benchmarks["discharge_risk"][0]
    assert record.cf_y == 1.0 and record.gain_y > 1.0
    assert f"{record.psi_short:.3f}" == "0.148"
    # The bisection finds the weight-side strength at which the lower limit reaches zero when
    # cf_y = 1, and it lies far below the recorded confounders' strengths.
    tipping = namespace["tipping_cf_d"]
    at = sensitivity.omitted_confounding(cf_y=1.0, cf_d=tipping, estimand=slope)
    assert abs(at.ci_lower) < 1e-6
    assert (
        sensitivity.omitted_confounding(cf_y=1.0, cf_d=tipping - 0.01, estimand=slope).ci_lower > 0
    )
    assert f"{tipping:.2f}" == "0.17"
    assert tipping < 0.5 * min(benchmarks[name][0].cf_d for name in ("discharge_risk", "age"))


def _check_refutation(namespace: dict[str, Any]) -> None:
    """Refutation: the placebo, subset, noise and generated-outcome readings of the slope."""
    trend, slope = namespace["trend_result"], namespace["SLOPE"]
    original = trend[slope].psi
    refutation = namespace["refutation"]
    tests = {test.name: test for test in refutation.tests}
    assert list(tests) == ["placebo", "subset", "random_common_cause"]
    for test in tests.values():
        # "No check prints a verdict": no registry row covers an MSM coefficient on three arms.
        assert test.passed is None and test.rung == "L0"
        assert len(test.values) == 20 and test.original == pytest.approx(original)
    assert refutation.passed is None
    assert (
        "This fit differs on estimand kind 'msm[assigned contacts]'"
        in tests["placebo"].scope_sentence
    )
    # The placebo reading: exact null 0, mean -0.00029724, spread 0.008957, t -0.1484 (p = 0.884).
    placebo = tests["placebo"]
    values = np.asarray(placebo.values)
    assert placebo.expectation == "~ 0"
    assert round(float(values.mean()), 8) == -0.00029724
    assert round(float(values.std(ddof=1)), 6) == 0.008957
    assert round(float(placebo.statistic), 4) == -0.1484
    assert round(float(placebo.statistic_pvalue), 3) == 0.884
    assert round(float(np.mean(placebo.standard_errors) / values.std(ddof=1)), 4) == 0.9241
    # The placebo mean sits far below the original slope; the no-intercept refit below is the
    # witness that the exact reference needs its stated conditions.
    assert abs(values.mean()) < 0.05 * original
    # The reference is exact only for the stated conditions. Without an intercept the line
    # through three equal means is not flat, so the placebo reference is a computed value.
    no_intercept = (
        namespace["study"]
        .identify(
            MSMProjection(
                MSM(
                    design=lambda arm, data: np.full(
                        (len(data), 1), namespace["CONTACTS_30D"][arm]
                    ),
                    terms=("assigned contacts",),
                    design_kind="known",
                )
            )
        )
        .estimate(method=namespace["method"])
    )
    nonzero = refute(
        no_intercept,
        estimand="msm[assigned contacts]",
        tests=("placebo",),
        n_replicates=10,
    ).tests[0]
    assert nonzero.expectation != "~ 0" and nonzero.passed is None
    intercept = refute(
        trend, estimand="msm[(intercept)]", tests=("placebo",), n_replicates=10
    ).tests[0]
    assert intercept.expectation != "~ 0"
    # The subset reading: 2100 of 3000 patients, scatter 0.0063258 against 0.0056268, 1.1242.
    subset = tests["subset"]
    scatter = float(np.std(subset.values, ddof=1))
    reference = float(trend[slope].std_error * np.sqrt((3000 / 2100 - 1) * 3000 / 2999))
    assert round(scatter, 7) == 0.0063258 and round(reference, 7) == 0.0056268
    assert round(scatter / reference, 4) == 1.1242
    assert round(float(subset.statistic_pvalue), 3) == 0.391
    # The noise reading: a nonzero, small paired shift in standard errors.
    noise = tests["random_common_cause"]
    shift = (np.asarray(noise.values) - original) / trend[slope].std_error
    assert 0.0 < abs(shift.mean()) < 0.01 and round(float(shift.mean()), 6) == -0.001259
    assert round(float(np.std(shift, ddof=1)), 5) == 0.04875

    # The generated laws declare the true means, so their true slope is the true line's slope.
    generated = {test.name: test for test in namespace["generated"].tests}
    assert list(generated) == ["dummy_outcome", "simulated_outcome", "plasmode"]
    projection = namespace["projection"]
    truth = {"dummy_outcome": 0.0, "simulated_outcome": projection[1], "plasmode": projection[1]}
    means = {"dummy_outcome": -0.00038784, "simulated_outcome": 0.26313, "plasmode": 0.2654}
    for name, test in generated.items():
        assert test.passed is None and test.rung == "L1" and len(test.values) == 20
        assert test.declared_effect == pytest.approx(truth[name], abs=1e-12)
        assert (
            round(float(np.mean(test.values)), 8 if name == "dummy_outcome" else 5) == means[name]
        )
        # "All nine readings print outside": 20 draws leave every interval across a band edge.
        assert len(test.readings) == 3 and not any(reading.inside for reading in test.readings)
    assert f"{projection[1]:.5f}" == "0.26571"
    by_name = {
        name: {reading.name: reading.estimate for reading in test.readings}
        for name, test in generated.items()
    }
    assert round(by_name["dummy_outcome"]["bias"], 5) == -0.02972
    assert round(by_name["simulated_outcome"]["bias"], 4) == -0.3191
    assert round(by_name["simulated_outcome"]["se_ratio"], 3) == 1.366
    assert round(by_name["plasmode"]["se_ratio"], 2) == 1.14
    inside_points = sum(
        reading.band[0] <= reading.estimate <= reading.band[1]
        for test in generated.values()
        for reading in test.readings
    )
    assert inside_points == 5, "the reading counts five point values inside their bands"
