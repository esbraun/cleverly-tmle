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
from cleverly.msm import MSM
from cleverly.sensitivity.omitted_variable import OMITTED_VARIABLE_OPERATIONS
from cleverly.validation import refute
from tests.documents import ROOT
from tests.unit.tutorial_semantics import (
    EXAMPLES,
    assert_protocol_recorded,
    changed_fields,
    covers,
    stored_output,
)

NOTEBOOK = EXAMPLES / "msm-projections.ipynb"

_PROBE = "reviews/notebook-review/probes/msm-projections-final"

#: Decimals the readings write that no stored output prints, each with its source.
UNPRINTED_DECIMALS = {
    "0.9405": f"Step 7 slope coverage, fitted g (3762 of 4000), seeds 30000-33999; the same value "
    f"for the 1:10:1 slope: {_PROBE}/sweep.log",
    "0.01": f"the true-g threshold, share below it 0.0112 for medium or high: "
    f"{_PROBE}/shortfall.log",
    "0.927": "the simulated_outcome SE ratio at n_replicates=400, a review probe recorded in the body of pull request 435",
    "0.734": "half the stored ate[high vs low], 1.4681 / 2, checked in check()",
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
    assert re.search(r"slope_cover: 3762/4000 = 0\.9405 ", sweep)
    assert _log_value(sweep, r"weighted_cover: 3762/4000 = ([0-9.]+)") == "0.9405"
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
    summary = effect.summary()

    assert len(levels) == 3, "the tutorial no longer witnesses multi-arm identification"
    assert fitted.nuisance.propensity.values.shape[1] == len(levels)
    for level in levels:
        assert str(level) in positivity
    assert "treatment_mechanism" in summary
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
    # "A score with no documented range has no range to declare as q_bounds": the page's own
    # reason for fitting in sample, and the fit reports it. Death uses the hypothetical strategy.
    protocol = namespace["protocol"]
    assert "no documented range" in protocol.outcome
    assert "standardized" not in protocol.outcome
    assert "hypothetical strategy" in protocol.intercurrent_event_handling[2]
    assert not method.cross_fitting.enabled
    assert "in-sample nuisances" in fitted.summary()
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
    # (b) Step 10's fixed 1:10:1 weight moves the fitted slope to its own projection.
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
    assert not covers(weighted_slope, projection[1])

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
    # The miss at medium is a fixed contrast of the three arm means, read through ``contrast``.
    # "The interval excludes zero" and "the interval contains the population miss".
    medium_misfit = namespace["medium_misfit"]
    means = np.array([fitted[f"ey[{arm}]"].psi for arm in namespace["ARMS"]])
    hat = design @ np.linalg.pinv(design)
    assert medium_misfit.psi == pytest.approx((hat @ means - means)[1], rel=1e-10, abs=1e-12)
    assert medium_misfit.ci[1] < 0.0
    assert covers(medium_misfit, residual[1])
    # "Its estimate differs from the Step 7 miss by 0.0008 on this draw."
    step7_miss = namespace["estimated_line"][1] - means[1]
    assert f"{abs(medium_misfit.psi - step7_miss):.4f}" == "0.0008"

    # Step 8: two known codings of the same means answer two questions. Each interval
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
    # "half of that is 0.734": the per-step slope is half the high-vs-low difference.
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
    # The status table prints only the rows that ran or run on request. The fit reports two
    # coefficients, so each omitted-variable row waits on an estimand (X82 part G2c).
    ledger = assessment.to_frame().set_index(["surface", "check"])["status"]
    stored = stored_output(NOTEBOOK, "assessment")
    for operation in OMITTED_VARIABLE_OPERATIONS:
        assert str(ledger.loc[("sensitivity", operation)]) == "deferred"
        assert re.search(rf"sensitivity\s+{operation}\s+deferred", stored)
    assert "unavailable" not in stored and "not_applicable" not in stored
    assert str(ledger.loc[("validation", "support")]) == "warning"
    assert re.search(r"support\s+warning", stored)
    assert re.search(r"refute\s+deferred", stored)
    support = namespace["support"]
    # "The fit truncated 1.20% of the units, and the support report warns above 1%."
    assert support.truncated["fraction"] > 0.01
    assert support.severity == "strain"
    ratios = {arm: row["ratio"] for arm, row in support.effective_sample_size.items()}
    assert min(ratios, key=ratios.get) == "high"
    # "The largest clever covariate is the high contact count divided by a fitted g[high]."
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

    # Step 13 reads the omitted-variable bound on the arm contrasts of the ATE fit, whose
    # influence curves are those of the saturated coefficients (checked above), and on the
    # slope of the trend fit.
    robustness = namespace["robustness"]
    assert set(robustness) == {"ate[medium vs low]", "ate[high vs low]", "msm[assigned contacts]"}
    assert 0.0 < robustness["ate[medium vs low]"]["rv"] < robustness["ate[high vs low]"]["rv"]
    for values in robustness.values():
        assert 0.0 < values["rva"] < values["rv"] < 1.0
    # "The slope's confidence-limit value is descriptive."
    assert robustness["msm[assigned contacts]"]["limit_licensed"] is False
    assert "limit_licensed" not in robustness["ate[high vs low]"]
    _check_refutation(namespace)
    _check_stress_surface(namespace)


def _check_refutation(namespace: dict[str, Any]) -> None:
    """Step 13: the placebo, subset, noise and generated-outcome readings of the slope."""
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
    assert "This fit differs on method 'TMLE'" in tests["placebo"].scope_sentence
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

    # The generated laws declare the Step 2 means, so their true slope is the Step 9 projection.
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


def _check_stress_surface(namespace: dict[str, Any]) -> None:
    """Step 14: the grid cells, the benchmark scale, and the stated movements."""
    surface, trend = namespace["surface"], namespace["trend_result"]
    slope = trend[namespace["SLOPE"]]
    frame = surface.to_frame()
    assert list(zip(frame["treatment_strength"], frame["outcome_strength"], strict=True)) == [
        (0.0, 0.0),
        (0.0, 0.9),
        (0.1, 0.0),
        (0.1, 0.9),
        (0.3, 0.0),
        (0.3, 0.9),
    ]
    # The anchor cell is the original fit.
    assert frame["displacement"].iloc[0] == 0.0
    assert frame["estimate"].iloc[0] == pytest.approx(slope.psi, abs=1e-12)
    moves = (frame["displacement"] / slope.std_error).round(2).tolist()
    assert moves == [0.0, -1.44, -8.98, -9.15, -19.74, -19.33]
    assert frame["estimate"].min() > 0.0 and round(float(frame["estimate"].min()), 4) == 0.0991
    association = frame["induced_treatment_association"].round(4).tolist()
    assert association == [0.0372, 0.0372, -0.0393, -0.0393, -0.1027, -0.1027]
    # A treatment swap moves the slope by misclassification: its movement dwarfs the outcome cell.
    assert abs(frame["displacement"].iloc[2]) > 5 * abs(frame["displacement"].iloc[1])
    strength = {(item.covariate, item.role): item.strength for item in surface.calibrations}
    assert round(strength[("discharge_risk", "treatment")], 3) == 0.300
    assert round(strength[("discharge_risk", "outcome")], 3) == 0.916
    assert round(strength[("age", "treatment")], 3) == 0.276
    assert round(strength[("age", "outcome")], 3) == -0.361
    assert round(slope.std_error, 4) == 0.0086
    # The slope has no E-value: the E-value reads an arm contrast.
    with pytest.raises(Exception, match="arm contrast"):
        trend.sensitivity.evalue(estimand=namespace["SLOPE"])
    assert "estimand kind" in namespace["robustness"]["msm[assigned contacts]"]["limit_sentence"]
