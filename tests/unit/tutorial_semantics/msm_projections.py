"""The reviewed semantic callback for ``docs/examples/msm-projections``.

``tests/unit/test_documentation_runtime.py`` runs the tutorial at its documented size and
passes the resulting namespace to :func:`check`.  The tutorial is a notebook, so its narrated
decimals are also compared against its stored outputs, and every decimal it writes is printed.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from sklearn.dummy import DummyRegressor

from cleverly import MSMProjection
from cleverly.datasets import navigation_protocol
from cleverly.msm import MSM
from cleverly.sensitivity.omitted_variable import OMITTED_VARIABLE_OPERATIONS
from tests.unit.tutorial_semantics import (
    EXAMPLES,
    assert_protocol_recorded,
    changed_fields,
    covers,
)

NOTEBOOK = EXAMPLES / "msm-projections.ipynb"


def _weighted_projection(design: np.ndarray, weights: np.ndarray, means: np.ndarray) -> np.ndarray:
    """Solve the weighted normal equations of the three-point projection."""
    return np.linalg.solve(design.T @ (weights[:, None] * design), design.T @ (weights * means))


def check(namespace: dict[str, Any]) -> None:
    """The projection tutorial's protocol, failure mode, control, and refusal claims hold."""
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
    # "A standardized score takes its scale from the data, so no finite support can be declared
    # for it": the page's own reason for fitting in sample, and the fit reports it.
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
    # The page refuses text cadence labels. Only the MSM.linear refusal may satisfy it.
    assert "reads the treatment level" in namespace["refusal"]

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
    # (b) A fixed, strongly nonuniform weight moves the fitted slope to its own projection.
    # This fit's correctly specified Q witnesses the weighted Gram matrix.
    fixed = {"low": 1.0, "medium": 10.0, "high": 1.0}
    weighted = study.identify(
        MSMProjection(
            MSM(
                design=contacts_design,
                terms=terms,
                weights=lambda arm, data: np.full(len(data), fixed[arm]),
                weights_kind="known",
                design_kind="known",
            )
        )
    ).estimate(method=method)
    fixed_target = _weighted_projection(
        design, np.array([fixed[arm] for arm in namespace["ARMS"]]), population
    )
    assert covers(weighted["msm[assigned contacts]"], fixed_target[1])
    assert not covers(trend["msm[assigned contacts]"], fixed_target[1])
    assert not covers(weighted["msm[assigned contacts]"], projection[1])
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
    # "Its estimate differs from the Step 7 miss in the fourth decimal."
    step7_miss = namespace["estimated_line"][1] - means[1]
    assert 0.0 < abs(medium_misfit.psi - step7_miss) < 0.005

    # The assessment names the support warning, and the slope barely moves along the curve
    # even where the largest bound clips most of the rows.
    assessment = namespace["assessment"]
    attention = {item.name for item in assessment.attention}
    assert attention == {"support"}
    assert namespace["scores"].passed
    # "the omitted-variable operations are unavailable for this fit"
    ledger = assessment.to_frame().set_index(["surface", "check"])["status"]
    for operation in OMITTED_VARIABLE_OPERATIONS:
        assert str(ledger.loc[("sensitivity", operation)]) == "unavailable"
    support = namespace["support"]
    # "The fit truncated 1.23% of the units, and the support report warns above 1%."
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

    # No omitted-variable bound is implemented for an MSM coefficient; the arm contrasts
    # have one. The claim the message makes about the representer reversed: an MSM
    # coefficient *has* one, so the bound is well posed and only the implementation is
    # missing. The refusal must not send the reader to ``evalue``, which refuses an ``msm``
    # target of its own.
    refusal = namespace["sensitivity_refusal"]
    assert "Riesz representer" in refusal
    assert "well posed" in refusal
    assert "evalue" not in refusal
    robustness = namespace["robustness"]
    assert 0.0 < robustness["ate[medium vs low]"]["rv"] < robustness["ate[high vs low]"]["rv"]
    for values in robustness.values():
        assert 0.0 < values["rva"] < values["rv"] < 1.0
