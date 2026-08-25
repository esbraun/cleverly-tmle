"""Structural gates for the ordinary survival LTMLE evidence study."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cleverly.datasets import rule_arm_at_node_two
from tests import discrete_law_survival as law
from tests.studies import canonical_ltmle_survival as study
from tests.studies import ltmle_survival_properties as properties


def test_dynamic_quadrature_is_stable_under_refinement() -> None:
    coarse = study.dynamic_survival_truth(nodes=48, panel=160)
    refined = study.dynamic_survival_truth(nodes=56, panel=192)
    assert coarse == pytest.approx(refined, abs=1e-12)


def test_the_dynamic_quadrature_integrates_the_rule_the_fits_are_given() -> None:
    r"""The witness :func:`longitudinal_rule_truth` gets from ``_check_step_rule``.

    ``dynamic_survival_truth`` splits its :math:`L_2` axis into two Gauss-Legendre panels
    meeting at ``0.0`` and reads the arm as ``0`` below and ``1`` above.  It never calls
    :func:`~cleverly.datasets.rule_arm_at_node_two`, which is the function both fits are
    handed -- so the threshold is written twice, once as a number and once as a comparison,
    and a change to either leaves a plausible number behind rather than an error.

    Both registered survival rows publish coverage for
    ``risk_regimen[treat then continue if l2 positive @ t=2]`` against that quadrature, so a
    silent disagreement here is a coverage claim for a regimen nobody estimated.  Checked on
    the panel interiors and on both sides of the jump rather than at ``0.0`` itself, which is
    a measure-zero point the integral cannot see.
    """
    below = np.array([-20.0, -3.0, -1e-9])
    above = np.array([1e-9, 3.0, 20.0])
    np.testing.assert_array_equal(rule_arm_at_node_two(below), np.zeros(3))
    np.testing.assert_array_equal(rule_arm_at_node_two(above), np.ones(3))


def test_the_primary_study_reports_every_unique_parameter_once() -> None:
    frame, truth = study.draw_from_seed(study.SCENARIO, 400, 3)
    result = study.fit_cleverly(frame)
    assert set(study.ESTIMANDS) == set(truth)
    assert set(study.ESTIMANDS) < set(result)
    assert len(study.ESTIMANDS) == 8

    dynamic = result.fits[f"{study.RULE_LABEL} @ t=1"]
    always = result.fits["always @ t=1"]
    assert dynamic.psi_scaled == always.psi_scaled
    np.testing.assert_array_equal(dynamic.influence_curve_scaled, always.influence_curve_scaled)


def _mass(probs: np.ndarray, **pattern: int) -> float:
    return float(sum(probs[index] for index in law._index(**pattern)))


def baseline_only(probs: np.ndarray, arm: int, horizon: int) -> float:
    """The cumulative risk an analysis that never conditions on ``L2`` would report.

    Written longhand off the support rather than by disabling something in the estimator, for
    the reason every deliberate-mutation control in this suite is: a flag on the code under
    audit makes the control a statement about a branch in it.  The only difference from
    :func:`law.functional` is where the second hazard conditions -- on ``(W, A1, A2, C2 = 1)``
    with ``L2`` marginalised out of the *observed* law, rather than on ``L2`` as well.  Since
    ``A2`` and ``C2`` both depend on ``L2``, conditioning on them reweights it, which is the
    bias a longitudinal fit exists to remove.
    """
    total = _mass(probs)
    psi = 0.0
    for w in (0, 1):
        share = _mass(probs, w=w) / total
        hazard1 = _mass(probs, w=w, a1=arm, c1=1, y1=1) / _mass(probs, w=w, a1=arm, c1=1)
        if horizon == 1:
            psi += share * hazard1
            continue
        reached = _mass(probs, w=w, a1=arm, c1=1, y1=0, a2=arm, c2=1)
        events = _mass(probs, w=w, a1=arm, c1=1, y1=0, a2=arm, c2=1, y2=1)
        psi += share * (hazard1 + (1.0 - hazard1) * events / reached)
    return psi


def test_the_null_is_exact_and_remains_a_survival_problem() -> None:
    assert properties.NULL_PROBS.sum() == pytest.approx(1.0, abs=1e-15)
    assert all(value == pytest.approx(0.0, abs=1e-14) for value in properties.NULL_TRUTH.values())
    assert properties.NULL_H1[0, 0] != properties.NULL_H1[0, 1]
    assert properties.NULL_H1[1, 0] != properties.NULL_H1[1, 1]
    assert properties.NULL_H2[0, 0, 0, 0] != properties.NULL_H2[0, 0, 1, 0]
    assert properties.NULL_H2[1, 0, 0, 0] != properties.NULL_H2[1, 0, 1, 0]


def test_the_horizon_two_null_is_one_a_longitudinal_fit_has_to_work_for() -> None:
    """The deliberate-mutation control: dropping ``L2`` must miss the null.

    The witnesses above say each hazard moves with something.  They do not say that an
    estimator has to be longitudinal to find the truth, and that is the claim the type-I cell
    rests on -- a null a baseline-only standardisation already recovers cannot tell a
    sequential-regression fit from a pair of cross-sections.
    """
    naive = baseline_only(properties.NULL_PROBS, 1, 2) - baseline_only(properties.NULL_PROBS, 0, 2)
    assert abs(naive) > 1e-3, "an analysis that ignores L2 already recovers the horizon-two null"


def test_the_horizon_one_null_carries_no_longitudinal_content() -> None:
    """And the other direction, pinned so the published limitation cannot drift from it.

    Nothing time varying precedes ``Y1``, so at the first horizon the baseline-only analysis
    *is* the identified functional and the cell cannot witness longitudinal adjustment.  It is
    still not a null no estimator has to work for: censoring at ``C1`` depends on ``W`` and on
    the arm, so a crude comparison of arms is biased under it.
    """
    naive = baseline_only(properties.NULL_PROBS, 1, 1) - baseline_only(properties.NULL_PROBS, 0, 1)
    assert naive == pytest.approx(0.0, abs=1e-15)

    def crude(arm: int) -> float:
        return _mass(properties.NULL_PROBS, a1=arm, c1=1, y1=1) / _mass(
            properties.NULL_PROBS, a1=arm, c1=1
        )

    assert abs(crude(1) - crude(0)) > 1e-3


def test_every_power_control_has_a_material_effect() -> None:
    for label, value in properties.POWER_TRUTH.items():
        assert abs(value) > 0.10, label


def test_the_unfluctuated_recursion_agrees_when_targeting_is_forced_to_be_zero() -> None:
    frame = law.frame()
    result = properties.fit(frame, "both_correct")
    for label, name in properties.CONTRASTS.items():
        left, right = name[len("ate_regimen[") : -1].rsplit(" @ t=", 1)[0].split(" vs ")
        horizon = int(name.rsplit(" @ t=", 1)[1][:-1])
        plug_in = properties.untargeted(
            frame, left, horizon, "both_correct"
        ) - properties.untargeted(frame, right, horizon, "both_correct")
        assert plug_in == pytest.approx(float(result[name].psi), abs=1e-9), label


def test_the_survivor_only_control_misses_cumulative_risk() -> None:
    naive = properties.survivor_only(law.frame())
    truth = law.TRUTH["risk_regimen[always @ t=2]"]
    assert abs(naive - truth) > 1e-2


def test_the_frozen_r_study_matches_beyond_its_acceptance_margin() -> None:
    path = study.STUDY.artifact("replicates.csv.gz")
    assert path.exists(), "published survival artifacts are required"
    rows = pd.read_csv(path)
    paired = rows.pivot(
        index=["scenario", "replicate", "estimand"],
        columns="implementation",
        values=["estimate", "std_error", "initial_estimate"],
    )
    for column, tolerance in {
        "estimate": 5e-7,
        "std_error": 1e-8,
        "initial_estimate": 5e-7,
    }.items():
        difference = paired[(column, "cleverly")] - paired[(column, "ltmle")]
        assert np.max(np.abs(difference)) < tolerance
