"""Structural gates for the ordinary survival LTMLE evidence study."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests import discrete_law_survival as law
from tests.studies import canonical_ltmle_survival as study
from tests.studies import ltmle_survival_properties as properties


def test_dynamic_quadrature_is_stable_under_refinement() -> None:
    coarse = study.dynamic_survival_truth(nodes=48, panel=160)
    refined = study.dynamic_survival_truth(nodes=56, panel=192)
    assert coarse == pytest.approx(refined, abs=1e-12)


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


def test_the_null_is_exact_and_remains_a_survival_problem() -> None:
    assert properties.NULL_PROBS.sum() == pytest.approx(1.0, abs=1e-15)
    assert all(value == pytest.approx(0.0, abs=1e-14) for value in properties.NULL_TRUTH.values())
    assert properties.NULL_H1[0, 0] != properties.NULL_H1[0, 1]
    assert properties.NULL_H1[1, 0] != properties.NULL_H1[1, 1]
    assert properties.NULL_H2[0, 0, 0, 0] != properties.NULL_H2[0, 0, 1, 0]
    assert properties.NULL_H2[1, 0, 0, 0] != properties.NULL_H2[1, 0, 1, 0]


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
