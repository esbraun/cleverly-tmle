"""Fast structural checks for the registered longitudinal TMLE evidence study."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tests import discrete_law_longitudinal as law
from tests.studies import canonical_ltmle as study
from tests.studies import ltmle_properties as properties


def test_quasibinomial_irls_solves_the_fractional_response_score() -> None:
    design = np.array(
        [
            [-1.0, 0.0],
            [-0.5, 1.0],
            [0.0, -1.0],
            [0.5, 0.5],
            [1.0, -0.5],
            [1.5, 1.0],
        ]
    )
    coefficient = np.array([-0.4, 0.7, -0.3])
    augmented = np.column_stack([np.ones(len(design)), design])
    target = 1.0 / (1.0 + np.exp(-(augmented @ coefficient)))
    fitted = study.QuasiBinomialGLM(tol=1e-12).fit(design, target)
    np.testing.assert_allclose(
        np.concatenate([fitted.intercept_, fitted.coef_[0]]), coefficient, atol=1e-10
    )


def test_the_sharp_null_is_exact_and_retains_baseline_confounding() -> None:
    assert properties.NULL_TRUTH == 0.0
    assert law.G1[0] != law.G1[1]
    for label in law.REGIMEN_ARMS:
        name = f"ey_regimen[{label}]"
        assert law.functional(properties.NULL_PROBS, name) == 0.5


def test_the_frozen_r_study_matches_beyond_its_acceptance_margin() -> None:
    rows = pd.read_csv(study.STUDY.artifact("replicates.csv.gz"))
    paired = rows.pivot(
        index=["scenario", "replicate", "estimand"],
        columns="implementation",
        values=["estimate", "std_error", "initial_estimate"],
    )
    tolerances = {"estimate": 3e-7, "std_error": 1e-8, "initial_estimate": 3e-7}
    for column, tolerance in tolerances.items():
        difference = paired[(column, "cleverly")] - paired[(column, "ltmle")]
        assert np.max(np.abs(difference)) < tolerance


def test_the_primary_dynamic_plan_assigns_both_second_node_arms() -> None:
    frame, _ = study.draw_scenario(study.SCENARIO, study.PRIMARY_N, 0)
    result = study.fit_cleverly(frame)
    assignment = result.fits[study.RULE_LABEL].assignment[:, 1]
    assert set(np.unique(assignment)) == {0.0, 1.0}
