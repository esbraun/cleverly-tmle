"""A held baseline treatment with an end-of-study outcome after two censoring nodes (D5).

Three identities on the exactly realised law of ``tests/discrete_law_point_survival.py``:

* the held fit is the g-formula, and its influence curve is the Gateaux derivative;
* the held fit equals the shipped fit on copied treatment columns;
* without a post-baseline covariate, the held fit equals point TMLE with ``delta=`` and
  :math:`\\Delta = C_1 C_2`.  With saturated nuisances
  :math:`\\prod_t c_t(A, W) = P(\\Delta = 1 \\mid A, W)`, so the telescoping influence curve
  keeps only the last node's term, which is the point-TMLE curve.  This is an indicator
  and missing-outcome reduction.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from cleverly.estimators import TMLE
from cleverly.longitudinal import LTMLE
from tests.discrete_law_point_survival import (
    CellMeans,
    CellProbabilities,
    end_of_study_law,
    static,
)

REGIMENS = {"a0": 0, "a1": 1, "a2": 2}
EXACT = os.environ.get("CI") is None
FLOOR = 1e-12


def _learners() -> dict[str, object]:
    return {
        "outcome_learner": CellMeans(),
        "treatment_learner": CellProbabilities(),
        "censoring_learner": CellMeans(),
    }


@pytest.mark.parametrize("with_l2", [False, True])
def test_the_held_end_of_study_fit_is_the_g_formula(with_l2: bool) -> None:
    law = end_of_study_law(with_l2=with_l2)
    result = LTMLE(REGIMENS, n_folds=1, **_learners()).fit(law.frame(), **law.fit_columns())
    rows = law.first_row_of()
    for label, arm in REGIMENS.items():
        estimate = result[f"ey_regimen[{label}]"]
        assert estimate.psi == pytest.approx(
            law.functional(law.probs, static(arm)), rel=1e-12, abs=FLOOR
        )
        np.testing.assert_allclose(
            estimate.influence_curve[rows], law.eif(static(arm)), rtol=1e-10, atol=1e-12
        )


@pytest.mark.parametrize("with_l2", [False, True])
def test_the_held_end_of_study_fit_equals_the_copied_column_fit(with_l2: bool) -> None:
    law = end_of_study_law(with_l2=with_l2)
    held = LTMLE(REGIMENS, n_folds=1, **_learners()).fit(law.frame(), **law.fit_columns())
    copied = LTMLE(REGIMENS, n_folds=1, **_learners()).fit(
        law.frame(), **law.fit_columns(held=False)
    )
    for name in held.estimates:
        if EXACT:
            assert held[name].psi == copied[name].psi
            assert np.array_equal(held[name].influence_curve, copied[name].influence_curve)
        else:
            assert held[name].psi == pytest.approx(copied[name].psi, rel=1e-12, abs=FLOOR)
            np.testing.assert_allclose(
                held[name].influence_curve, copied[name].influence_curve, rtol=1e-12, atol=FLOOR
            )


def test_without_l2_the_held_fit_is_point_tmle_with_a_missing_outcome() -> None:
    law = end_of_study_law()
    frame = law.frame()
    frame["Delta"] = ((frame["C1"] == 1) & (frame["C2"] == 1)).astype(float)
    held = LTMLE(REGIMENS, n_folds=1, **_learners()).fit(frame, **law.fit_columns())
    point = (
        TMLE(
            outcome_learner=CellMeans(),
            treatment_learner=CellProbabilities(),
            missingness_learner=CellMeans(),
            cross_fit=False,
        )
        .fit(frame, outcome="Y", treatment="A", covariates=["W"], delta="Delta")
        .single()
    )
    for label, arm in REGIMENS.items():
        left = held[f"ey_regimen[{label}]"]
        right = point[f"ey[{float(arm)}]"]
        assert left.psi == pytest.approx(right.psi, rel=1e-12, abs=FLOOR)
        np.testing.assert_allclose(
            left.influence_curve, right.influence_curve, rtol=1e-12, atol=1e-12
        )


def test_the_node_count_comes_from_censoring_and_must_agree() -> None:
    law = end_of_study_law(with_l2=True)
    columns = law.fit_columns()
    columns["time_varying"] = [[], ["L2"], []]
    with pytest.raises(Exception, match="held over every node"):
        LTMLE(REGIMENS, n_folds=1, **_learners()).fit(law.frame(), **columns)
    result = LTMLE(REGIMENS, n_folds=1, **_learners()).fit(law.frame(), **law.fit_columns())
    assert result.data.n_times == 2
    assert result.data.treatment_names == ("A",)
