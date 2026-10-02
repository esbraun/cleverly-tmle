"""Stable display of numerically negligible diagnostic residue."""

from __future__ import annotations

import pytest

from cleverly.validation.drtmle import CorrectionCheck, CorrectionRow
from cleverly.validation.score import ScoreCheck, ScoreCheckRow


def test_score_summary_stabilizes_only_values_far_below_the_threshold() -> None:
    rows = (
        ScoreCheckRow("tiny", "fluctuation", 1e-18, 1e-8, 0.1, True, True, 1, "newton"),
        ScoreCheckRow("visible", "fluctuation", 5e-13, 1e-8, 0.1, True, True, 1, "newton"),
    )

    summary = ScoreCheck(rows, tolerance=1e-3, n=100).summary()

    assert "0.000e+00" in summary
    assert "0.00e+00" in summary
    assert "5.000e-13" in summary
    assert "5.00e-05" in summary


def test_correction_summary_stabilizes_only_residue_far_below_the_identity_bar() -> None:
    rows = (
        CorrectionRow(0, 0.0, "0", "D*_Q", 1e-18, 0.0, 0, True),
        CorrectionRow(0, 1.0, "1", "D*_Q", 5e-10, 0.0, 0, True),
    )
    check = CorrectionCheck(
        rows,
        tolerance=1e-3,
        identity_tolerance=1e-10,
        n=100,
        std_error=0.1,
        cross_fitted=False,
    )

    summary = check.summary()

    assert "0.000e+00" in summary
    assert "5.000e-10" in summary


def test_correction_check_requires_a_cross_fitting_declaration() -> None:
    row = CorrectionRow(0, 0.0, "0", "D*_g", 0.0, 0.0, 0, True, margin=0.2, gr1_margin=0.2)
    with pytest.raises(TypeError, match="required keyword-only argument: 'cross_fitted'"):
        CorrectionCheck((row,), tolerance=1e-3, identity_tolerance=1e-10, n=100, std_error=0.1)


@pytest.mark.parametrize("cross_fitted", [False, True])
def test_correction_check_distinguishes_construction_from_theorem_scope(
    cross_fitted: bool,
) -> None:
    row = CorrectionRow(0, 0.0, "0", "D*_g", 0.0, 0.0, 0, True, margin=0.2, gr1_margin=0.2)
    check = CorrectionCheck(
        (row,),
        tolerance=1e-3,
        identity_tolerance=1e-10,
        n=100,
        std_error=0.1,
        cross_fitted=cross_fitted,
    )
    assert check.rows
    assert check.truncations_active == ()
    assert check.contract == "theorem"
    assert check.passed
    summary = check.summary()
    assert ("this fit is Theorem 1's estimator" in summary) is not cross_fitted
    assert ("uses Theorem 1's construction" in summary) is cross_fitted
    assert ("Theorem 1 does not cover cross-fitting" in summary) is cross_fitted
