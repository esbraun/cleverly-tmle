"""Stable display of numerically negligible diagnostic residue."""

from __future__ import annotations

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
    )

    summary = check.summary()

    assert "0.000e+00" in summary
    assert "5.000e-10" in summary
