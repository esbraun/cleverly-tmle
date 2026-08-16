"""What :class:`~cleverly.validation.EstimandSummary` reports, on which scale.

One property here and it is arithmetic rather than statistical: ``se_ratio`` divides a
reported standard error by an observed spread, and the two have to be on the same scale.
They were not for a ratio estimand -- ``psi`` is the odds ratio while ``std_error`` is
``SE(log OR)`` -- so the quotient came back at roughly ``1 / psi`` and a perfectly
calibrated ``or`` of ``0.42`` read as an interval 2.8 times too wide.
"""

from __future__ import annotations

import numpy as np
import pytest

from cleverly.validation import EstimandSummary


def _summary(**overrides: object) -> EstimandSummary:
    defaults: dict[str, object] = {
        "estimand": "or",
        "truth": 0.5,
        "n": 500,
        "n_replicates": 4,
        "estimates": np.array([0.40, 0.50, 0.60, 0.55]),
        "std_errors": np.array([0.20, 0.22, 0.24, 0.21]),
        "covered": np.ones(4),
        "rejected": np.zeros(4),
    }
    return EstimandSummary(**{**defaults, "estimand": "or", **overrides})  # type: ignore[arg-type]


def test_the_ratio_is_taken_on_the_scale_the_error_is_reported_on() -> None:
    estimates = np.array([0.40, 0.50, 0.60, 0.55])
    logs = np.log(estimates)
    summary = _summary(inference_estimates=logs)
    assert summary.se_ratio == pytest.approx(
        float(np.mean(summary.std_errors)) / float(np.std(logs, ddof=1))
    )


def test_the_two_scales_disagree_enough_to_matter() -> None:
    """The mutation control: without it the assertion above could be a tautology."""
    estimates = np.array([0.40, 0.50, 0.60, 0.55])
    on_log = _summary(inference_estimates=np.log(estimates)).se_ratio
    on_ratio = _summary().se_ratio
    # Roughly `1 / psi` apart, which is the whole defect.
    assert on_ratio / on_log == pytest.approx(1.0 / float(np.mean(estimates)), rel=0.1)


def test_a_difference_estimand_is_arithmetically_what_it_always_was() -> None:
    """``inference_estimates=None`` means the scales coincide, which is every non-ratio.

    This is the regression guard for the ``se_ratio`` bands the slow tier asserts on
    ``ate``, ``ey_regimen`` and ``risk_regimen``: none of them may move by so much as a
    rounding.
    """
    summary = _summary(estimand="ate")
    assert summary.inference_estimates is None
    np.testing.assert_array_equal(summary.inference_scale_estimates, summary.estimates)
    assert summary.se_ratio == float(np.mean(summary.std_errors)) / summary.monte_carlo_se


def test_a_degenerate_spread_reports_nothing_rather_than_dividing_by_zero() -> None:
    summary = _summary(estimand="ate", estimates=np.full(4, 0.3))
    assert np.isnan(summary.se_ratio)
