"""What :class:`~cleverly.validation.EstimandSummary` reports, on which scale.

One property here and it is arithmetic rather than statistical: ``se_ratio`` divides a
reported standard error by an observed spread, and the two have to be on the same scale.
They were not for a ratio estimand -- ``psi`` is the odds ratio while ``std_error`` is
``SE(log OR)`` -- so the quotient came back at roughly ``1 / psi`` and a perfectly
calibrated ``or`` of ``0.42`` read as an interval 2.8 times too wide.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from cleverly.inference import make_estimate
from cleverly.validation import CoverageStudy, EstimandSummary


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

    This is the regression guard for the ``se_ratio`` bands registered studies assert on
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


class _Result:
    def __init__(self, *, alpha: float) -> None:
        estimate = make_estimate(
            "ate",
            0.0,
            np.array([-1.0, 1.0]),
            n=2,
            scale="difference",
            alpha=alpha,
        )
        self.estimates = {"ate": estimate}

    def __getitem__(self, name: str):  # type: ignore[no-untyped-def]
        return self.estimates[name]


class _Estimator:
    def __init__(self, *, alpha: float = 0.05, error: str | None = None) -> None:
        self.alpha = alpha
        self.error = error

    def fit(self, frame, **kwargs):  # type: ignore[no-untyped-def]
        if self.error is not None:
            raise RuntimeError(self.error)
        return _Result(alpha=self.alpha)


def test_coverage_study_uses_the_estimates_non_default_alpha() -> None:
    study = CoverageStudy(
        dgp=lambda n, seed: (SimpleNamespace(seed=seed), {"ate": 0.0}),
        estimator=lambda: _Estimator(alpha=0.10),
        n=2,
        n_replicates=3,
        seed=4,
    ).run()
    assert study.alpha == 0.10
    assert {record.alpha for record in study.replications} == {0.10}
    assert not any(record.rejected for record in study.replications)


def test_failed_replications_retain_the_seed_and_cause() -> None:
    seeds = np.random.SeedSequence(7).generate_state(3)

    def dgp(n, seed):  # type: ignore[no-untyped-def]
        if seed == int(seeds[0]):
            raise LookupError("bad draw")
        return SimpleNamespace(seed=seed), {"ate": 0.0}

    study = CoverageStudy(
        dgp=dgp,
        estimator=lambda: _Estimator(),
        n=2,
        n_replicates=3,
        seed=7,
    ).run()
    assert study.n_failed == 1
    assert study.failures[0].replicate == 0
    assert study.failures[0].seed == int(seeds[0])
    assert study.failures[0].error_type == "LookupError"
    assert study.failures[0].message == "bad draw"
    assert {record.replicate for record in study.replications} == {1, 2}


def test_an_all_failed_study_reports_the_first_cause() -> None:
    study = CoverageStudy(
        dgp=lambda n, seed: (None, {"ate": 0.0}),
        estimator=lambda: _Estimator(error="deliberate failure"),
        n=2,
        n_replicates=2,
        seed=8,
    )
    with pytest.raises(RuntimeError, match="RuntimeError: deliberate failure"):
        study.run()
