"""A benchmark hands its strengths to the omitted-variable bound unchanged (RM47).

Remark 8 of Appendix E.5 of Chernozhukov et al. (arXiv:2112.13398v6) states that the gain
metrics of a benchmark serve as proxies for the share-form sensitivity parameters.  The
bound therefore reads the gains as they are.  These tests pin that hand-off, the unclipped
gains a record reports, and the refusals of a benchmark that cannot calibrate the bound.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import ATE, CausalStudy, PointTreatment
from cleverly._summary.vocabulary import NOTE_TEXTS
from cleverly.datasets import make_linear_ate
from cleverly.sensitivity import benchmark, omitted_variable_bounds

pytestmark = pytest.mark.xdist_group("omitted_variable_benchmark_handoff")


def _fit(seed: int = 1) -> object:
    frame, _ = make_linear_ate(n=400, seed=seed)
    study = CausalStudy(
        frame,
        design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2", "W3", "W4")),
    )
    return study.identify(ATE()).estimate(
        outcome_learner=LinearRegression(),
        treatment_learner=LogisticRegression(max_iter=1000),
        cross_fit=False,
        random_state=0,
    )


@pytest.fixture(scope="module")
def result() -> object:
    return _fit()


@pytest.fixture(scope="module")
def moderate(result: object) -> object:
    """A benchmark whose two gains are positive and unclipped."""
    record = benchmark(result, ["W2"])
    assert 0.0 < record.cf_y < 1.0 and 0.0 < record.cf_d < 1.0
    return record


@pytest.fixture(scope="module")
def strong(result: object) -> object:
    """A benchmark whose outcome gain exceeds 1, so its ``cf_y`` is a clip."""
    record = benchmark(result, ["W1"])
    assert record.gain_y > 1.0
    return record


class TestGains:
    def test_the_gains_are_the_unclipped_ratios(self, moderate: object, strong: object) -> None:
        for record in (moderate, strong):
            assert record.gain_y == pytest.approx(record.sigma2_short / record.sigma2_long - 1)
            assert record.gain_d == pytest.approx(record.nu2_long / record.nu2_short - 1)
        assert moderate.cf_y == pytest.approx(moderate.gain_y, rel=1e-12)
        assert moderate.cf_d == pytest.approx(moderate.gain_d, rel=1e-12)

    def test_a_clipped_gain_is_reported_and_printed(self, strong: object) -> None:
        assert strong.cf_y == 1.0
        assert strong.to_dict()["gain_y"] == strong.gain_y
        assert strong.summary().note("strength_clipped").text == NOTE_TEXTS[
            "strength_clipped_y"
        ].format(covariates=", ".join(strong.covariates), gain=f"{strong.gain_y:.3g}")

    def test_an_unclipped_record_prints_no_clip(self, moderate: object) -> None:
        assert "clipped" not in moderate.summary()


class TestHandOff:
    def test_the_bound_reads_the_gains_unchanged(self, result: object, moderate: object) -> None:
        handed = omitted_variable_bounds(result, benchmark=moderate)
        explicit = omitted_variable_bounds(result, cf_y=moderate.cf_y, cf_d=moderate.cf_d)
        assert (handed.cf_y, handed.cf_d, handed.rho) == (moderate.cf_y, moderate.cf_d, 1.0)
        assert (handed.lower, handed.upper) == (explicit.lower, explicit.upper)
        # Nonzero witness: the share form gain / (1 + gain) is a different bound, so a
        # hand-off that converted the gains would fail the equality above.
        share_y, share_d = (g / (1.0 + g) for g in (moderate.cf_y, moderate.cf_d))
        shares = omitted_variable_bounds(result, cf_y=share_y, cf_d=share_d)
        assert shares.upper - shares.lower < 0.9 * (handed.upper - handed.lower)

    def test_the_facade_and_the_assessment_name_the_benchmark(
        self, result: object, moderate: object
    ) -> None:
        from cleverly.assessment import _omitted_item

        handed = result.sensitivity.omitted_confounding(benchmark=moderate)
        assert handed.lower == omitted_variable_bounds(result, benchmark=moderate).lower
        item = _omitted_item(handed, result, {"benchmark": moderate})
        assert "at the strengths of the benchmark against ['W2']" in item.detail
        assert "default" not in item.detail

    def test_rho_stays_the_callers(self, result: object, moderate: object) -> None:
        aligned = omitted_variable_bounds(result, benchmark=moderate, rho=moderate.rho)
        assert aligned.rho == moderate.rho
        worst = omitted_variable_bounds(result, benchmark=moderate)
        assert aligned.confounding_strength <= worst.confounding_strength

    def test_a_clipped_outcome_gain_calibrates_a_finite_bound(
        self, result: object, strong: object
    ) -> None:
        bounds = omitted_variable_bounds(result, benchmark=strong)
        assert bounds.cf_y == 1.0
        assert np.isfinite(bounds.lower) and np.isfinite(bounds.upper)
        assert bounds.confounding_strength == pytest.approx(
            np.sqrt(strong.cf_d / (1.0 - strong.cf_d))
        )

    def test_a_clipped_riesz_gain_is_refused(self, result: object, moderate: object) -> None:
        unbounded = replace(moderate, nu2_short=moderate.nu2_long / 2.5, cf_d=1.0)
        assert unbounded.gain_d == pytest.approx(1.5)
        with pytest.raises(ValueError, match="calibrates no finite bound"):
            omitted_variable_bounds(result, benchmark=unbounded)
        summary = unbounded.summary()
        assert (
            NOTE_TEXTS["strength_clipped_d"].format(gain="1.5")
            in summary.note("strength_clipped").text
        )
        assert NOTE_TEXTS["benchmark_pointer"] not in summary.note("sensitivity_assumption").text

    def test_explicit_strengths_conflict(self, result: object, moderate: object) -> None:
        with pytest.raises(ValueError, match="cf_y cannot also be passed"):
            omitted_variable_bounds(result, benchmark=moderate, cf_y=0.1)
        with pytest.raises(ValueError, match="cf_y and cf_d cannot also be passed"):
            omitted_variable_bounds(result, benchmark=moderate, cf_y=0.1, cf_d=0.1)

    def test_another_estimand_is_refused(self, result: object, moderate: object) -> None:
        with pytest.raises(ValueError, match="Pass estimand='ate'"):
            omitted_variable_bounds(result, "ey1", benchmark=moderate)

    def test_another_fit_is_refused(self, moderate: object) -> None:
        with pytest.raises(ValueError, match="computed on another fit"):
            omitted_variable_bounds(_fit(seed=2), benchmark=moderate)

    def test_a_non_benchmark_is_refused(self, result: object) -> None:
        with pytest.raises(TypeError, match="BenchmarkResult"):
            omitted_variable_bounds(result, benchmark={"cf_y": 0.1, "cf_d": 0.1})
