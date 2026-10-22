"""The omitted-variable bound of a fit with ``repeats`` above one (X82 part G, X65 f).

Each draw of the cross-fitting split has its own bound. The fit combines the draws' bounds by
the median, and their one-sided limits by the rule that it applies to the estimate:
``median_r{se_r^2 + (t_r - t)^2}``. The witnesses below rest on one fact the repeated fit
documents: draw ``r`` is the fit with one draw at that draw's seed. So every per-draw number
is an independent fit, and the combined bound is checked against the median of those fits.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from cleverly.assessment import AssessmentStatus
from cleverly.datasets import make_binary_outcome
from cleverly.exceptions import CapabilityError
from cleverly.sensitivity import omitted_variable as ov
from cleverly.sensitivity.omitted_variable import (
    benchmark,
    contour_data,
    omitted_variable_bounds,
    robustness_value,
    sensitivity_elements,
)
from tests.conftest import fast_tmle

REPEATS = 3
CF = 0.04


def fit(repeats: int = REPEATS, **overrides: Any) -> Any:
    frame, _ = make_binary_outcome(n=400, seed=17)
    estimator = fast_tmle(repeats=repeats, estimands=("ey1", "ate"), **overrides)
    return estimator.fit(frame, outcome="Y", treatment="A").single()


@pytest.fixture(scope="module")
def repeated() -> Any:
    return fit()


@pytest.fixture(scope="module")
def single_draws(repeated: Any) -> list[Any]:
    """The fit with one draw at each draw's seed: draw ``r`` of the repeated fit."""
    return [fit(repeats=1, random_state=draw.seed) for draw in repeated.repeats]


@pytest.fixture(scope="module")
def single_bounds(single_draws: list[Any]) -> list[Any]:
    return [omitted_variable_bounds(draw, "ate", cf_y=CF, cf_d=CF) for draw in single_draws]


def median_rule(points: np.ndarray, std_errors: np.ndarray) -> tuple[float, float]:
    center = float(np.median(points))
    return center, float(np.sqrt(np.median(std_errors**2 + (points - center) ** 2)))


class TestEachDrawIsTheFitWithOneDrawAtItsSeed:
    def test_the_draws_reproduce_the_stored_estimates(
        self, repeated: Any, single_draws: list[Any]
    ) -> None:
        for draw, single in zip(repeated.repeats, single_draws, strict=True):
            assert draw.psi["ate"] == pytest.approx(single["ate"].psi, rel=1e-12)

    def test_each_draws_elements_are_that_fits_elements(
        self, repeated: Any, single_draws: list[Any]
    ) -> None:
        for index, single in enumerate(single_draws):
            own = sensitivity_elements(repeated, "ate", repeat=index)
            other = sensitivity_elements(single, "ate")
            assert own.max_bias == pytest.approx(other.max_bias, rel=1e-12)
            np.testing.assert_allclose(own.psi_max_bias, other.psi_max_bias, rtol=1e-10)


class TestTheMedianRule:
    def test_the_draws_differ(self, single_bounds: list[Any]) -> None:
        """The nonzero witness: a median over equal draws checks nothing."""
        assert len({round(bound.lower, 12) for bound in single_bounds}) == REPEATS

    def test_the_bounds_are_the_medians_of_the_draws_bounds(
        self, repeated: Any, single_bounds: list[Any]
    ) -> None:
        combined = omitted_variable_bounds(repeated, "ate", cf_y=CF, cf_d=CF)
        lowers = [bound.lower for bound in single_bounds]
        uppers = [bound.upper for bound in single_bounds]
        assert combined.lower == pytest.approx(float(np.median(lowers)), rel=1e-12)
        assert combined.upper == pytest.approx(float(np.median(uppers)), rel=1e-12)
        assert combined.max_bias == pytest.approx(
            float(np.median([bound.max_bias for bound in single_bounds])), rel=1e-12
        )
        assert combined.psi == repeated["ate"].psi

    @pytest.mark.parametrize("end", ["lower", "upper"])
    def test_each_limit_follows_the_median_rule(
        self, repeated: Any, single_bounds: list[Any], end: str
    ) -> None:
        combined = omitted_variable_bounds(repeated, "ate", cf_y=CF, cf_d=CF)
        quantile = ov._one_sided_quantile(combined.level)
        edges = np.array([getattr(bound, end) for bound in single_bounds])
        limits = np.array([getattr(bound, f"ci_{end}") for bound in single_bounds])
        center, error = median_rule(edges, np.abs(edges - limits) / quantile)
        side = -1.0 if end == "lower" else 1.0
        assert getattr(combined, f"ci_{end}") == pytest.approx(
            center + side * quantile * error, rel=1e-10
        )

    def test_dropping_the_displacement_term_moves_the_limit(
        self, repeated: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mutation control: the between-draw term of the variance is nonzero."""
        shipped = omitted_variable_bounds(repeated, "ate", cf_y=CF, cf_d=CF).ci_lower

        def within_only(points: np.ndarray, std_errors: np.ndarray) -> tuple[float, float]:
            return float(np.median(points)), float(np.sqrt(np.median(std_errors**2)))

        monkeypatch.setattr(ov, "_median_rule", within_only)
        mutated = omitted_variable_bounds(repeated, "ate", cf_y=CF, cf_d=CF).ci_lower
        assert abs(mutated - shipped) > 1e-6

    def test_reading_the_first_draw_alone_moves_the_bound(
        self, repeated: Any, single_bounds: list[Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mutation control: the median is not the first draw."""

        def first(points: np.ndarray, std_errors: np.ndarray) -> tuple[float, float]:
            return float(points[0]), float(std_errors[0])

        monkeypatch.setattr(ov, "_median_rule", first)
        mutated = omitted_variable_bounds(repeated, "ate", cf_y=CF, cf_d=CF).lower
        assert mutated == pytest.approx(single_bounds[0].lower, rel=1e-12)
        assert mutated != pytest.approx(float(np.median([b.lower for b in single_bounds])))


class TestTheRobustnessValuesSolveTheCombinedBound:
    def test_the_point_value_moves_the_median_bound_to_the_null(self, repeated: Any) -> None:
        rv = robustness_value(repeated, "ate")["rv"]
        assert rv is not None and 0.0 < rv < 1.0
        at = omitted_variable_bounds(repeated, "ate", cf_y=rv, cf_d=rv)
        assert at.lower == pytest.approx(0.0, abs=1e-10)

    def test_the_limit_value_moves_the_median_limit_to_the_null(self, repeated: Any) -> None:
        rva = robustness_value(repeated, "ate")["rva"]
        assert rva is not None and 0.0 < rva < 1.0
        at = omitted_variable_bounds(repeated, "ate", cf_y=rva, cf_d=rva)
        assert at.ci_lower == pytest.approx(0.0, abs=1e-10)

    def test_the_median_point_value_is_the_median_draws(
        self, repeated: Any, single_draws: list[Any]
    ) -> None:
        """With an odd count, the median bound meets the null where the median draw does."""
        values = [robustness_value(single, "ate")["rv"] for single in single_draws]
        assert robustness_value(repeated, "ate")["rv"] == pytest.approx(
            float(np.median(values)), rel=1e-8
        )


def test_each_contour_cell_is_the_combined_bound(repeated: Any) -> None:
    frame = contour_data(repeated, "ate", grid_size=3)
    for cf_d, cf_y, value in zip(frame["cf_d"], frame["cf_y"], frame["value"], strict=True):
        bound = omitted_variable_bounds(repeated, "ate", cf_y=float(cf_y), cf_d=float(cf_d))
        assert value == pytest.approx(bound.lower, rel=1e-12, abs=1e-15)


class TestOneDrawIsTheShippedBound:
    def test_the_bound_reads_the_reported_estimate_and_curve(self) -> None:
        result = fit(repeats=1)
        (draw,) = ov._bound_draws(result, "ate", "auto")
        assert draw.psi == result["ate"].psi
        assert draw.curve is result["ate"].influence_curve
        elements = sensitivity_elements(result, "ate")
        bound = omitted_variable_bounds(result, "ate", cf_y=CF, cf_d=CF)
        strength = ov._confounding_strength(CF, CF, 1.0)
        assert bound.lower == result["ate"].psi - strength * elements.max_bias
        assert bound.features == ()
        named = sensitivity_elements(result, "ate", repeat=0)
        assert (named.sigma2, named.nu2) == (elements.sigma2, elements.nu2)


class TestTheElementsNameADraw:
    def test_a_bare_request_is_refused(self, repeated: Any) -> None:
        with pytest.raises(CapabilityError, match="repeat=0 to repeat=2"):
            sensitivity_elements(repeated, "ate")

    @pytest.mark.parametrize("repeat", [3, -1, True, 1.0])
    def test_a_draw_the_fit_lacks_is_a_value_error(self, repeated: Any, repeat: Any) -> None:
        with pytest.raises(ValueError, match="repeat must") as raised:
            sensitivity_elements(repeated, "ate", repeat=repeat)
        assert type(raised.value) is ValueError

    def test_the_row_defers_on_the_draw(self, repeated: Any) -> None:
        row = repeated.sensitivity.capability("elements")
        assert row.available
        assert "repeat" in row.requires_arguments
        assert repeated.sensitivity.capability("omitted_confounding").status is (
            AssessmentStatus.PASSED
        )


class TestTheLimitsStayDescriptive:
    def test_the_rule_is_repeats(self, repeated: Any) -> None:
        bound = omitted_variable_bounds(repeated, "ate")
        assert bound.features == ("repeats",)
        assert bound.limit_descriptive
        assert robustness_value(repeated, "ate")["limit_licensed"] is False

    def test_the_plugin_limits_refuse_by_configuration(self, repeated: Any) -> None:
        elements = sensitivity_elements(repeated, "ate", nu2_estimator="plugin", repeat=0)
        assert elements.limit_refusal is not None
        assert elements.limit_refusal.key == "configuration"
        assert "median rule" in elements.limit_refusal.text


class TestTheRetargetGuard:
    def test_a_draw_whose_retarget_disagrees_is_a_defect(self, repeated: Any) -> None:
        """Mutation control: the retarget must reproduce the stored draw."""
        first = repeated.repeats[0]
        tampered = replace(first, psi={**first.psi, "ate": first.psi["ate"] + 0.01})
        broken = replace(repeated, repeats=(tampered, *repeated.repeats[1:]))
        with pytest.raises(RuntimeError, match="package defect"):
            omitted_variable_bounds(broken, "ate")


def test_the_benchmark_compares_the_median_second_moments() -> None:
    result = fit(repeats=2)
    record = benchmark(result, ["W1"], estimand="ate")
    long = [sensitivity_elements(result, "ate", repeat=r) for r in range(2)]
    assert record.sigma2_long == pytest.approx(float(np.median([e.sigma2 for e in long])))
    assert record.nu2_long == pytest.approx(float(np.median([e.nu2 for e in long])))
    assert record.features == ("repeats",)
    assert 0.0 <= record.cf_y <= 1.0 and 0.0 <= record.cf_d <= 1.0
