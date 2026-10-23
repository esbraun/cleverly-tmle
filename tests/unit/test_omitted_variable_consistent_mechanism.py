"""The point bound of a DR-TMLE or C-TMLE fit under ``assume_consistent_mechanism`` (X82 part G).

By default the bound refuses both fits: neither estimator assumes a consistent treatment
mechanism, and the published bound's ``nu^2`` is derived under that assumption.  The
keyword states the assumption and lifts that one refusal.  The witnesses below rest on
two facts.

* A DR-TMLE fit stores its primary mechanism, which the ordinary TMLE with the same
  learner and seed also fits.  So the two fits' ``nu^2`` agree.
* A C-TMLE fit selects its working mechanism.  The bound refits the fit's treatment
  learner on every covariate, which is the ordinary TMLE's mechanism.  So the refit's
  ``nu^2`` agrees with the ordinary fit's, and differs from the selected mechanism's.

Each bound is a point bound.  Its limits refuse, because no cited source gives their
standard error.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from cleverly.datasets import make_binary_outcome, make_instrument
from cleverly.estimators import CTMLE, DRTMLE
from cleverly.exceptions import CapabilityError, RefusalReason
from cleverly.learners._l1_logistic import L1Logistic
from cleverly.sensitivity import omitted_variable as ov
from cleverly.sensitivity.omitted_variable import (
    contour_data,
    fit_wide_bound_refusal,
    omitted_variable_bounds,
    robustness_value,
    sensitivity_elements,
)
from tests.conftest import FAST_KWARGS, IN_SAMPLE, fast_tmle
from tests.unit.test_outcome_adaptive_per_arm_extensions import (
    BINARY_ACTIVE,
    extended_frame,
    per_arm_fit,
    roles,
)

CF = 0.04


def _fit(estimator: Any, frame: Any, **roles: Any) -> Any:
    return estimator.fit(frame, outcome="Y", treatment="A", **roles).single()


@pytest.fixture(scope="module")
def binary_frame() -> Any:
    return make_binary_outcome(n=400, seed=17)[0]


@pytest.fixture(scope="module")
def drtmle_pair(binary_frame: Any) -> tuple[Any, Any]:
    """The ordinary and the DR-TMLE fit of one frame, with one learner and seed."""
    plain = _fit(fast_tmle(**IN_SAMPLE, estimands=("ate",)), binary_frame)
    guarded = _fit(DRTMLE(**FAST_KWARGS, **IN_SAMPLE, estimands=("ate",)), binary_frame)
    return plain, guarded


@pytest.fixture(scope="module")
def ctmle_pair() -> tuple[Any, Any]:
    """The ordinary and the C-TMLE fit of one instrument law, with one learner and seed."""
    frame, _ = make_instrument(n=600, seed=44)
    plain = _fit(fast_tmle(**IN_SAMPLE, estimands=("ate",)), frame)
    collaborative = _fit(
        CTMLE(**FAST_KWARGS, **IN_SAMPLE, estimands=("ate",), selection_folds=3), frame
    )
    return plain, collaborative


def _elements(result: Any) -> Any:
    (draw,) = ov._consistent_draws(result, "ate", "auto")
    return draw.elements


def _bound(result: Any, **kwargs: Any) -> Any:
    return omitted_variable_bounds(
        result, "ate", cf_y=CF, cf_d=CF, assume_consistent_mechanism=True, **kwargs
    )


class TestTheDefaultStillRefuses:
    @pytest.mark.parametrize("pair", ["drtmle_pair", "ctmle_pair"])
    def test_the_refusal_names_the_keyword(self, request: pytest.FixtureRequest, pair: str) -> None:
        result = request.getfixturevalue(pair)[1]
        refusal = fit_wide_bound_refusal(result)
        assert refusal is not None
        assert "assume_consistent_mechanism=True" in refusal.message()
        with pytest.raises(CapabilityError, match="no nu\\^2 estimate"):
            omitted_variable_bounds(result, "ate", cf_y=CF, cf_d=CF)


class TestTheDRTMLEPointBound:
    def test_nu2_is_the_ordinary_fits(self, drtmle_pair: tuple[Any, Any]) -> None:
        """The primary mechanism is the ordinary TMLE's, so its ``nu^2`` is too."""
        plain, guarded = drtmle_pair
        assert _elements(guarded).nu2 == pytest.approx(
            sensitivity_elements(plain, "ate").nu2, rel=1e-12
        )

    def test_the_bound_centres_on_the_fit_and_spans_the_bias(
        self, drtmle_pair: tuple[Any, Any]
    ) -> None:
        guarded = drtmle_pair[1]
        bound = _bound(guarded)
        elements = _elements(guarded)
        strength = ov._confounding_strength(CF, CF, 1.0)
        assert bound.psi == guarded["ate"].psi
        assert bound.max_bias == pytest.approx(np.sqrt(elements.sigma2 * elements.nu2))
        assert bound.lower == pytest.approx(guarded["ate"].psi - strength * bound.max_bias)
        assert bound.upper == pytest.approx(guarded["ate"].psi + strength * bound.max_bias)
        assert bound.max_bias > 0

    def test_sigma2_reads_the_fits_own_outcome_regression(
        self, drtmle_pair: tuple[Any, Any]
    ) -> None:
        """Nonzero witness: the DR-TMLE update moves ``Q``, so ``sigma^2`` is the fit's."""
        plain, guarded = drtmle_pair
        assert _elements(guarded).sigma2 != pytest.approx(
            sensitivity_elements(plain, "ate").sigma2, rel=1e-9
        )

    def test_the_assumption_is_stated(self, drtmle_pair: tuple[Any, Any]) -> None:
        bound = _bound(drtmle_pair[1])
        assert bound.assumption is not None
        assert "primary treatment mechanism" in bound.assumption
        assert bound.assumption in bound.summary()
        assert bound.to_dict()["assumption"] == bound.assumption

    def test_the_point_robustness_value_moves_the_bound_to_the_null(
        self, drtmle_pair: tuple[Any, Any]
    ) -> None:
        guarded = drtmle_pair[1]
        values = robustness_value(guarded, "ate", assume_consistent_mechanism=True)
        rv = values["rv"]
        assert rv is not None and 0.0 < rv < 1.0
        at = omitted_variable_bounds(
            guarded, "ate", cf_y=rv, cf_d=rv, assume_consistent_mechanism=True
        )
        assert min(abs(at.lower), abs(at.upper)) == pytest.approx(0.0, abs=1e-10)
        assert "rva" not in values
        assert "point bound" in values["limit_refusal"]

    def test_the_contour_is_the_bound(self, drtmle_pair: tuple[Any, Any]) -> None:
        guarded = drtmle_pair[1]
        frame = contour_data(guarded, "ate", grid_size=3, assume_consistent_mechanism=True)
        for cf_d, cf_y, value in zip(frame["cf_d"], frame["cf_y"], frame["value"], strict=True):
            bound = omitted_variable_bounds(
                guarded, "ate", cf_y=float(cf_y), cf_d=float(cf_d), assume_consistent_mechanism=True
            )
            assert value == pytest.approx(bound.lower, rel=1e-12, abs=1e-15)

    def test_a_missing_treatment_reads_the_composite_mechanism(self) -> None:
        """The DR-TMLE primary mechanism of a missing treatment is the ordinary composite one."""
        frame = extended_frame(BINARY_ACTIVE, 500, 93, treatment_delta=True)
        plain = _fit(fast_tmle(**IN_SAMPLE, estimands=("ate",)), frame, **roles(frame))
        guarded = _fit(
            DRTMLE(**FAST_KWARGS, **IN_SAMPLE, estimands=("ate",)), frame, **roles(frame)
        )
        assert _bound(guarded).features == ("missing_treatment",)
        assert _elements(guarded).nu2 == pytest.approx(
            sensitivity_elements(plain, "ate").nu2, rel=1e-12
        )


class TestTheCTMLEPointBound:
    def test_the_refit_nu2_is_the_ordinary_fits(self, ctmle_pair: tuple[Any, Any]) -> None:
        """The refit on every covariate is the ordinary TMLE's mechanism."""
        plain, collaborative = ctmle_pair
        assert _elements(collaborative).nu2 == pytest.approx(
            sensitivity_elements(plain, "ate").nu2, rel=1e-10
        )

    def test_the_selected_mechanism_gives_another_nu2(
        self, ctmle_pair: tuple[Any, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mutation control: without the refit, the bound reads the selected mechanism."""
        plain, collaborative = ctmle_pair
        monkeypatch.setattr(
            ov,
            "_full_mechanism",
            lambda result, repeat, learner=None: repeat.nuisance.propensity,
        )
        mutated = _elements(collaborative).nu2
        assert mutated != pytest.approx(sensitivity_elements(plain, "ate").nu2, rel=1e-3)

    def test_the_assumption_names_the_refit(self, ctmle_pair: tuple[Any, Any]) -> None:
        bound = _bound(ctmle_pair[1])
        assert bound.assumption is not None
        assert "not the selected mechanism" in bound.assumption
        assert bound.psi == ctmle_pair[1]["ate"].psi

    def test_a_missing_treatment_refuses(self) -> None:
        """No refit of the composite mechanism of a missing treatment is implemented."""
        frame = extended_frame(BINARY_ACTIVE, 500, 93, delta=True, treatment_delta=True)
        collaborative = per_arm_fit(frame)
        assert fit_wide_bound_refusal(collaborative) is ov._CTMLE_BOUND_REFUSAL
        with pytest.raises(CapabilityError, match="missing treatment") as raised:
            _bound(collaborative)
        assert raised.value.refusal.reason is RefusalReason.IMPLEMENTATION_GAP


class TestTheLassoFullMechanism:
    """A lasso fit never uses ``treatment_learner``: its full mechanism is the path's own."""

    @pytest.fixture(scope="class")
    def lasso_pair(self) -> tuple[Any, Any]:
        frame, _ = make_instrument(n=600, seed=44)
        plain = _fit(
            fast_tmle(**IN_SAMPLE, estimands=("ate",), treatment_learner=L1Logistic(0.0)), frame
        )
        lasso = _fit(
            CTMLE(
                **FAST_KWARGS,
                **IN_SAMPLE,
                estimands=("ate",),
                strategy="lasso",
                lambdas=(0.2, 0.05, 0.02, 0.0),
            ),
            frame,
        )
        return plain, lasso

    def test_the_refit_is_the_unpenalized_member(self, lasso_pair: tuple[Any, Any]) -> None:
        plain, lasso = lasso_pair
        assert fit_wide_bound_refusal(lasso) is ov._CTMLE_BOUND_REFUSAL
        assert _elements(lasso).nu2 == pytest.approx(
            sensitivity_elements(plain, "ate").nu2, rel=1e-8
        )
        bound = _bound(lasso)
        assert bound.assumption is not None and "lambda = 0" in bound.assumption

    def test_refitting_the_treatment_learner_moves_nu2(
        self, lasso_pair: tuple[Any, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mutation control: the declared ``treatment_learner`` is another mechanism."""
        plain, lasso = lasso_pair
        monkeypatch.setattr(ov, "_lasso_fit", lambda result: False)
        assert _elements(lasso).nu2 != pytest.approx(
            sensitivity_elements(plain, "ate").nu2, rel=1e-6
        )


def test_a_fit_that_selects_nothing_keeps_its_redirection() -> None:
    """The keyword lifts the selection refusal only; this fit is the ordinary TMLE fit."""
    frame, _ = make_instrument(n=600, seed=44)
    nothing = _fit(
        CTMLE(
            **FAST_KWARGS,
            **IN_SAMPLE,
            estimands=("ate",),
            strategy="learners",
            treatment_candidates=[],
        ),
        frame,
    )
    refusal = fit_wide_bound_refusal(nothing)
    assert refusal is ov._CTMLE_FULL_CANDIDATE_BOUND_REFUSAL
    with pytest.raises(CapabilityError) as raised:
        _bound(nothing)
    assert raised.value.refusal == refusal


class TestTheLimitsRefuse:
    @pytest.mark.parametrize("pair", ["drtmle_pair", "ctmle_pair"])
    def test_every_limit_reading_is_absent(self, request: pytest.FixtureRequest, pair: str) -> None:
        bound = _bound(request.getfixturevalue(pair)[1])
        assert bound.limit_refusal is not None
        assert bound.limit_refusal.key == "consistent_mechanism"
        assert bound.limit_refusal.reason is RefusalReason.INFERENCE_GAP
        assert "point bound" in bound.limit_refusal.text
        for accessor in ("ci_lower", "ci_upper"):
            with pytest.raises(CapabilityError):
                getattr(bound, accessor)

    def test_the_drtmle_limit_names_the_point_bound(self, drtmle_pair: tuple[Any, Any]) -> None:
        """A DR-TMLE fit reports an interval, so the bound's own refusal is the one heard."""
        with pytest.raises(CapabilityError, match="is a point bound") as raised:
            _ = _bound(drtmle_pair[1]).ci_lower
        assert raised.value.refusal.reason is RefusalReason.INFERENCE_GAP


class TestTheKeywordOnlyLiftsItsRule:
    def test_an_ordinary_fit_is_a_value_error(self, drtmle_pair: tuple[Any, Any]) -> None:
        with pytest.raises(ValueError, match="Omit the argument") as raised:
            _bound(drtmle_pair[0])
        assert type(raised.value) is ValueError

    def test_the_default_bound_has_no_assumption(self, drtmle_pair: tuple[Any, Any]) -> None:
        bound = omitted_variable_bounds(drtmle_pair[0], "ate", cf_y=CF, cf_d=CF)
        assert bound.assumption is None
        assert "assumption" not in bound.to_dict()


def test_repeats_combine_the_point_bounds_by_the_median(binary_frame: Any) -> None:
    """Each draw keeps its own estimate, and the combined bound is the median."""
    guarded = _fit(DRTMLE(**FAST_KWARGS, estimands=("ate",), repeats=3), binary_frame)
    draws = ov._consistent_draws(guarded, "ate", "auto")
    assert len(draws) == 3
    strength = ov._confounding_strength(CF, CF, 1.0)
    lowers = [draw.psi - strength * draw.elements.max_bias for draw in draws]
    assert len({round(value, 12) for value in lowers}) == 3
    bound = _bound(guarded)
    assert bound.lower == pytest.approx(float(np.median(lowers)), rel=1e-12)
    assert bound.limit_refusal is not None


class _RecordingLogit(LogisticRegression):
    """A logistic mechanism that records the width of every design it is fitted on."""

    widths: ClassVar[list[int]] = []

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> Any:
        type(self).widths.append(int(np.shape(X)[1]))
        return super().fit(X, y, sample_weight=sample_weight)


class TestThePerArmParametricPointBound:
    """A per-arm fit's treatment learner reads one column, so the bound needs a declared one."""

    @pytest.fixture(scope="class")
    def frame(self) -> Any:
        return extended_frame(BINARY_ACTIVE, 500, 93)

    @pytest.fixture(scope="class")
    def collaborative(self, frame: Any) -> Any:
        return per_arm_fit(frame)

    @pytest.fixture(scope="class")
    def plain(self, frame: Any) -> Any:
        """The ordinary TMLE with the declared learner as its mechanism, in sample."""
        return _fit(
            fast_tmle(
                **IN_SAMPLE,
                estimands=("ate",),
                treatment_learner=LogisticRegression(max_iter=1000),
                g_bounds=(0.01, 0.99),
            ),
            frame,
        )

    def test_without_a_learner_the_bound_refuses(self, collaborative: Any) -> None:
        with pytest.raises(CapabilityError) as raised:
            omitted_variable_bounds(
                collaborative, "ate", cf_y=CF, cf_d=CF, assume_consistent_mechanism=True
            )
        assert raised.value.refusal.reason is RefusalReason.INVALID_DECLARATION
        assert "per-arm" in str(raised.value)
        alternative = raised.value.refusal.alternative
        assert "mechanism_learner=" in alternative
        assert "ordinary TMLE" in alternative

    def test_the_declared_learner_gives_the_ordinary_nu2(
        self, collaborative: Any, plain: Any
    ) -> None:
        """Nonzero witness: the representer is the declared mechanism's, through the g bounds.

        The ordinary fit clips its mechanism to the same declared ``g_bounds``, so equal
        ``nu^2`` under both estimators also witnesses the clipping.
        """
        for estimator in ("doubly_robust", "plugin"):
            (draw,) = ov._consistent_draws(
                collaborative, "ate", estimator, LogisticRegression(max_iter=1000)
            )
            expected = sensitivity_elements(plain, "ate", nu2_estimator=estimator).nu2
            assert draw.elements.nu2 == pytest.approx(expected, rel=1e-6), estimator
            assert draw.elements.nu2 > 0.0

    def test_the_learner_reads_every_adjustment_column(self, collaborative: Any) -> None:
        _RecordingLogit.widths.clear()
        ov._consistent_draws(collaborative, "ate", "auto", _RecordingLogit(max_iter=1000))
        assert _RecordingLogit.widths
        assert set(_RecordingLogit.widths) == {collaborative.data.covariates.shape[1]}
        assert collaborative.data.covariates.shape[1] > 1

    def test_an_intercept_only_learner_moves_nu2(self, collaborative: Any) -> None:
        """Mutation control: the declared learner is the one the representer reads."""
        from sklearn.dummy import DummyClassifier

        (fitted,) = ov._consistent_draws(
            collaborative, "ate", "auto", LogisticRegression(max_iter=1000)
        )
        (marginal,) = ov._consistent_draws(
            collaborative, "ate", "auto", DummyClassifier(strategy="prior")
        )
        assert abs(marginal.elements.nu2 / fitted.elements.nu2 - 1.0) > 0.10

    def test_the_bound_records_and_states_the_learner(self, collaborative: Any) -> None:
        bound = omitted_variable_bounds(
            collaborative,
            "ate",
            cf_y=CF,
            cf_d=CF,
            assume_consistent_mechanism=True,
            mechanism_learner=LogisticRegression(max_iter=1000),
        )
        assert bound.mechanism_learner is not None
        assert "LogisticRegression" in bound.mechanism_learner
        assert bound.to_dict()["mechanism_learner"] == bound.mechanism_learner
        assert bound.assumption is not None
        assert "declared mechanism learner" in bound.assumption
        assert "consistent for the mechanism" in bound.assumption
        assert bound.limit_refusal is not None
        assert bound.psi == collaborative["ate"].psi


class TestTheDeclaredMechanismLearner:
    def test_the_fit_s_own_learner_reproduces_the_default(
        self, ctmle_pair: tuple[Any, Any]
    ) -> None:
        """Equality witness: the keyword reaches the refit and nothing else."""
        collaborative = ctmle_pair[1]
        default = _bound(collaborative)
        declared = omitted_variable_bounds(
            collaborative,
            "ate",
            cf_y=CF,
            cf_d=CF,
            assume_consistent_mechanism=True,
            mechanism_learner=collaborative.estimator.treatment_learner,
        )
        assert (declared.lower, declared.upper, declared.max_bias) == (
            default.lower,
            default.upper,
            default.max_bias,
        )
        assert default.mechanism_learner is None

    def test_it_needs_the_consistent_mechanism_flag(self, ctmle_pair: tuple[Any, Any]) -> None:
        for call in (omitted_variable_bounds, robustness_value, contour_data):
            with pytest.raises(ValueError, match="assume_consistent_mechanism=True"):
                call(ctmle_pair[1], "ate", mechanism_learner=LogisticRegression())

    def test_a_drtmle_or_tmle_fit_rejects_it(self, drtmle_pair: tuple[Any, Any]) -> None:
        plain, guarded = drtmle_pair
        with pytest.raises(ValueError, match="refits nothing"):
            omitted_variable_bounds(
                guarded,
                "ate",
                assume_consistent_mechanism=True,
                mechanism_learner=LogisticRegression(),
            )
        with pytest.raises(ValueError, match="refits nothing"):
            omitted_variable_bounds(
                plain,
                "ate",
                assume_consistent_mechanism=True,
                mechanism_learner=LogisticRegression(),
            )
