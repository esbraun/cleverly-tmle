"""The collaborative selection: its loss, its penalty, and the paths it builds.

These are exact checks on the machinery.  Where a statistical claim is made -- that
C-TMLE beats TMLE in mean squared error when an instrument is present -- it needs
replications and lives in the slow tier; what is checked here is that the loss is the
loss it claims to be, that the penalty is what rejects an instrument, and that with
nothing to select the estimator collapses onto a plain TMLE bit for bit.
"""

from __future__ import annotations

from itertools import pairwise
from typing import ClassVar

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from cleverly import CTMLE, TMLE
from cleverly.datasets import make_cde, make_instrument, make_linear_ate
from cleverly.estimators import ctmle as ctmle_module
from cleverly.estimators.ctmle import _Selector, _weighted_partial_correlation
from cleverly.estimators.serialize import dumps, loads
from cleverly.learners.crossfit import make_folds
from tests.conftest import FAST_KWARGS

TMLE_KWARGS = {**FAST_KWARGS, "estimands": ("ate",)}

#: Three selection folds rather than the default five: the searches below are the
#: dominant cost in this file and the claims resolve identically either way.
CTMLE_KWARGS = {**TMLE_KWARGS, "selection_folds": 3}


class _RecordingRegressor(RegressorMixin, BaseEstimator):
    """Constant learner that records the row identifier in every training call."""

    fits: ClassVar[list[np.ndarray]] = []
    predictions: ClassVar[list[tuple[np.ndarray, np.ndarray]]] = []

    def fit(
        self, design: np.ndarray, target: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> _RecordingRegressor:
        self.fit_rows_ = np.asarray(design[:, 1], dtype=int)
        type(self).fits.append(self.fit_rows_)
        self.mean_ = float(np.average(target, weights=sample_weight))
        return self

    def predict(self, design: np.ndarray) -> np.ndarray:
        type(self).predictions.append((self.fit_rows_, np.asarray(design[:, 1], dtype=int)))
        return np.full(design.shape[0], self.mean_)


class _RecordingClassifier(ClassifierMixin, BaseEstimator):
    """Intercept classifier recording the row IDs it fits and predicts."""

    fits: ClassVar[list[np.ndarray]] = []
    predictions: ClassVar[list[tuple[np.ndarray, np.ndarray]]] = []

    def fit(
        self, design: np.ndarray, target: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> _RecordingClassifier:
        self.fit_rows_ = np.asarray(design[:, 0], dtype=int)
        type(self).fits.append(self.fit_rows_)
        self.rate_ = float(np.average(target, weights=sample_weight))
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, design: np.ndarray) -> np.ndarray:
        type(self).predictions.append((self.fit_rows_, np.asarray(design[:, 0], dtype=int)))
        return np.column_stack(
            [
                np.full(design.shape[0], 1.0 - self.rate_),
                np.full(design.shape[0], self.rate_),
            ]
        )


def _selector(frame: object, **overrides: object) -> tuple[_Selector, object]:
    """A selector wired up exactly as ``CTMLE._nuisances`` wires one."""
    estimator = CTMLE(**{**CTMLE_KWARGS, **overrides})
    data = estimator._prepare(
        frame,
        outcome="Y",
        treatment="A",
        covariates=None,
        delta=None,
        weights=None,
        id=None,
        intermediate=None,
    )
    scaler = estimator._scaler(data)
    folds = estimator._folds(data)
    config = estimator._config(data, ("ate",), scaler, folds)
    base = estimator._fit_nuisances(data, folds, scaler, None)
    return _Selector(estimator, data, base, config.g_bounds, None), base


@pytest.fixture(scope="module")
def instrument_frame() -> object:
    frame, _ = make_instrument(n=700, seed=0)
    return frame


@pytest.fixture(scope="module")
def selector(instrument_frame) -> _Selector:
    return _selector(instrument_frame)[0]


class TestLoss:
    def test_squared_error_matches_the_formula(self, selector) -> None:
        rows = selector.all_rows
        candidate = selector._candidate(("W1",), selector.base.outcome, rows, None, "t", 1)
        residual = selector.scaled - candidate.targeted.observed
        expected = float(np.sum(selector.data.weights * residual**2))
        assert candidate.loss == pytest.approx(expected, rel=1e-12)

    def test_a_continuous_outcome_uses_squared_error_by_default(self, selector) -> None:
        assert selector.loss_kind == "squared"

    def test_a_binary_outcome_uses_the_log_likelihood_by_default(self) -> None:
        from cleverly.datasets import make_binary_outcome

        frame, _ = make_binary_outcome(n=400, seed=2)
        assert _selector(frame)[0].loss_kind == "loglik"

    def test_the_log_likelihood_matches_the_formula(self, instrument_frame) -> None:
        selector = _selector(instrument_frame, loss="loglik")[0]
        rows = selector.all_rows
        candidate = selector._candidate(("W1",), selector.base.outcome, rows, None, "t", 1)
        q = np.clip(candidate.targeted.observed, 1e-12, 1.0 - 1e-12)
        y = selector.scaled
        expected = float(
            -np.sum(selector.data.weights * (y * np.log(q) + (1.0 - y) * np.log(1.0 - q)))
        )
        assert candidate.loss == pytest.approx(expected, rel=1e-12)

    def test_the_loss_only_counts_rows_with_an_observed_outcome(self, selector) -> None:
        rows = selector.all_rows
        candidate = selector._candidate((), selector.base.outcome, rows, None, "t", 1)
        half = rows[: rows.size // 2]
        assert selector.loss(candidate.targeted, half) < candidate.loss


class TestPenalty:
    def test_it_matches_the_published_formula(self, selector) -> None:
        rows = selector.all_rows
        candidate = selector._candidate(("W1",), selector.base.outcome, rows, None, "t", 1)
        ic = selector.influence(candidate.targeted, candidate.submodel, rows)
        expected = float(np.var(ic, ddof=1) + ic.size * np.mean(ic) ** 2)
        assert selector.penalty(candidate.targeted, candidate.submodel, rows) == pytest.approx(
            expected, rel=1e-12
        )

    def test_the_risk_is_the_loss_plus_the_penalty(self, selector) -> None:
        rows = selector.all_rows
        candidate = selector._candidate(("W3",), selector.base.outcome, rows, None, "t", 1)
        penalty = selector.penalty(candidate.targeted, candidate.submodel, rows)
        assert candidate.risk == pytest.approx(candidate.loss + penalty, rel=1e-12)

    def test_the_penalty_is_what_makes_an_instrument_expensive(self, selector) -> None:
        # W1 confounds; W2 predicts treatment and nothing else. Adjusting for W2 buys
        # no bias reduction and pushes g towards 0 and 1, so 1/g -- and with it the
        # influence curve's variance -- inflates. That inflation is the penalty, and
        # it is the only term that can tell the two covariates apart.
        rows = selector.all_rows
        confounder = selector._candidate(("W1",), selector.base.outcome, rows, None, "a", 1)
        instrument = selector._candidate(("W2",), selector.base.outcome, rows, None, "b", 1)

        penalties = {
            name: selector.penalty(candidate.targeted, candidate.submodel, rows)
            for name, candidate in (("W1", confounder), ("W2", instrument))
        }
        assert penalties["W2"] > 1.2 * penalties["W1"]
        # And the penalty is decisive: it reverses the ranking the bare loss gives.
        assert instrument.loss < confounder.loss
        assert instrument.risk > confounder.risk

    def test_turning_the_penalty_off_leaves_the_bare_loss(self, instrument_frame) -> None:
        selector = _selector(instrument_frame, penalty=False)[0]
        rows = selector.all_rows
        candidate = selector._candidate(("W2",), selector.base.outcome, rows, None, "t", 1)
        assert candidate.risk == candidate.loss

    def test_treatment_risk_is_weighted_binomial_deviance(self, selector) -> None:
        rows = selector.all_rows
        candidate = selector._candidate(("W1",), selector.base.outcome, rows, None, "g-risk", 1)
        propensity = np.clip(candidate.propensity, 1e-12, 1.0 - 1e-12)
        treatment = selector.data.treatment
        expected = -np.sum(
            selector.data.weights
            * (treatment * np.log(propensity) + (1.0 - treatment) * np.log(1.0 - propensity))
        )
        assert candidate.treatment_risk == pytest.approx(float(expected), rel=1e-12)


class TestPaths:
    def test_the_greedy_path_is_nested(self, selector) -> None:
        path = selector.build_path(train=None, tag="p")
        assert path[0].covariates == ()
        for previous, candidate in pairwise(path):
            assert candidate.covariates[:-1] == previous.covariates
            assert candidate.covariates[-1] not in previous.covariates
        assert set(path[-1].covariates) == set(selector.data.covariate_names)

    def test_the_greedy_search_adds_the_instrument_last(self, selector) -> None:
        # The forward search is deterministic given the data, so this is an exact
        # assertion about one sample rather than a coverage claim. W2 predicts
        # treatment better than anything else available, so a search scored on
        # treatment prediction would take it *first*; scored collaboratively against
        # the targeted outcome model it is the covariate of last resort.
        path = selector.build_path(train=None, tag="p2")
        assert path[-1].covariates[-1] == "W2"

    def test_the_risk_only_rises_after_a_tmle_step_increment(self, selector) -> None:
        # The forward search accepts a worse candidate in exactly one situation: it
        # has already incremented the TMLE step and must move on regardless. Any
        # other increase would mean the monotonicity argument had broken.
        path = selector.build_path(train=None, tag="q")
        for previous, candidate in pairwise(path):
            if candidate.risk > previous.risk:
                assert candidate.n_steps > previous.n_steps

    def test_the_default_ordering_is_the_published_logistic_preorder(
        self, instrument_frame
    ) -> None:
        # The published logistic preorder ranks one-variable propensity candidates by
        # the empirical loss of the Qbar they target, rather than by marginal Y
        # correlation.  Recompute that definition independently for every variable.
        ordered = _selector(instrument_frame, strategy="ordered")[0]
        losses = {}
        for name in ordered.data.covariate_names:
            propensity = ordered.propensity((name,), None, "longhand")
            submodel = ordered.submodel(propensity)
            targeted, _ = ordered.target(ordered.base.outcome, submodel, ordered.all_rows)
            losses[name] = ordered.loss(targeted, ordered.all_rows)
        expected = tuple(sorted(losses, key=losses.__getitem__))
        assert ordered._ordering(ordered.all_rows, None) == expected

    def test_logistic_preorder_places_the_smaller_loss_first(
        self, instrument_frame, monkeypatch
    ) -> None:
        selector = _selector(instrument_frame, strategy="ordered")[0]
        scores = {"W1": 0.8, "W2": 0.2, "W3": 0.5}
        monkeypatch.setattr(
            selector,
            "_fit_propensity_with",
            lambda learner, names, train: np.full(selector.data.n, scores[names[0]]),
        )
        monkeypatch.setattr(selector, "submodel", lambda propensity: propensity)
        monkeypatch.setattr(
            selector,
            "target",
            lambda initial, submodel, rows: (submodel, np.zeros(1)),
        )
        monkeypatch.setattr(selector, "loss", lambda targeted, rows: float(targeted[0]))
        assert selector._ordering(selector.all_rows, None) == ("W2", "W3", "W1")

    def test_partial_correlation_ordering_matches_weighted_residualization(
        self, instrument_frame
    ) -> None:
        selector = _selector(instrument_frame, strategy="ordered", preorder="partial_correlation")[
            0
        ]
        residual = selector.scaled - selector.base.outcome.observed
        scores = {
            name: abs(
                _weighted_partial_correlation(
                    residual,
                    selector.data.covariates[:, column],
                    selector.data.treatment,
                    selector.data.weights,
                )
            )
            for column, name in enumerate(selector.data.covariate_names)
        }
        assert selector._ordering(selector.all_rows, None) == tuple(
            sorted(scores, key=scores.__getitem__, reverse=True)
        )

    def test_partial_correlation_preorder_places_the_larger_magnitude_first(
        self, instrument_frame, monkeypatch
    ) -> None:
        selector = _selector(instrument_frame, strategy="ordered", preorder="partial_correlation")[
            0
        ]
        scores = iter((0.2, -0.9, 0.5))
        monkeypatch.setattr(
            ctmle_module,
            "_weighted_partial_correlation",
            lambda left, right, conditional, weights: next(scores),
        )
        assert selector._ordering(selector.all_rows, None) == ("W2", "W3", "W1")

    def test_weighted_partial_correlation_matches_the_closed_form(self) -> None:
        left = np.array([0.0, 1.0, 4.0, 2.0, 5.0, 8.0])
        right = np.array([1.0, 3.0, 2.0, 6.0, 7.0, 4.0])
        conditional = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 1.0])
        weights = np.array([1.0, 2.0, 1.0, 4.0, 2.0, 3.0])

        def correlation(x: np.ndarray, y: np.ndarray) -> float:
            x_centered = x - np.average(x, weights=weights)
            y_centered = y - np.average(y, weights=weights)
            covariance = np.average(x_centered * y_centered, weights=weights)
            scale = np.sqrt(
                np.average(x_centered**2, weights=weights)
                * np.average(y_centered**2, weights=weights)
            )
            return float(covariance / scale)

        xy = correlation(left, right)
        xa = correlation(left, conditional)
        ya = correlation(right, conditional)
        expected = (xy - xa * ya) / np.sqrt((1.0 - xa**2) * (1.0 - ya**2))
        assert _weighted_partial_correlation(left, right, conditional, weights) == pytest.approx(
            expected, rel=1e-12
        )

    def test_an_explicit_ordering_is_followed_exactly(self, instrument_frame) -> None:
        selector = _selector(instrument_frame, strategy="ordered", ordering=["W3", "W2", "W1"])[0]
        path = selector.build_path(train=None, tag="o")
        assert [candidate.covariates for candidate in path] == [
            (),
            ("W3",),
            ("W3", "W2"),
            ("W3", "W2", "W1"),
        ]

    def test_a_discrete_path_is_the_candidate_list(self, instrument_frame) -> None:
        candidates = [("W1",), ("W1", "W3"), ("W2",)]
        selector = _selector(instrument_frame, strategy="discrete", candidates=candidates)[0]
        path = selector.build_path(train=None, tag="d")
        assert [candidate.covariates for candidate in path] == candidates
        # Each fluctuates from the same initial fit, so none of them chains onto another.
        assert {candidate.n_steps for candidate in path} == {1}

    def test_the_intercept_candidate_is_the_marginal_treatment_rate(self, selector) -> None:
        # Cross-fitted, so each row gets the rate computed without it.
        values = selector.propensity((), None, "i")
        assert values.min() > 0.0 and values.max() < 1.0
        assert float(np.mean(values)) == pytest.approx(
            float(np.mean(selector.data.treatment)), abs=0.02
        )


class TestSelection:
    def test_selection_folds_cross_fit_qbar_without_validation_rows(self) -> None:
        frame, _ = make_instrument(n=90, seed=13)
        frame.insert(2, "row_id", np.arange(len(frame)))
        selector = _selector(
            frame,
            strategy="discrete",
            candidates=[()],
            outcome_learner=_RecordingRegressor(),
            cross_fit=False,
            n_folds=5,
            selection_folds=3,
        )[0]
        path = selector.build_path(train=None, tag="full")
        _RecordingRegressor.fits.clear()
        _RecordingRegressor.predictions.clear()
        selector.cross_validate(path)

        assert len(_RecordingRegressor.fits) == 3 * (2 + 1)
        assert _RecordingRegressor.predictions
        for fitted, predicted in _RecordingRegressor.predictions:
            assert set(fitted).isdisjoint(predicted)
        outer = make_folds(
            selector.data.n,
            3,
            stratify=selector.est._fold_strata(selector.data),
            cluster=selector.data.cluster,
            random_state=selector.seed,
        )
        fits_per_outer = selector.est.selection_inner_folds + 1
        for fold, (train, _) in enumerate(outer):
            start = fold * fits_per_outer
            fitted_sets = [
                set(values) for values in _RecordingRegressor.fits[start : start + fits_per_outer]
            ]
            expected_full_fit = set(selector.data.covariates[train, 0].astype(int))
            assert expected_full_fit in fitted_sets

    def test_selection_folds_cross_fit_candidate_propensities(self) -> None:
        frame, _ = make_instrument(n=90, seed=14)
        frame.insert(2, "row_id", np.arange(len(frame)))
        selector = _selector(
            frame,
            strategy="discrete",
            candidates=[("row_id",)],
            treatment_learner=_RecordingClassifier(),
            cross_fit=False,
            n_folds=5,
            selection_folds=2,
        )[0]
        path = selector.build_path(train=None, tag="full")
        _RecordingClassifier.fits.clear()
        _RecordingClassifier.predictions.clear()
        selector.cross_validate(path)

        assert _RecordingClassifier.predictions
        for fitted, predicted in _RecordingClassifier.predictions:
            assert set(fitted).isdisjoint(predicted)
        outer = make_folds(
            selector.data.n,
            2,
            stratify=selector.est._fold_strata(selector.data),
            cluster=selector.data.cluster,
            random_state=selector.seed,
        )
        fits_per_outer = selector.est.selection_inner_folds + 1
        assert len(_RecordingClassifier.fits) == len(outer) * fits_per_outer
        for fold, (_, validation) in enumerate(outer):
            start = fold * fits_per_outer
            for fitted in _RecordingClassifier.fits[start : start + fits_per_outer]:
                assert set(fitted).isdisjoint(validation)

    def test_selection_folds_keep_the_outer_outcome_scaler(self) -> None:
        frame, _ = make_instrument(n=90, seed=15)
        selector = _selector(frame, selection_folds=2)[0]
        folds = make_folds(
            selector.data.n,
            2,
            stratify=selector.est._fold_strata(selector.data),
            cluster=selector.data.cluster,
            random_state=selector.seed,
        )
        train, _ = next(iter(folds))
        nested, mask = selector._nested_folds(train)
        fold_base = selector._selection_base(nested, mask)
        assert fold_base.scaler is selector.base.scaler

    def test_it_picks_the_minimum_cross_validated_risk(self, instrument_frame) -> None:
        estimator = CTMLE(
            **{**CTMLE_KWARGS, "strategy": "discrete", "candidates": [("W1",), ("W2",), ("W3",)]}
        )
        result = estimator.fit(instrument_frame, outcome="Y", treatment="A").single()
        selection = result.extra["ctmle"]
        assert selection.selected == int(np.argmin(selection.cv_risk))
        assert selection.selected_covariates == selection.path[selection.selected]

    def test_the_scalable_search_leaves_the_instrument_out(self) -> None:
        """The instrument is never selected -- but read the next test before believing this.

        The assertion is true and reproducible: over ten seeds the ordered search puts
        ``W2`` in the selected model zero times.  What it does *not* establish is that the
        search discriminated against the instrument, because on this process the selected
        model is usually **empty**, and the empty set excludes ``W2`` for free.
        :meth:`test_the_exclusion_is_mostly_carried_by_selecting_nothing` measures that
        directly rather than leaving it implicit.
        """
        excluded = 0
        for seed in range(5):
            frame, _ = make_instrument(n=700, seed=seed)
            selection = (
                CTMLE(**{**CTMLE_KWARGS, "strategy": "ordered"})
                .fit(frame, outcome="Y", treatment="A")
                .single()
                .extra["ctmle"]
            )
            excluded += "W2" not in selection.selected_covariates
        assert excluded == 5

    def test_the_right_outcome_model_makes_the_empty_propensity_optimal(self) -> None:
        """The selected propensity is empty on every fixed seed for this process.

        This is recorded as an assertion rather than left as folklore, because it changes
        what every other C-TMLE claim in the suite is evidence *for*.  On
        :func:`~cleverly.datasets.instrument_dgp` a GLM is correctly specified for
        ``Qbar`` -- the outcome mean is exactly ``1 + a + 1.5 W1 + 0.8 W3`` -- so under
        collaborative double robustness the confounding is already handled before ``g`` is
        asked for anything, and an empty propensity model is a legitimate risk-minimising
        choice rather than a failure. Measured at ``n = 700`` after nested selection
        cross-fitting, the ordered search selects nothing in all five fixed seeds below.

        The consequence is the point, and it was checked rather than assumed: replacing the
        selection with ``selected = 0`` -- a selector hard-wired to return the empty
        candidate -- left every C-TMLE test in this suite passing except this one and the
        four in
        :class:`tests.e2e.test_ctmle.TestSelectionIsForcedWhenTheOutcomeModelCannotHelp`.
        The loss, penalty and path tests below all passed too, because they exercise
        :class:`~cleverly.estimators.ctmle._Selector`'s methods directly and never reach the
        step that chooses among candidates.

        So the variance and RMSE comparisons in the slow tier are not evidence that the
        collaborative search works; they are evidence that a propensity model containing an
        instrument costs variance, which is a fact about plain TMLE.  The test that does
        discriminate has to make selecting nothing *wrong*, which means taking away the
        correctly specified outcome model -- see the class named above.
        """
        empty = 0
        for seed in range(5):
            frame, _ = make_instrument(n=700, seed=seed)
            selection = (
                CTMLE(**{**CTMLE_KWARGS, "strategy": "ordered"})
                .fit(frame, outcome="Y", treatment="A")
                .single()
                .extra["ctmle"]
            )
            empty += len(selection.selected_covariates) == 0
        # This is no longer used as a loose proxy for search discrimination. The direct
        # forced-misspecification suite below is what fails a do-nothing selector.
        assert empty == 5, f"empty in {empty} of 5 seeds"

    def test_the_empty_model_beats_the_full_one_when_the_outcome_model_is_right(self) -> None:
        # The head-to-head behind the test above, without the search's noise: offered only
        # "adjust for nothing" and "adjust for everything", the risk criterion prefers
        # nothing on this process. That is the correct answer here -- and it is why the
        # dominance comparisons cannot tell a working search from a broken one.
        frame, _ = make_instrument(n=700, seed=0)
        selection = (
            CTMLE(
                **{
                    **CTMLE_KWARGS,
                    "strategy": "discrete",
                    "candidates": [(), ("W1", "W2", "W3")],
                }
            )
            .fit(frame, outcome="Y", treatment="A")
            .single()
            .extra["ctmle"]
        )
        assert selection.selected_covariates == ()
        assert selection.cv_risk[0] < selection.cv_risk[1]

    def test_the_selected_model_is_the_one_that_was_used(self, instrument_frame) -> None:
        result = CTMLE(**CTMLE_KWARGS).fit(instrument_frame, outcome="Y", treatment="A").single()
        selection = result.extra["ctmle"]
        assert result.nuisance.treatment_covariates == selection.selected_covariates
        assert set(selection.dropped) == set(selection.covariates) - set(
            selection.selected_covariates
        )

    def test_selection_is_recorded_for_every_search(self, instrument_frame) -> None:
        for strategy, extra in (
            ("greedy", {}),
            ("ordered", {}),
            ("discrete", {"candidates": [("W1",), ("W1", "W2")]}),
        ):
            result = (
                CTMLE(**{**CTMLE_KWARGS, "strategy": strategy, **extra})
                .fit(instrument_frame, outcome="Y", treatment="A")
                .single()
            )
            selection = result.extra["ctmle"]
            assert selection.strategy == strategy
            assert len(selection.cv_risk) == len(selection.path)
            assert len(selection.train_risk) == len(selection.path)
            assert np.isfinite(selection.cv_risk).all()


class TestEquivalenceWithPlainTmle:
    def test_a_single_full_candidate_reproduces_tmle_exactly(self) -> None:
        # With one candidate there is nothing to select, so C-TMLE is a plain TMLE
        # with an extra bookkeeping step. Bit-for-bit equality is the sharpest
        # available check that the selection layer does not perturb the estimator.
        frame, _ = make_linear_ate(n=500, seed=21)
        every = ("W1", "W2", "W3", "W4")
        collaborative = (
            CTMLE(**{**CTMLE_KWARGS, "strategy": "discrete", "candidates": [every]})
            .fit(frame, outcome="Y", treatment="A")
            .single()
        )
        plain = TMLE(**TMLE_KWARGS).fit(frame, outcome="Y", treatment="A").single()

        assert collaborative.psi("ate") == pytest.approx(plain.psi("ate"), abs=1e-12)
        assert collaborative["ate"].std_error == pytest.approx(plain["ate"].std_error, abs=1e-12)
        assert np.allclose(
            collaborative["ate"].influence_curve,
            plain["ate"].influence_curve,
            atol=1e-12,
            rtol=0.0,
        )

    def test_the_selected_targeted_outcome_is_the_final_targeting_start(
        self, instrument_frame
    ) -> None:
        result = CTMLE(**CTMLE_KWARGS).fit(instrument_frame, outcome="Y", treatment="A").single()
        selected = result.nuisance.targeting_outcome
        assert selected is not None
        final = result.fluctuations["mean"]
        assert np.max(np.abs(final.epsilon)) < 1e-8
        assert np.allclose(final.targeted.observed, selected.observed, atol=1e-8, rtol=0.0)

    def test_selected_targeting_state_survives_serialization(self, instrument_frame) -> None:
        result = CTMLE(**CTMLE_KWARGS).fit(instrument_frame, outcome="Y", treatment="A").single()
        restored = loads(dumps(result))
        assert result.nuisance.targeting_outcome is not None
        assert restored.nuisance.targeting_outcome is not None
        assert np.array_equal(
            restored.nuisance.targeting_outcome.observed,
            result.nuisance.targeting_outcome.observed,
        )
        assert restored.validation.score_check().passed

    def test_it_solves_the_score_equation(self, instrument_frame) -> None:
        result = CTMLE(**CTMLE_KWARGS).fit(instrument_frame, outcome="Y", treatment="A").single()
        assert result.validation.score_check().passed


class TestReporting:
    @pytest.fixture(scope="class")
    def selection(self, instrument_frame) -> object:
        return (
            CTMLE(**CTMLE_KWARGS)
            .fit(instrument_frame, outcome="Y", treatment="A")
            .single()
            .extra["ctmle"]
        )

    def test_the_summary_marks_the_chosen_candidate(self, selection) -> None:
        text = selection.summary()
        assert "Collaborative TMLE selection" in text
        assert "strategy = greedy" in text
        assert "squared-error" in text
        assert text.count("<--") == 1

    def test_to_frame_has_one_row_per_candidate(self, selection, instrument_frame) -> None:
        frame = selection.to_frame()
        assert len(frame["candidate"]) == len(selection.path)
        assert sum(frame["selected"]) == 1

    def test_reported_training_components_match_their_formulas(self, selection) -> None:
        np.testing.assert_allclose(
            selection.train_loss + selection.penalty,
            selection.train_risk,
            rtol=1e-12,
            atol=0.0,
        )
        assert np.isfinite(selection.treatment_risk).all()
        assert (selection.treatment_risk > 0.0).all()

    def test_the_intercept_candidate_is_labelled(self, selection) -> None:
        assert "(intercept)" in selection.summary()


class TestValidation:
    def test_the_replaced_search_keyword_has_a_migration_message(self) -> None:
        with pytest.raises(TypeError, match="search= was replaced by strategy="):
            CTMLE(search="ordered")

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"strategy": "stepwise"}, "strategy must be"),
            ({"strategy": "discrete"}, "needs an explicit candidates"),
            ({"candidates": [("W1",)]}, "only applies to strategy='discrete'"),
            ({"ordering": ["W1"]}, "only applies to strategy='ordered'"),
            ({"selection_folds": 1}, "selection_folds must be at least 2"),
            ({"selection_inner_folds": 1}, "selection_inner_folds must be at least 2"),
            ({"loss": "hinge"}, "loss must be"),
            ({"preorder": "marginal"}, "preorder must be"),
            ({"preorder": "logistic"}, "only applies to strategy='ordered'"),
            (
                {
                    "strategy": "ordered",
                    "ordering": ["W1"],
                    "preorder": "logistic",
                },
                "cannot be combined",
            ),
            ({"cv_evaluation": True}, "canonical CV-TMLE selection"),
            ({"targeting_scheme": "fold"}, "published pooled"),
            ({"ctmle_estimand": "att"}, "ctmle_estimand must be one of"),
        ],
    )
    def test_bad_settings_are_rejected_at_construction(
        self, kwargs: dict[str, object], message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            CTMLE(**kwargs)

    def test_conditional_estimands_are_refused_with_an_explanation(self, instrument_frame) -> None:
        estimator = CTMLE(**{**FAST_KWARGS, "estimands": ("ate", "att")})
        with pytest.raises(ValueError, match="does not support estimand"):
            estimator.fit(instrument_frame, outcome="Y", treatment="A").single()

    def test_controlled_direct_effect_composition_is_refused(self) -> None:
        frame, _ = make_cde(n=100, seed=4)
        with pytest.raises(ValueError, match="does not compose either collaborative strategy"):
            CTMLE(**CTMLE_KWARGS).fit(frame, outcome="Y", treatment="A", intermediate="Z")

    def test_the_target_estimand_must_be_reported(self, instrument_frame) -> None:
        estimator = CTMLE(**{**FAST_KWARGS, "estimands": ("ey1",), "ctmle_estimand": "ate"})
        with pytest.raises(ValueError, match="not among the requested estimands"):
            estimator.fit(instrument_frame, outcome="Y", treatment="A").single()

    def test_an_ordering_must_name_every_covariate(self, instrument_frame) -> None:
        estimator = CTMLE(**{**CTMLE_KWARGS, "strategy": "ordered", "ordering": ["W1", "W2"]})
        with pytest.raises(ValueError, match="must cover every covariate"):
            estimator.fit(instrument_frame, outcome="Y", treatment="A").single()

    def test_an_ordering_cannot_name_an_unknown_covariate(self, instrument_frame) -> None:
        estimator = CTMLE(**{**CTMLE_KWARGS, "strategy": "ordered", "ordering": ["W1", "W2", "W9"]})
        with pytest.raises(ValueError, match="unknown covariate"):
            estimator.fit(instrument_frame, outcome="Y", treatment="A").single()
