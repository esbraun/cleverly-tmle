"""Exact fast checks for the audited stacked MAR natural-course CV-TMLE."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator

from cleverly import CapabilityError, CrossFitting, DataError, SplitPlan, TMLEMethod
from cleverly.inference.cluster import (
    cross_validated_variance,
    influence_covariance,
    influence_variance,
)
from cleverly.learners import make_folds
from cleverly.utils.bounds import logit
from tests import discrete_law_mar as law
from tests.conftest import OracleMissingness, OracleOutcome
from tests.studies.missing_outcome_study_helpers import NaturalCourseLaw
from tests.unit._natural_course_support import (
    NeverFit,
    exact_remainder,
    never_fit_learners,
    stacked_tmle,
)


def _fit(frame: pd.DataFrame | None = None, **overrides: Any) -> Any:
    return (
        stacked_tmle(**overrides)
        .fit(
            law.frame() if frame is None else frame,
            outcome="Y",
            treatment="A",
            covariates=("W",),
            delta="Delta",
        )
        .single()
    )


def test_public_configuration_translates_the_unstratified_policy() -> None:
    cross_fitting = CrossFitting(
        enabled=True,
        n_folds=10,
        repeats=1,
        stratify_by="none",
        targeting_scheme="pooled",
        fold_evaluation=False,
        split_plan=None,
    )
    kwargs = TMLEMethod(cross_fitting=cross_fitting).estimator_kwargs()

    assert kwargs["cross_fit"] is True
    assert kwargs["n_folds"] == 10
    assert kwargs["repeats"] == 1
    assert kwargs["stratify_folds"] == "none"
    assert kwargs["targeting_scheme"] == "pooled"
    assert kwargs["cv_evaluation"] is False
    assert kwargs["split_plan"] is None


def test_the_audited_engine_cell_records_generated_unstratified_folds() -> None:
    result = _fit()

    assert result.config.cross_fit is True
    assert result.config.targeting_scheme == "pooled"
    assert result.config.cv_evaluation is False
    assert result.config.crossfit.scheme == "vfold"
    assert result.config.crossfit.stratify_by == ()
    assert result.config.crossfit.repeats == 1
    assert result.nuisance.fits_treatment is False


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"stratify_folds": "treatment"}, "stratify_folds='none'"),
        ({"stratify_folds": "treatment+outcome"}, "stratify_folds='none'"),
        ({"repeats": 2}, "repeats=1"),
        ({"targeting_scheme": "fold"}, "targeting_scheme='pooled'"),
        ({"cv_evaluation": True}, "cv_evaluation=False"),
        (
            {"split_plan": SplitPlan((tuple(np.arange(law.N) % 2),))},
            "package-generated folds",
        ),
    ],
)
def test_cross_fitted_variants_outside_the_audited_cell_refuse_before_fitting(
    overrides: dict[str, Any], message: str
) -> None:
    with pytest.raises(CapabilityError, match=message):
        _fit(**never_fit_learners(), **overrides)
    assert NeverFit.calls == 0


def test_unstratified_folds_refuse_an_established_complete_outcome_fit() -> None:
    frame = law.frame().assign(Y=lambda value: value["Y"].fillna(0.0), Delta=1.0)
    with pytest.raises(CapabilityError, match="reserved"):
        _fit(frame, **never_fit_learners())
    assert NeverFit.calls == 0


def test_unstratified_folds_refuse_an_intermediate_fit_before_its_shared_nuisances() -> None:
    """The intermediate path fits shared nuisances before ``_fit_single`` runs.

    The fold-policy reservation must therefore run in the resolver that precedes that
    shared fit, or this fit trains every learner and only then refuses.
    """
    frame = law.frame().assign(
        Y=lambda value: value["Y"].fillna(0.0), Z=(np.arange(law.N) % 2).astype(float)
    )
    estimator = stacked_tmle(estimands=("ate",), **never_fit_learners())
    with pytest.raises(CapabilityError, match="reserved"):
        estimator.fit(frame, outcome="Y", treatment="A", covariates=("W",), intermediate="Z")
    assert NeverFit.calls == 0


#: Rows in the response-support fixtures. Three folds make each training complement the
#: union of two validation folds, so a check that read a validation fold instead of its
#: complement names a different fold, or refuses a partition that fits.
PREFLIGHT_N = 60
PREFLIGHT_FOLDS = 3


def _preflight_folds() -> Any:
    return make_folds(PREFLIGHT_N, PREFLIGHT_FOLDS, stratify=None, random_state=17)


def _response_frame(rare: str, rows: np.ndarray) -> pd.DataFrame:
    """A sample in which ``rows`` are the only respondents or the only nonrespondents."""
    observed = np.full(PREFLIGHT_N, rare != "response")
    observed[rows] = rare == "response"
    latent = (np.arange(PREFLIGHT_N) % 2).astype(float)
    respondents = np.flatnonzero(observed)
    latent[respondents[:2]] = np.array([0.0, 1.0])[: min(2, respondents.size)]
    return pd.DataFrame(
        {
            "Y": np.where(observed, latent, np.nan),
            "A": (np.arange(PREFLIGHT_N) % 2).astype(float),
            "W": np.arange(PREFLIGHT_N, dtype=float),
            "Delta": observed.astype(float),
        }
    )


@pytest.mark.parametrize("rare", ["response", "nonresponse"])
def test_a_training_complement_without_a_response_kind_refuses_before_any_learner(
    rare: str,
) -> None:
    folds = _preflight_folds()
    last = PREFLIGHT_FOLDS - 1
    # Both rare rows sit in the last validation fold, so only that fold's complement lacks
    # them. The earlier validation folds lack them too, which a validation-fold check
    # would report as fold 0.
    frame = _response_frame(rare, folds.test_index(last)[:2])
    absent = "respondent" if rare == "response" else "nonrespondent"

    with pytest.raises(DataError, match=f"repeat 0, fold {last}'s training complement") as caught:
        _fit(frame, n_folds=PREFLIGHT_FOLDS, **never_fit_learners())
    message = str(caught.value)
    assert f"contains no {absent}" in message
    assert "Increase n_folds" in message and "use n_folds=1" in message
    assert "reduce n_folds" not in message
    assert NeverFit.calls == 0


@pytest.mark.parametrize("rare", ["response", "nonresponse"])
def test_a_validation_fold_without_a_response_kind_still_fits(rare: str) -> None:
    folds = _preflight_folds()
    # One rare row in each of the first two validation folds: every training complement
    # holds at least one, while the last validation fold holds none.
    rows = np.array([folds.test_index(0)[0], folds.test_index(1)[0]])
    frame = _response_frame(rare, rows)
    rare_rows = (frame["Delta"] == 1.0) if rare == "response" else (frame["Delta"] == 0.0)
    assert not rare_rows.to_numpy()[folds.test_index(PREFLIGHT_FOLDS - 1)].any()

    _RecordingProbability.fits = {"outcome": [], "response": []}
    result = _fit(
        frame,
        n_folds=PREFLIGHT_FOLDS,
        outcome_learner=_RecordingProbability("outcome"),
        missingness_learner=_RecordingProbability("response"),
    )

    np.testing.assert_array_equal(result.nuisance.folds.assignment, folds.assignment)
    assert np.isfinite(result["ey_obs"].psi)


@pytest.mark.parametrize("rare", ["response", "nonresponse"])
def test_a_sample_with_one_of_a_response_kind_names_the_in_sample_remedy(rare: str) -> None:
    frame = _response_frame(rare, np.array([0]))
    kind = "respondent" if rare == "response" else "nonrespondent"

    with pytest.raises(DataError, match=rf"the sample has 1 {kind}\(s\).*use cross_fit=False"):
        _fit(frame, n_folds=PREFLIGHT_FOLDS, **never_fit_learners())
    assert NeverFit.calls == 0


class _RecordingProbability(BaseEstimator):
    fits: ClassVar[dict[str, list[np.ndarray]]] = {"outcome": [], "response": []}

    def __init__(self, kind: str) -> None:
        self.kind = kind

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> _RecordingProbability:
        design = np.asarray(X, dtype=float)
        type(self).fits[self.kind].append(design[:, 1].copy())
        self.probability_ = float(np.clip(np.mean(np.asarray(y, dtype=float)), 0.05, 0.95))
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        probability = np.full(len(X), self.probability_)
        return np.column_stack([1.0 - probability, probability])


def test_outcome_and_response_fits_are_independently_out_of_fold_and_aligned() -> None:
    n = 80
    row = np.arange(n)
    observed = row % 5 != 0
    frame = pd.DataFrame(
        {
            "Y": np.where(observed, ((row // 2) % 2).astype(float), np.nan),
            "A": (row % 2).astype(float),
            "W": row.astype(float),
            "Delta": observed.astype(float),
        }
    )
    _RecordingProbability.fits = {"outcome": [], "response": []}
    result = _fit(
        frame,
        n_folds=4,
        outcome_learner=_RecordingProbability("outcome"),
        missingness_learner=_RecordingProbability("response"),
    )

    folds = result.nuisance.folds
    assert len(_RecordingProbability.fits["outcome"]) == folds.n_folds
    assert len(_RecordingProbability.fits["response"]) == folds.n_folds
    for fold, (train, test) in enumerate(folds):
        outcome_rows = _RecordingProbability.fits["outcome"][fold].astype(int)
        response_rows = _RecordingProbability.fits["response"][fold].astype(int)
        assert set(outcome_rows) == set(train[observed[train]])
        assert set(response_rows) == set(train)
        assert not set(outcome_rows) & set(test)
        assert not set(response_rows) & set(test)
        expected_outcome = frame.loc[train[observed[train]], "Y"].mean()
        expected_response = frame.loc[train, "Delta"].mean()
        np.testing.assert_allclose(result.nuisance.outcome.observed[test], expected_outcome)
        assert result.nuisance.missingness is not None
        np.testing.assert_allclose(result.nuisance.missingness[test], expected_response)


def test_pooled_score_eif_and_variance_match_the_stacked_rows_exactly() -> None:
    result = _fit()
    estimate = result["ey_obs"]
    fluctuation = result.fluctuations["natural_course"]
    targeted = np.asarray(fluctuation.targeted.observed, dtype=float)
    observed = result.data.observed
    response = result.nuisance.missingness_at_realised_arm(result.data.treatment)
    assert response is not None
    residual = np.where(observed, result.data.outcome - targeted, 0.0)
    score = residual / response
    expected = score + targeted - estimate.psi

    assert float(np.mean(score)) == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(estimate.influence_curve, expected, atol=1e-14, rtol=0.0)
    assert estimate.covariance_rule == "second_moment"
    assert estimate.variance == pytest.approx(np.mean(np.square(expected)) / result.data.n)
    assert result.covariance()[0, 0] == estimate.variance
    centered_ddof_variance = np.var(expected, ddof=1) / result.data.n
    assert result.covariance()[0, 0] != pytest.approx(centered_ddof_variance, rel=1e-8)

    doubled = result.contrast(
        lambda point: 2.0 * point[0],
        ["ey_obs"],
        gradient=lambda point: np.array([2.0]),
    )
    np.testing.assert_allclose(doubled.influence_curve, 2.0 * expected, atol=1e-14, rtol=0.0)
    assert doubled.variance == 4.0 * estimate.variance
    assert doubled.variance != pytest.approx(
        np.var(doubled.influence_curve, ddof=1) / result.data.n,
        rel=1e-8,
    )

    # Distinguishability witnesses, built from the stored arrays rather than by mutating
    # the package. Each is a curve a plausible defect would report, and each is far from
    # the reported curve, so the exact comparison above would reject that defect.
    without_response = targeted - estimate.psi
    assert np.max(np.abs(estimate.influence_curve - without_response)) > 0.5

    wrong_fold_response = np.empty_like(response)
    tests = [test for _, test in result.nuisance.folds]
    for fold, test in enumerate(tests):
        source = tests[(fold + 1) % len(tests)]
        wrong_fold_response[test] = response[source]
    wrong_fold_curve = (
        np.where(observed, residual / wrong_fold_response, 0.0) + targeted - estimate.psi
    )
    assert np.max(np.abs(estimate.influence_curve - wrong_fold_curve)) > 0.1


def test_the_in_sample_estimator_keeps_the_centered_variance() -> None:
    """The second-moment rule belongs to the cross-fitted estimator alone.

    An in-sample fit of the same target on the same law must keep the ordinary centred
    ``ddof=1`` variance. Keying the rule on the natural-course group without
    ``cross_fit`` fails here, because the two variances differ by ``n / (n - 1)``.
    """
    result = _fit(cross_fit=False, stratify_folds="treatment")
    estimate = result["ey_obs"]
    curve = np.asarray(estimate.influence_curve, dtype=float)

    assert estimate.covariance_rule == "centered"
    assert estimate.variance == influence_variance(curve)
    assert result.covariance()[0, 0] == influence_covariance(curve[:, None])[0, 0]
    assert estimate.variance != pytest.approx(np.mean(np.square(curve)) / curve.size, rel=1e-6)


class _FoldOracle(BaseEstimator):
    """An oracle whose law is the candidate table of the fold it is fitted for.

    The package fits fold ``v``'s learner ``v``-th, so a class-level counter selects the
    table. :func:`test_each_signed_fold_remainder_matches_its_exact_expansion` checks that
    ordering against the stored nuisances.
    """

    candidates: ClassVar[tuple[np.ndarray, ...]]
    next_fold: ClassVar[int] = 0

    @staticmethod
    def _oracle(table: np.ndarray) -> Any:  # pragma: no cover - overridden
        raise NotImplementedError

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> _FoldOracle:
        self.fold_ = type(self).next_fold
        type(self).next_fold += 1
        self.oracle_ = self._oracle(type(self).candidates[self.fold_]).fit(X, y)
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        return np.asarray(self.oracle_.predict_proba(X), dtype=float)


class _FoldOutcome(_FoldOracle):
    @staticmethod
    def _oracle(table: np.ndarray) -> Any:
        return OracleOutcome(NaturalCourseLaw(q=table))


class _FoldResponse(_FoldOracle):
    @staticmethod
    def _oracle(table: np.ndarray) -> Any:
        return OracleMissingness(NaturalCourseLaw(pi=table))


def _fold_exact_frame() -> pd.DataFrame:
    assignment = make_folds(2 * law.N, 2, stratify=None, random_state=17)
    frame = pd.DataFrame(np.nan, index=np.arange(2 * law.N), columns=("W", "A", "Y", "Delta"))
    values = law.frame().to_numpy()
    for _, test in assignment:
        frame.iloc[test] = values
    return frame.astype(float)


def test_odd_near_balanced_folds_use_whole_sample_row_weighting() -> None:
    row = np.arange(101)
    observed = row % 4 != 0
    frame = pd.DataFrame(
        {
            "Y": np.where(observed, ((row // 2) % 2).astype(float), np.nan),
            "A": ((row // 3) % 2).astype(float),
            "W": (row % 3).astype(float),
            "Delta": observed.astype(float),
        }
    )
    _FoldResponse.candidates = (
        np.array([[0.35, 0.40], [0.45, 0.50], [0.55, 0.60]]),
        np.array([[0.75, 0.70], [0.65, 0.60], [0.55, 0.50]]),
    )
    _FoldOutcome.candidates = (
        np.array([[0.15, 0.25], [0.20, 0.30], [0.25, 0.35]]),
        np.array([[0.80, 0.70], [0.75, 0.65], [0.70, 0.60]]),
    )
    _FoldResponse.next_fold = 0
    _FoldOutcome.next_fold = 0
    result = _fit(
        frame,
        outcome_learner=_FoldOutcome(),
        missingness_learner=_FoldResponse(),
    )
    estimate = result["ey_obs"]
    targeted = result.fluctuations["natural_course"].targeted.observed
    tests = [test for _, test in result.nuisance.folds]
    assert sorted(len(test) for test in tests) == [50, 51]

    row_weighted_point = float(np.mean(targeted))
    equal_fold_point = float(np.mean([np.mean(targeted[test]) for test in tests]))
    assert estimate.psi == row_weighted_point
    assert estimate.psi != pytest.approx(equal_fold_point, abs=1e-6)

    curve = estimate.influence_curve
    row_weighted_variance = float(np.mean(np.square(curve)) / len(curve))
    equal_fold_variance = cross_validated_variance(curve, tests)
    assert estimate.variance == row_weighted_variance
    assert estimate.variance != pytest.approx(equal_fold_variance, abs=1e-9)


PI_SHIFT = np.array([[0.10, -0.08], [-0.12, 0.10], [-0.20, 0.08]])
Q_SHIFT = np.array([[0.08, -0.12], [-0.10, 0.12], [0.06, 0.15]])


def _fold_candidate_fit() -> Any:
    """A fit on two exact copies of the law, each fold with its own wrong nuisances."""
    _FoldResponse.candidates = (law.PI + PI_SHIFT, law.PI - 0.4 * PI_SHIFT)
    _FoldOutcome.candidates = (law.Q + Q_SHIFT, law.Q - 0.7 * Q_SHIFT)
    _FoldResponse.next_fold = 0
    _FoldOutcome.next_fold = 0
    return _fit(
        _fold_exact_frame(),
        outcome_learner=_FoldOutcome(),
        missingness_learner=_FoldResponse(),
    )


def _cells(result: Any, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    subset = result.data.subset(test)
    return subset.covariates[:, 0].astype(int), subset.treatment.astype(int)


def test_each_signed_fold_remainder_matches_its_exact_expansion() -> None:
    result = _fold_candidate_fit()
    estimate = result["ey_obs"]
    targeted = result.fluctuations["natural_course"].targeted.observed
    response = result.nuisance.missingness_at_realised_arm(result.data.treatment)
    assert response is not None
    initial = result.nuisance.outcome.observed

    remainders = []
    pooled = 0.0
    for fold, (_, test) in enumerate(result.nuisance.folds):
        subset = result.data.subset(test)
        w, a = _cells(result, test)
        pi = _FoldResponse.candidates[fold][w, a]
        # The stored held-out nuisances are this fold's candidates, which ties the fold
        # index used below to the package's own assignment.
        np.testing.assert_allclose(response[test], pi, rtol=0.0, atol=1e-15)
        np.testing.assert_allclose(
            initial[test], _FoldOutcome.candidates[fold][w, a], rtol=0.0, atol=1e-15
        )
        m_star = targeted[test]
        residual = np.where(subset.observed, subset.outcome - m_star, 0.0)
        expansion = float(np.mean(m_star) - law.TRUTH["ey_obs"] + np.mean(residual / pi))
        remainder = exact_remainder(w, a, pi, m_star)
        assert expansion == pytest.approx(remainder, abs=1e-12)
        assert expansion != pytest.approx(-remainder, abs=1e-3)
        remainders.append(remainder)
        pooled += test.size / result.data.n * remainder

    assert min(abs(value) for value in remainders) > 0.005
    assert remainders[0] != pytest.approx(remainders[1], abs=1e-3)
    # Each fold's identity above holds for any per-fold candidates. The row-weighted sum
    # equals the estimator's error only when the pooled score solved the targeting step
    # with each row's own fold response probability.
    assert estimate.psi - law.TRUTH["ey_obs"] == pytest.approx(pooled, abs=1e-10)
    assert abs(pooled) > 0.005


def test_one_common_fluctuation_moves_every_stacked_row() -> None:
    """``logit m* - logit m`` is one coefficient times each row's own fold ``1 / pi``.

    Fold-specific coefficients, or a covariate read from another fold's response model,
    would each leave a fold off this line.
    """
    result = _fold_candidate_fit()
    epsilon = np.asarray(result.fluctuations["natural_course"].epsilon, dtype=float)
    assert epsilon.shape == (1,)
    assert abs(epsilon[0]) > 1e-3

    shift = logit(result.fluctuations["natural_course"].targeted.observed) - logit(
        result.nuisance.outcome.observed
    )
    for fold, (_, test) in enumerate(result.nuisance.folds):
        w, a = _cells(result, test)
        covariate = 1.0 / _FoldResponse.candidates[fold][w, a]
        np.testing.assert_allclose(shift[test], epsilon[0] * covariate, rtol=0.0, atol=1e-10)
