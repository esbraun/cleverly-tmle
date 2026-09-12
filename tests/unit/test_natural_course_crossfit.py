"""Exact fast checks for the audited stacked MAR natural-course CV-TMLE."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator

from cleverly import CapabilityError, CrossFitting, DataError, SplitPlan, TMLEMethod
from cleverly.estimators import TMLE
from cleverly.inference.cluster import cross_validated_variance
from cleverly.learners import make_folds
from tests import discrete_law_mar as law
from tests.conftest import OracleMissingness, OracleOutcome
from tests.studies.missing_outcome_study_helpers import FailTreatment, NaturalCourseLaw


def _estimator(**overrides: Any) -> TMLE:
    settings: dict[str, Any] = {
        "outcome_learner": OracleOutcome(NaturalCourseLaw()),
        "treatment_learner": FailTreatment(),
        "missingness_learner": OracleMissingness(NaturalCourseLaw()),
        "estimands": ("ey_obs",),
        "cross_fit": True,
        "n_folds": 2,
        "repeats": 1,
        "stratify_folds": "none",
        "targeting_scheme": "pooled",
        "cv_evaluation": False,
        "simultaneous": False,
        "random_state": 17,
        "max_iter": 100,
        "tol": 1e-12,
    }
    settings.update(overrides)
    return TMLE(**settings)


def _fit(frame: pd.DataFrame | None = None, **overrides: Any) -> Any:
    return (
        _estimator(**overrides)
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
        _fit(**overrides)


def test_unstratified_folds_refuse_an_established_complete_outcome_fit() -> None:
    frame = law.frame().assign(Y=lambda value: value["Y"].fillna(0.0), Delta=1.0)
    with pytest.raises(CapabilityError, match="reserved"):
        _fit(frame)


@pytest.mark.parametrize(
    ("transform", "fit_roles", "message"),
    [
        (
            lambda frame: frame.assign(weight=np.linspace(0.5, 1.5, len(frame))),
            {"weights": "weight"},
            "observation weights",
        ),
        (
            lambda frame: frame.assign(id=np.arange(len(frame)) // 2),
            {"id": "id"},
            "clustered inference",
        ),
        (
            lambda frame: frame.assign(stratum=np.arange(len(frame)) % 2),
            {"strata": ("stratum",)},
            "baseline strata",
        ),
        (
            lambda frame: frame.assign(Z=np.arange(len(frame)) % 2),
            {"intermediate": "Z"},
            "intermediate=",
        ),
        (
            lambda frame: frame.assign(A=np.arange(len(frame)) % 3),
            {},
            "exactly two arms",
        ),
        (
            lambda frame: frame.assign(
                Y=np.where(frame["Delta"] == 1.0, np.arange(len(frame)) / len(frame), np.nan)
            ),
            {},
            "binary outcome",
        ),
    ],
    ids=("weighted", "clustered", "strata", "intermediate", "multi-arm", "continuous"),
)
def test_unstratified_folds_refuse_every_unaudited_data_axis(
    transform: Any, fit_roles: dict[str, Any], message: str
) -> None:
    frame = transform(law.frame())
    covariates = ("W", "stratum") if "strata" in fit_roles else ("W",)
    with pytest.raises(CapabilityError, match=message):
        _estimator().fit(
            frame,
            outcome="Y",
            treatment="A",
            covariates=covariates,
            delta="Delta",
            **fit_roles,
        )


class _NeverFit(BaseEstimator):
    calls: ClassVar[int] = 0

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> _NeverFit:
        type(self).calls += 1
        raise AssertionError("the response-support preflight must run before either learner")

    def predict_proba(self, X: Any) -> Any:  # pragma: no cover - fit must fail first
        raise AssertionError("the response-support preflight must run before prediction")


@pytest.mark.parametrize("rare", ["response", "nonresponse"])
def test_every_training_complement_is_preflighted_before_either_learner(rare: str) -> None:
    n = 40
    folds = make_folds(n, 2, stratify=None, random_state=17)
    # Put the rare level in the last validation fold. The first complement is valid,
    # so this fails unless the preflight checks every complement before any fit begins.
    rare_rows = folds.test_index(folds.n_folds - 1)[:2]
    observed = np.zeros(n, dtype=bool) if rare == "response" else np.ones(n, dtype=bool)
    observed[rare_rows] = rare == "response"
    latent = np.arange(n) % 2
    latent[rare_rows] = np.array([0, 1])
    frame = pd.DataFrame(
        {
            "Y": np.where(observed, latent, np.nan),
            "A": np.arange(n) % 2,
            "W": np.arange(n, dtype=float),
            "Delta": observed.astype(float),
        }
    )
    _NeverFit.calls = 0

    absent = "respondent" if rare == "response" else "nonrespondent"
    with pytest.raises(DataError, match=f"no {absent}"):
        _fit(
            frame,
            outcome_learner=_NeverFit(),
            missingness_learner=_NeverFit(),
        )
    assert _NeverFit.calls == 0


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

    without_response = targeted - estimate.psi
    assert np.max(np.abs(estimate.influence_curve - without_response)) > 0.5

    rotated = np.empty_like(response)
    tests = [test for _, test in result.nuisance.folds]
    for fold, test in enumerate(tests):
        source = tests[(fold + 1) % len(tests)]
        rotated[test] = response[source]
    wrong_fold_curve = np.where(observed, residual / rotated, 0.0) + targeted - estimate.psi
    assert np.max(np.abs(estimate.influence_curve - wrong_fold_curve)) > 0.1


class _FoldOracle(BaseEstimator):
    candidates: ClassVar[tuple[np.ndarray, ...]]
    next_fold: ClassVar[int] = 0

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> _FoldOracle:
        self.fold_ = type(self).next_fold
        type(self).next_fold += 1
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        design = np.asarray(X, dtype=float)
        a = np.rint(design[:, 0]).astype(int)
        w = np.rint(design[:, 1]).astype(int)
        probability = type(self).candidates[self.fold_][w, a]
        return np.column_stack([1.0 - probability, probability])


class _FoldOutcome(_FoldOracle):
    pass


class _FoldResponse(_FoldOracle):
    pass


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


def test_each_signed_fold_remainder_matches_its_exact_expansion() -> None:
    pi_shift = np.array([[0.10, -0.08], [-0.12, 0.10], [-0.20, 0.08]])
    q_shift = np.array([[0.08, -0.12], [-0.10, 0.12], [0.06, 0.15]])
    _FoldResponse.candidates = (law.PI + pi_shift, law.PI - 0.6 * pi_shift)
    _FoldOutcome.candidates = (law.Q + q_shift, law.Q - 0.7 * q_shift)
    _FoldResponse.next_fold = 0
    _FoldOutcome.next_fold = 0
    result = _fit(
        _fold_exact_frame(),
        outcome_learner=_FoldOutcome(),
        missingness_learner=_FoldResponse(),
    )
    targeted = result.fluctuations["natural_course"].targeted.observed
    remainders = []
    for fold, (_, test) in enumerate(result.nuisance.folds):
        subset = result.data.subset(test)
        w = subset.covariates[:, 0].astype(int)
        a = subset.treatment.astype(int)
        pi = _FoldResponse.candidates[fold][w, a]
        m_star = targeted[test]
        m_zero = law.Q[w, a]
        residual = np.where(subset.observed, subset.outcome - m_star, 0.0)
        expansion = float(np.mean(m_star) - law.TRUTH["ey_obs"] + np.mean(residual / pi))
        remainder = float(np.mean((1.0 - law.PI[w, a] / pi) * (m_star - m_zero)))
        assert expansion == pytest.approx(remainder, abs=1e-12)
        assert expansion != pytest.approx(-remainder, abs=1e-3)
        remainders.append(remainder)

    assert min(abs(value) for value in remainders) > 0.005
    assert remainders[0] != pytest.approx(remainders[1], abs=1e-3)
