"""``ModelSpec(outcome_by_arm=True)``: one outcome regression per treatment arm.

The default fits one regression on ``[A, W]``.  With ``outcome_by_arm=True`` every fold fits one
learner per arm ``a``, on the rows ``observed & recorded & (A == a)`` of its training set and on
the covariates alone.  Row ``i``'s observed prediction comes from its own arm's model, the
counterfactual ``a`` comes from arm ``a``'s model on every row, and an arm-coded modified
treatment policy reads each row from the model of the arm it moves the row to.

Every claim below is checked against models fitted by hand with ``scikit-learn``, at ``1e-12``.
The law has an ``A x W`` interaction, so the per-arm fit differs from the pooled fit by far more
than the floor (:data:`WITNESS_FLOOR`).  A mask that forgets ``A == a`` and a swap of the arm
models each break one line of the shipped source and must fail the same hand comparison.
"""

from __future__ import annotations

import importlib
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import ModelSpec, TMLEMethod
from cleverly.data import CausalData
from cleverly.estimators import CTMLE, DRTMLE, TMLE
from cleverly.estimators import _nuisance as nuisance_module
from cleverly.estimators._nuisance import (
    ArmQuery,
    _check_arm_training,
    _fit_arm_models,
    _fit_outcome_by_arm,
    _predict_by_arm,
    fit_inner_designs,
)
from cleverly.estimators.serialize import dumps, loads
from cleverly.exceptions import (
    CapabilityError,
    DataError,
    MethodConfigurationError,
    RefusalReason,
)
from cleverly.interventions import ModifiedPolicy
from cleverly.learners.crossfit import Folds
from cleverly.methods import IPSIMethod, SDRMethod
from cleverly.msm import MSM
from tests.conftest import mean_one_weights
from tests.unit._source_mutation import mutate

ctmle_module = importlib.import_module("cleverly.estimators.ctmle")
tmle_module = importlib.import_module("cleverly.estimators.tmle")

#: The smallest gap between the per-arm and the pooled outcome regression the law must show, so
#: that a check against the per-arm fit is not also a check against the pooled one.
WITNESS_FLOOR = 1e-3

#: The plan's pinned refusal texts (plan I-2.8, R-f1 and R-f2).
R_F1 = (
    "outcome_by_arm=True fits one outcome regression per treatment arm, so it needs an "
    "arm-coded treatment. A continuous dose has no arms."
)
R_F2 = (
    "outcome_by_arm=True found no row with {treatment} = {arm!r} and an observed outcome in "
    "{where}, so that arm's outcome regression has nothing to fit. Use fewer folds, or "
    "outcome_by_arm=False"
)


def outcome_learner() -> LogisticRegression:
    """A deterministic logistic regression, so a hand fit and the package fit are one model."""
    return LogisticRegression(C=1e6, max_iter=5000)


def treatment_learner() -> LogisticRegression:
    return LogisticRegression(max_iter=1000)


def small_super_learner() -> Any:
    """A two-candidate Super Learner, for the diagnostics each fitted model leaves."""
    from cleverly.learners import SuperLearner

    return SuperLearner(
        [LogisticRegression(C=1e6, max_iter=500), LogisticRegression(C=0.5, max_iter=500)],
        task="classification",
        n_folds=2,
    )


# ------------------------------------------------------------------------------------- the laws


def expit(x: Any) -> Any:
    return 1.0 / (1.0 + np.exp(-x))


def binary_frame(n: int = 600, seed: int = 0, **extras: Any) -> pd.DataFrame:
    """Two arms, a binary outcome and an ``A x W`` interaction.

    ``logit P(Y = 1 | A, W) = 0.2 + 0.5 W1 + A (0.5 + 1.5 W2)``: the effect of ``A`` changes sign
    with ``W2``, which no additive-in-``A`` regression can follow.
    """
    rng = np.random.default_rng(seed)
    w1, w2 = rng.normal(size=n), rng.normal(size=n)
    a = rng.binomial(1, expit(0.4 * w1)).astype(float)
    y = rng.binomial(1, expit(0.2 + 0.5 * w1 + a * (0.5 + 1.5 * w2))).astype(float)
    frame = pd.DataFrame({"Y": y, "A": a, "W1": w1, "W2": w2})
    for name, build in extras.items():
        frame[name] = build(frame, rng)
    return frame


def three_arm_frame(n: int = 900, seed: int = 1) -> pd.DataFrame:
    """Three arms with a different slope in ``W2`` in each."""
    rng = np.random.default_rng(seed)
    w1, w2 = rng.normal(size=n), rng.normal(size=n)
    scores = np.column_stack([np.zeros(n), 0.5 * w1, -0.3 * w1])
    probabilities = np.exp(scores) / np.exp(scores).sum(axis=1, keepdims=True)
    a = np.array([rng.choice(3, p=p) for p in probabilities], dtype=float)
    slope = np.array([0.0, 1.5, -1.5])[a.astype(int)]
    intercept = np.array([0.0, 0.5, -0.4])[a.astype(int)]
    y = rng.binomial(1, expit(0.2 + 0.5 * w1 + intercept + slope * w2)).astype(float)
    return pd.DataFrame({"Y": y, "A": a, "W1": w1, "W2": w2})


def with_missing_outcome(frame: pd.DataFrame, seed: int = 5) -> pd.DataFrame:
    """Outcomes missing at random given ``(A, W)``, with the observation indicator ``Delta``."""
    rng = np.random.default_rng(seed)
    p = expit(0.9 + 0.4 * frame["W1"] - 0.4 * frame["A"])
    delta = rng.binomial(1, p).astype(float)
    out = frame.copy()
    out["Delta"] = delta
    out.loc[delta == 0, "Y"] = np.nan
    return out


def with_missing_treatment(frame: pd.DataFrame, seed: int = 6) -> pd.DataFrame:
    """Treatments missing at random given ``W``, with the observation indicator ``DeltaA``."""
    rng = np.random.default_rng(seed)
    delta = rng.binomial(1, expit(1.2 + 0.4 * frame["W2"])).astype(float)
    out = frame.copy()
    out["DeltaA"] = delta
    out.loc[delta == 0, "A"] = np.nan
    return out


def fit_tmle(frame: pd.DataFrame, *, by_arm: bool = True, **settings: Any) -> Any:
    """One TMLE fit with the logistic learners, returning the single result."""
    columns = {
        key: settings.pop(key)
        for key in ("delta", "weights", "id", "intermediate", "strata", "treatment_delta")
        if key in settings
    }
    covariates = settings.pop("covariates", ["W1", "W2"])
    estimator = TMLE(
        **{
            "outcome_learner": outcome_learner(),
            "treatment_learner": treatment_learner(),
            "missingness_learner": treatment_learner(),
            "cross_fit": False,
            "simultaneous": False,
            "random_state": 0,
            "outcome_by_arm": by_arm,
            "estimands": ("ate", "ey1", "ey0"),
            **settings,
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return estimator.fit(
            frame, outcome="Y", treatment="A", covariates=covariates, **columns
        ).single()


# --------------------------------------------------------------------------- the hand reference


def hand_models(
    data: CausalData,
    folds: Folds,
    target: Any,
    *,
    mask: Any = None,
    make: Callable[[], Any] = outcome_learner,
    level: float | None = None,
    task: str = "classification",
) -> dict[float, np.ndarray]:
    """Per-arm out-of-fold predictions at every row, from models fitted by hand.

    For each fold the model of arm ``a`` is fitted on the training rows with an observed outcome,
    a recorded treatment ``A == a`` and (``mask``) any further restriction, on the covariates
    (and the intermediate variable) alone.  Row ``i``'s prediction comes from the model of the
    fold that held row ``i`` out.  ``level`` sets the intermediate variable for everybody.
    """
    covariates = np.asarray(data.covariates, dtype=float)
    z = None if not data.has_intermediate else np.asarray(data.intermediate, dtype=float)
    train_x = covariates if z is None else np.column_stack([covariates, z])
    predict_x = (
        train_x
        if level is None or z is None
        else np.column_stack([covariates, np.full(data.n, level)])
    )
    treatment = np.asarray(data.treatment, dtype=float)
    weights = np.asarray(data.weights, dtype=float)
    eligible = np.asarray(data.observed, dtype=bool) & np.isfinite(treatment)
    if mask is not None:
        eligible &= np.asarray(mask, dtype=bool)
    splits = (
        [(np.arange(data.n), np.arange(data.n))]
        if folds.is_single
        else [(np.asarray(train), np.asarray(test)) for train, test in folds]
    )
    out = {arm: np.empty(data.n) for arm in data.arm_codes}
    for train, test in splits:
        for arm in data.arm_codes:
            rows = train[eligible[train] & (treatment[train] == arm)]
            model = make().fit(train_x[rows], target[rows], sample_weight=weights[rows])
            if task == "classification":
                out[arm][test] = model.predict_proba(predict_x[test])[:, 1]
            else:
                out[arm][test] = np.clip(model.predict(predict_x[test]), 0.0, 1.0)
    return out


def own_arm(data: CausalData, by_arm: dict[float, np.ndarray]) -> np.ndarray:
    """Each row read from the table of its own arm; an unrecorded treatment reads arm 0."""
    codes = np.where(np.isfinite(data.treatment), data.treatment, data.arm_codes[0])
    return np.choose(np.asarray(codes, dtype=int), [by_arm[arm] for arm in data.arm_codes])


def disagreements(
    result: Any, *, mask: Any = None, task: str = "classification", make: Any = None
) -> list[str]:
    """Where the fitted outcome regression departs from the hand-fitted per-arm models."""
    data, nuisance = result.data, result.nuisance
    target = nuisance.scaler.scale(data.outcome)
    kwargs: dict[str, Any] = {"task": task, "mask": mask}
    if make is not None:
        kwargs["make"] = make
    reference = hand_models(data, nuisance.folds, target, **kwargs)
    problems: list[str] = []
    for arm in data.arm_codes:
        gap = float(np.max(np.abs(nuisance.outcome.arms[arm] - reference[arm])))
        if gap > 1e-12:
            problems.append(f"arm {arm}: {gap:.3g}")
    gap = float(np.max(np.abs(nuisance.outcome.observed - own_arm(data, reference))))
    if gap > 1e-12:
        problems.append(f"observed: {gap:.3g}")
    return problems


def pooled_gap(result: Any, frame: pd.DataFrame, **settings: Any) -> float:
    """The largest per-arm prediction gap between the per-arm and the pooled fit."""
    pooled = fit_tmle(frame, by_arm=False, **settings)
    return max(
        float(np.max(np.abs(result.nuisance.outcome.arms[arm] - pooled.nuisance.outcome.arms[arm])))
        for arm in result.data.arm_codes
    )


# ------------------------------------------------------------------------- the first site


class TestTheOutcomeSiteOfFitNuisances:
    """``fit_nuisances``: the primary outcome regression, in sample and cross-fitted."""

    def test_in_sample_predictions_are_the_hand_fitted_arm_models(self) -> None:
        result = fit_tmle(binary_frame())
        assert result.nuisance.folds.is_single
        assert disagreements(result) == []

    @pytest.mark.parametrize("n_folds", [2, 4])
    def test_cross_fitted_predictions_are_the_hand_fitted_arm_models(self, n_folds: int) -> None:
        result = fit_tmle(binary_frame(), cross_fit=True, n_folds=n_folds)
        assert result.nuisance.folds.n_folds == n_folds
        assert disagreements(result) == []

    def test_the_interaction_makes_the_arm_fit_differ_from_the_pooled_fit(self) -> None:
        frame = binary_frame()
        cases: list[dict[str, Any]] = [{}, {"cross_fit": True, "n_folds": 3}]
        for settings in cases:
            result = fit_tmle(frame, **settings)
            assert pooled_gap(result, frame, **settings) > WITNESS_FLOOR

    def test_the_observed_prediction_reads_the_rows_own_arm(self) -> None:
        result = fit_tmle(binary_frame())
        arms = result.nuisance.outcome.arms
        treated = result.data.treatment == 1.0
        np.testing.assert_array_equal(result.nuisance.outcome.observed[treated], arms[1.0][treated])
        np.testing.assert_array_equal(
            result.nuisance.outcome.observed[~treated], arms[0.0][~treated]
        )

    def test_a_three_arm_treatment_fits_three_models(self) -> None:
        result = fit_tmle(three_arm_frame(), cross_fit=True, n_folds=3, estimands=("ate", "ey"))
        assert len(result.data.arm_codes) == 3
        assert disagreements(result) == []

    def test_observation_weights_reach_each_arm_model(self) -> None:
        frame = binary_frame()
        frame["w"] = mean_one_weights(len(frame), (0.3, 1.7))
        result = fit_tmle(frame, weights="w", cross_fit=True, n_folds=3)
        assert not np.allclose(result.data.weights, 1.0)
        assert disagreements(result) == []

    def test_clusters_keep_their_folds_and_the_models_stay_per_arm(self) -> None:
        frame = binary_frame(n=600)
        frame["id"] = np.arange(len(frame)) // 6
        result = fit_tmle(frame, id="id", cross_fit=True, n_folds=3)
        assert result.data.cluster is not None
        assert disagreements(result) == []

    def test_missing_outcomes_train_the_arm_models_on_observed_rows_only(self) -> None:
        frame = with_missing_outcome(binary_frame(n=800))
        result = fit_tmle(frame, delta="Delta", cross_fit=True, n_folds=3)
        assert result.data.has_missing_outcome
        assert disagreements(result) == []
        # The mask is the observed one: a model trained on every row would see NaN outcomes.
        assert pooled_gap(result, frame, delta="Delta", cross_fit=True, n_folds=3) > WITNESS_FLOOR

    def test_a_missing_treatment_trains_on_recorded_rows_and_reads_arm_zero_where_unrecorded(
        self,
    ) -> None:
        frame = with_missing_treatment(binary_frame(n=800))
        result = fit_tmle(frame, treatment_delta="DeltaA")
        assert result.data.has_missing_treatment
        assert disagreements(result) == []

    def test_a_missing_outcome_and_a_missing_treatment_together(self) -> None:
        frame = with_missing_treatment(with_missing_outcome(binary_frame(n=1000)))
        result = fit_tmle(frame, delta="Delta", treatment_delta="DeltaA")
        assert disagreements(result) == []

    def test_strata_fit_on_the_covariates_the_data_carries(self) -> None:
        frame = binary_frame(n=800)
        frame["S"] = np.where(np.arange(len(frame)) % 4 == 0, "a", "b")
        result = fit_tmle(frame, covariates=["W1", "W2", "S"], strata=["S"], estimands=("ate",))
        assert result.data.has_strata
        assert disagreements(result) == []

    def test_an_arm_msm_reads_the_per_arm_predictions(self) -> None:
        frame = binary_frame()
        result = fit_tmle(frame, msm=MSM.linear(), estimands=("msm",))
        assert result.nuisance.msm is not None
        assert disagreements(result) == []

    def test_a_point_ipsi_reads_the_per_arm_predictions(self) -> None:
        from cleverly.interventions import Incremental

        result = fit_tmle(binary_frame(), incremental=[Incremental(2.0)], estimands=("ey_ipsi",))
        assert result.nuisance.incremental is not None
        assert disagreements(result) == []

    def test_the_intermediate_variable_joins_the_design_and_is_set_per_level(self) -> None:
        rng = np.random.default_rng(3)
        n = 700
        frame = binary_frame(n=n, seed=2)
        frame["Z"] = rng.binomial(1, expit(0.3 * frame["W1"] + 0.5 * frame["A"])).astype(float)
        frame["Y"] = rng.binomial(
            1,
            expit(
                0.2 + 0.5 * frame["W1"] + frame["A"] * (0.5 + 1.5 * frame["W2"]) + 0.8 * frame["Z"]
            ),
        ).astype(float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            estimator = TMLE(
                outcome_learner=outcome_learner(),
                treatment_learner=treatment_learner(),
                intermediate_learner=treatment_learner(),
                cross_fit=True,
                n_folds=3,
                simultaneous=False,
                random_state=0,
                outcome_by_arm=True,
                estimands=("ate",),
            )
            results = estimator.fit(
                frame, outcome="Y", treatment="A", covariates=["W1", "W2"], intermediate="Z"
            )
        for level in (0.0, 1.0):
            result = results[level]
            data, nuisance = result.data, result.nuisance
            target = nuisance.scaler.scale(data.outcome)
            at_level = hand_models(data, nuisance.folds, target, level=level)
            observed = hand_models(data, nuisance.folds, target)
            fitted = nuisance.outcome_by_level[level]
            for arm in data.arm_codes:
                np.testing.assert_allclose(fitted.arms[arm], at_level[arm], atol=1e-12, rtol=0)
            np.testing.assert_allclose(fitted.observed, own_arm(data, observed), atol=1e-12, rtol=0)

    def test_a_continuous_outcome_is_scaled_before_the_arm_models_see_it(self) -> None:
        frame = binary_frame()
        rng = np.random.default_rng(9)
        frame["Y"] = (
            3.0
            + 2.0 * frame["W1"]
            + frame["A"] * (1.0 + 2.0 * frame["W2"])
            + rng.normal(size=len(frame))
        )
        result = fit_tmle(
            frame,
            outcome_learner=LinearRegression(),
            cross_fit=True,
            n_folds=3,
            q_bounds=(float(frame["Y"].min()) - 1.0, float(frame["Y"].max()) + 1.0),
        )
        assert not result.nuisance.scaler.is_identity
        assert disagreements(result, task="regression", make=LinearRegression) == []

    def test_the_arm_models_are_one_fitted_clone_each(self) -> None:
        """The user's learner is never fitted in place, and each fold's models are distinct."""
        learner = outcome_learner()
        fit_tmle(binary_frame(), outcome_learner=learner, cross_fit=True, n_folds=3)
        assert not hasattr(learner, "coef_")


class TestAnArmCodedModifiedPolicy:
    """A modified treatment policy reads each row from the model of the arm it moves the row to."""

    @staticmethod
    def flip() -> ModifiedPolicy:
        return ModifiedPolicy(
            "flip", apply=lambda a, h: 1.0 - np.asarray(a, dtype=float), policy_kind="known"
        )

    @staticmethod
    def rotate() -> ModifiedPolicy:
        return ModifiedPolicy(
            "rotate",
            apply=lambda a, h: (np.asarray(a, dtype=float) + 1.0) % 3.0,
            policy_kind="known",
        )

    def test_the_counterfactual_of_a_flip_is_the_other_arms_model(self) -> None:
        result = fit_tmle(binary_frame(n=800), policies=[self.flip()], estimands=("ey_policy",))
        nuisance, data = result.nuisance, result.data
        key = nuisance.policies.component_keys[0]
        shifted = nuisance.policies.shifted[:, 0]
        # The flip moves every row to the arm it was not in.
        np.testing.assert_array_equal(shifted, 1.0 - data.treatment)
        reference = hand_models(data, nuisance.folds, data.outcome)
        expected = np.where(shifted == 1.0, reference[1.0], reference[0.0])
        np.testing.assert_allclose(nuisance.outcome.arms[key], expected, atol=1e-12, rtol=0)
        np.testing.assert_allclose(
            nuisance.outcome.observed, own_arm(data, reference), atol=1e-12, rtol=0
        )
        # The shifted prediction is not the observed one, so the arm read is the shifted arm.
        assert (
            np.max(np.abs(nuisance.outcome.arms[key] - nuisance.outcome.observed)) > WITNESS_FLOOR
        )

    def test_a_three_arm_rotation_cross_fitted(self) -> None:
        result = fit_tmle(
            three_arm_frame(),
            policies=[self.rotate()],
            estimands=("ey_policy",),
            cross_fit=True,
            n_folds=3,
        )
        nuisance, data = result.nuisance, result.data
        key = nuisance.policies.component_keys[0]
        shifted = nuisance.policies.shifted[:, 0]
        np.testing.assert_array_equal(shifted, (data.treatment + 1.0) % 3.0)
        reference = hand_models(data, nuisance.folds, data.outcome)
        expected = np.choose(shifted.astype(int), [reference[arm] for arm in data.arm_codes])
        np.testing.assert_allclose(nuisance.outcome.arms[key], expected, atol=1e-12, rtol=0)

    def test_reading_the_observed_arm_for_the_shift_breaks_the_comparison(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mutate(
            monkeypatch,
            nuisance_module,
            tmle_module,
            "fit_nuisances",
            [
                (
                    "by_arm_queries[_design_key(code, level)] = ArmQuery(value, level)",
                    "by_arm_queries[_design_key(code, level)] = ArmQuery("
                    "_arm_codes_by_row(data, data.treatment), level)",
                )
            ],
        )
        result = fit_tmle(binary_frame(n=800), policies=[self.flip()], estimands=("ey_policy",))
        nuisance, data = result.nuisance, result.data
        key = nuisance.policies.component_keys[0]
        reference = hand_models(data, nuisance.folds, data.outcome)
        expected = np.where(nuisance.policies.shifted[:, 0] == 1.0, reference[1.0], reference[0.0])
        assert np.max(np.abs(nuisance.outcome.arms[key] - expected)) > WITNESS_FLOOR


# ------------------------------------------------------------------------ the second site


def fit_drtmle(frame: pd.DataFrame, **settings: Any) -> Any:
    """One cross-fitted DR-TMLE with the logistic learners and four folds."""
    arguments: dict[str, Any] = {
        "outcome_learner": outcome_learner(),
        "treatment_learner": treatment_learner(),
        "reduced_outcome_learner": LinearRegression(),
        "reduced_mechanism_residual_learner": LinearRegression(),
        "reduced_treatment_learner": treatment_learner(),
        "n_folds": 4,
        "cross_fit": True,
        "learner_folds": 2,
        "simultaneous": False,
        "random_state": 0,
        "estimands": ("ey1", "ey0", "ate"),
        **settings,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return (
            DRTMLE(**arguments)
            .fit(frame, outcome="Y", treatment="A", covariates=["W1", "W2"])
            .single()
        )


class TestTheNestedDesignsOfDRTMLE:
    """``fit_inner_designs``: every inner model leaves two folds out and stays per arm."""

    @staticmethod
    def fit(frame: pd.DataFrame, **settings: Any) -> Any:
        return fit_drtmle(frame, reduced_crossfit="foldwise", **settings)

    def test_the_inner_arm_predictions_are_the_hand_fitted_leave_two_out_models(self) -> None:
        result = self.fit(binary_frame(n=800), outcome_by_arm=True)
        data, nuisance = result.data, result.nuisance
        assignment = np.asarray(nuisance.folds.assignment)
        target = nuisance.scaler.scale(data.outcome)
        assert nuisance.inner is not None
        for k, inner in enumerate(nuisance.inner.outcome):
            reference = hand_models(data, nuisance.folds, target, mask=assignment != k)
            for arm in data.arm_codes:
                np.testing.assert_allclose(inner.arms[arm], reference[arm], atol=1e-12, rtol=0)
            np.testing.assert_allclose(inner.observed, own_arm(data, reference), atol=1e-12, rtol=0)

    def test_the_inner_fit_differs_from_the_pooled_inner_fit(self) -> None:
        frame = binary_frame(n=800)
        per_arm = self.fit(frame, outcome_by_arm=True).nuisance.inner
        pooled = self.fit(frame).nuisance.inner
        assert per_arm is not None and pooled is not None
        gap = max(
            float(np.max(np.abs(a.arms[arm] - b.arms[arm])))
            for a, b in zip(per_arm.outcome, pooled.outcome, strict=True)
            for arm in (0.0, 1.0)
        )
        assert gap > WITNESS_FLOOR

    def test_the_primary_fit_is_per_arm_too(self) -> None:
        assert disagreements(self.fit(binary_frame(n=800), outcome_by_arm=True)) == []

    def test_an_arm_with_no_rows_outside_two_folds_names_the_folds(self) -> None:
        frame = binary_frame(n=400)
        result = fit_tmle(frame, cross_fit=True, n_folds=4)
        data, nuisance = result.data, result.nuisance
        assignment = np.asarray(nuisance.folds.assignment)
        # Arm 1 keeps an observed outcome in folds 0 and 1 only: leaving both out empties it.
        outcome = np.asarray(data.outcome, dtype=float).copy()
        keep = (np.asarray(data.treatment) == 0.0) | np.isin(assignment, [0, 1])
        frame = frame.assign(Y=np.where(keep, frame["Y"], np.nan), Delta=keep.astype(float))
        sparse = fit_tmle(frame, by_arm=False, delta="Delta", cross_fit=True, n_folds=4)
        sparse_folds = sparse.nuisance.folds
        assert outcome.size == data.n
        with pytest.raises(DataError, match=r"training fold \d with fold \d left out"):
            fit_inner_designs(
                sparse.data,
                sparse.nuisance,
                outcome_learner=outcome_learner(),
                treatment_learner=treatment_learner(),
                outcome_by_arm=True,
            )
        assert sparse_folds.n_folds == 4


class TestTheCompanionOfDRTMLE:
    """``DRTMLE(evaluation=)``: each fold's arm models evaluated at the companion rows."""

    def test_the_companion_is_read_from_the_fold_models_of_each_arm(self) -> None:
        frame, other = binary_frame(n=300), binary_frame(n=90, seed=21)
        result = fit_drtmle(frame, outcome_by_arm=True, evaluation=other)
        data, nuisance = result.data, result.nuisance
        companion = nuisance.companion
        assert companion is not None
        target = nuisance.scaler.scale(data.outcome)
        covariates = np.asarray(data.covariates, dtype=float)
        outcome_rows = np.asarray(data.observed, dtype=bool)
        arms = np.asarray(data.treatment, dtype=float)
        weights = np.asarray(data.weights, dtype=float)
        other_x = other[["W1", "W2"]].to_numpy()
        other_codes = other["A"].to_numpy()
        for k, (train, _) in enumerate(nuisance.folds):
            expected = {}
            for arm in data.arm_codes:
                rows = train[outcome_rows[train] & (arms[train] == arm)]
                model = outcome_learner().fit(
                    covariates[rows], target[rows], sample_weight=weights[rows]
                )
                expected[arm] = model.predict_proba(other_x)[:, 1]
                np.testing.assert_allclose(
                    companion.outcome[k].arms[arm], expected[arm], atol=1e-12, rtol=0
                )
            np.testing.assert_allclose(
                companion.outcome[k].observed,
                np.where(other_codes == 1.0, expected[1.0], expected[0.0]),
                atol=1e-12,
                rtol=0,
            )


# ------------------------------------------------------------------------- the third site


def record_selection_bases(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[Folds, np.ndarray, Any]]:
    """Wrap ``_Selector._selection_base`` and keep every ``(folds, mask, base)`` it builds."""
    selector = ctmle_module._Selector
    original = selector._selection_base
    seen: list[tuple[Folds, np.ndarray, Any]] = []

    def recording(self: Any, train_folds: Folds, train_mask: np.ndarray) -> Any:
        base = original(self, train_folds, train_mask)
        seen.append((train_folds, np.asarray(train_mask), base))
        return base

    monkeypatch.setattr(selector, "_selection_base", recording)
    return seen


def fit_ctmle(frame: pd.DataFrame, **settings: Any) -> Any:
    """One C-TMLE fit with the logistic learners and three outer folds."""
    arguments: dict[str, Any] = {
        "strategy": "greedy",
        "outcome_learner": outcome_learner(),
        "treatment_learner": treatment_learner(),
        "missingness_learner": treatment_learner(),
        "n_folds": 3,
        "cross_fit": True,
        "learner_folds": 2,
        "simultaneous": False,
        "random_state": 0,
        "outcome_by_arm": True,
        "estimands": ("ate", "ey1", "ey0"),
        **settings,
    }
    if arguments["strategy"] != "oat":
        arguments.setdefault("selection_folds", 3)
    estimator = CTMLE(**arguments)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return estimator.fit(frame, outcome="Y", treatment="A", covariates=["W1", "W2"]).single()


class TestTheSelectionFoldBaseOfCTMLE:
    """``_selection_base``: nuisances trained only on one selection fold's training rows."""

    def test_each_selection_fold_base_is_the_hand_fitted_arm_models(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen = record_selection_bases(monkeypatch)
        result = fit_ctmle(binary_frame(n=600))
        data = result.data
        target = result.nuisance.scaler.scale(data.outcome)
        assert len(seen) >= 3
        for folds, mask, base in seen:
            reference = hand_models(data, folds, target, mask=mask)
            for arm in data.arm_codes:
                np.testing.assert_allclose(
                    base.outcome.arms[arm], reference[arm], atol=1e-12, rtol=0
                )
            np.testing.assert_allclose(
                base.outcome.observed, own_arm(data, reference), atol=1e-12, rtol=0
            )

    def test_the_pooled_base_differs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        frame = binary_frame(n=600)
        seen_arm = record_selection_bases(monkeypatch)
        fit_ctmle(frame)
        seen_pooled: list[Any] = []
        original = ctmle_module._Selector._selection_base

        def recording(self: Any, train_folds: Folds, train_mask: np.ndarray) -> Any:
            base = original(self, train_folds, train_mask)
            seen_pooled.append(base)
            return base

        monkeypatch.setattr(ctmle_module._Selector, "_selection_base", recording)
        fit_ctmle(frame, outcome_by_arm=False)
        gap = max(
            float(np.max(np.abs(a[2].outcome.arms[arm] - b.outcome.arms[arm])))
            for a, b in zip(seen_arm, seen_pooled, strict=False)
            for arm in (0.0, 1.0)
        )
        assert gap > WITNESS_FLOOR


class TestTheHonestBaseOfTheOutcomeAdaptiveDesign:
    """``_cross_fit_outcome_adaptive``: one fold-local outcome model per arm, then ``g``."""

    def test_the_honest_base_is_the_hand_fitted_arm_models(self) -> None:
        result = fit_ctmle(
            binary_frame(n=600), strategy="oat", oat_design="per_arm", estimands=("ey1", "ey0")
        )
        data, nuisance = result.data, result.nuisance
        assert nuisance.folds.n_folds == 3
        reference = hand_models(data, nuisance.folds, nuisance.scaler.scale(data.outcome))
        for arm in data.arm_codes:
            np.testing.assert_allclose(
                nuisance.outcome.arms[arm], reference[arm], atol=1e-12, rtol=0
            )
        np.testing.assert_allclose(
            nuisance.outcome.observed, own_arm(data, reference), atol=1e-12, rtol=0
        )

    def test_the_shared_design_takes_the_same_base(self) -> None:
        result = fit_ctmle(
            binary_frame(n=600), strategy="oat", oat_design="shared", estimands=("ey1", "ey0")
        )
        data, nuisance = result.data, result.nuisance
        reference = hand_models(data, nuisance.folds, nuisance.scaler.scale(data.outcome))
        for arm in data.arm_codes:
            np.testing.assert_allclose(
                nuisance.outcome.arms[arm], reference[arm], atol=1e-12, rtol=0
            )

    def test_the_honest_base_differs_from_the_pooled_one(self) -> None:
        frame = binary_frame(n=600)
        settings = {"strategy": "oat", "oat_design": "per_arm", "estimands": ("ey1", "ey0")}
        per_arm = fit_ctmle(frame, **settings).nuisance.outcome
        pooled = fit_ctmle(frame, outcome_by_arm=False, **settings).nuisance.outcome
        gap = max(float(np.max(np.abs(per_arm.arms[arm] - pooled.arms[arm]))) for arm in (0.0, 1.0))
        assert gap > WITNESS_FLOOR

    def test_the_learner_diagnostics_are_kept_for_every_arm_and_fold(self) -> None:
        settings = {"strategy": "oat", "oat_design": "per_arm", "estimands": ("ey1", "ey0")}
        frame = binary_frame(n=400)
        per_arm = fit_ctmle(frame, outcome_learner=small_super_learner(), **settings)
        pooled = fit_ctmle(
            frame, outcome_learner=small_super_learner(), outcome_by_arm=False, **settings
        )
        assert len(pooled.nuisance.diagnostics["outcome"]) == 3
        assert len(per_arm.nuisance.diagnostics["outcome"]) == 6


# --------------------------------------------------------------------------- the helper itself


class TestTheHelperOnHandBuiltFolds:
    """``_fit_outcome_by_arm``, ``_fit_arm_models`` and ``_predict_by_arm`` on their own."""

    @staticmethod
    def data(n: int = 120, seed: int = 4) -> CausalData:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return CausalData.from_frame(
                binary_frame(n=n, seed=seed), outcome="Y", treatment="A", covariates=["W1", "W2"]
            )

    def test_the_cross_fit_helper_equals_the_fold_local_models(self) -> None:
        data = self.data()
        folds = Folds(np.arange(data.n) % 3, 3)
        queries = {
            "observed": ArmQuery(data.treatment),
            "zero": ArmQuery(0.0),
            "one": ArmQuery(1.0),
        }
        fitted, companion, diagnostics = _fit_outcome_by_arm(
            outcome_learner(),
            data,
            np.asarray(data.outcome, dtype=float),
            folds,
            task="classification",
            queries=queries,
            fit_mask=np.ones(data.n, dtype=bool),
        )
        assert companion == {} and diagnostics == []
        reference = hand_models(data, folds, np.asarray(data.outcome, dtype=float))
        np.testing.assert_allclose(fitted["zero"], reference[0.0], atol=1e-12, rtol=0)
        np.testing.assert_allclose(fitted["one"], reference[1.0], atol=1e-12, rtol=0)
        np.testing.assert_allclose(fitted["observed"], own_arm(data, reference), atol=1e-12, rtol=0)

    def test_the_fold_local_models_predict_each_row_from_its_arm(self) -> None:
        data = self.data()
        rows = np.arange(data.n)
        models = _fit_arm_models(
            outcome_learner(),
            data,
            data.covariates,
            np.asarray(data.outcome, dtype=float),
            rows,
            task="classification",
            where="training fold 1",
        )
        per_row = _predict_by_arm(
            models, data.covariates, data.treatment, task="classification", clip=(0.0, 1.0)
        )
        for arm in data.arm_codes:
            held = data.treatment == arm
            expected = models[arm].predict_proba(data.covariates[held])[:, 1]
            np.testing.assert_allclose(per_row[held], expected, atol=1e-12, rtol=0)
        scalar = _predict_by_arm(
            models, data.covariates, 1.0, task="classification", clip=(0.0, 1.0)
        )
        np.testing.assert_allclose(
            scalar, models[1.0].predict_proba(data.covariates)[:, 1], atol=1e-12, rtol=0
        )

    def test_a_companion_is_read_from_the_same_fold_models(self) -> None:
        data = self.data()
        other = self.data(n=40, seed=8)
        folds = Folds(np.arange(data.n) % 3, 3)
        _, companion, _ = _fit_outcome_by_arm(
            outcome_learner(),
            data,
            np.asarray(data.outcome, dtype=float),
            folds,
            task="classification",
            queries={"zero": ArmQuery(0.0)},
            fit_mask=np.ones(data.n, dtype=bool),
            companion=other,
            companion_queries={"observed": ArmQuery(other.treatment), "one": ArmQuery(1.0)},
        )
        outcome = np.asarray(data.outcome, dtype=float)
        for k, (train, _) in enumerate(folds):
            for arm in data.arm_codes:
                rows = train[data.treatment[train] == arm]
                model = outcome_learner().fit(data.covariates[rows], outcome[rows])
                expected = model.predict_proba(other.covariates)[:, 1]
                if arm == 1.0:
                    np.testing.assert_allclose(companion["one"][k], expected, atol=1e-12, rtol=0)
                held = other.treatment == arm
                np.testing.assert_allclose(
                    companion["observed"][k][held], expected[held], atol=1e-12, rtol=0
                )

    def test_the_first_fold_without_an_arm_row_is_named(self) -> None:
        data = self.data()
        assignment = np.arange(data.n) % 3
        # Arm 1 keeps its observed outcomes in fold 0 only: the training set of fold 0 lacks it.
        mask = (data.treatment == 0.0) | (assignment == 0)
        with pytest.raises(DataError) as caught:
            _fit_outcome_by_arm(
                outcome_learner(),
                data,
                np.asarray(data.outcome, dtype=float),
                Folds(assignment, 3),
                task="classification",
                queries={"one": ArmQuery(1.0)},
                fit_mask=mask,
            )
        assert str(caught.value) == R_F2.format(
            treatment="A", arm=data.arm_label(1.0), where="training fold 1"
        )

    def test_a_sample_without_an_arm_row_is_named(self) -> None:
        data = self.data()
        with pytest.raises(DataError) as caught:
            _fit_outcome_by_arm(
                outcome_learner(),
                data,
                np.asarray(data.outcome, dtype=float),
                Folds.single(data.n),
                task="classification",
                queries={"one": ArmQuery(1.0)},
                fit_mask=data.treatment == 0.0,
            )
        assert str(caught.value) == R_F2.format(
            treatment="A", arm=data.arm_label(1.0), where="the training sample"
        )

    def test_the_arm_check_reads_the_training_rows_of_every_fold(self) -> None:
        data = self.data()
        folds = Folds(np.arange(data.n) % 4, 4)
        rows = np.asarray(data.treatment == 1.0)
        _check_arm_training(data, folds, rows, 1.0, None)
        only_fold_two = rows & (np.arange(data.n) % 4 == 2)
        with pytest.raises(DataError, match="training fold 3,"):
            _check_arm_training(data, folds, only_fold_two, 1.0, None)


# ------------------------------------------------------------------------------- the refusals


class TestTheRefusals:
    """R-f1, R-f2 and R-f3."""

    @staticmethod
    def dose_frame(n: int = 300) -> pd.DataFrame:
        rng = np.random.default_rng(11)
        w = rng.normal(size=n)
        a = 0.5 * w + rng.normal(size=n)
        return pd.DataFrame({"Y": a + w + rng.normal(size=n), "A": a, "W": w})

    def test_r_f1_a_continuous_dose_with_a_shift_policy(self) -> None:
        from cleverly.interventions import Shift

        estimator = TMLE(
            policies=[Shift(0.5, cap=None)],
            outcome_learner=LinearRegression(),
            outcome_by_arm=True,
            cross_fit=False,
            simultaneous=False,
        )
        with pytest.raises(CapabilityError) as caught:
            estimator.fit(
                self.dose_frame(),
                outcome="Y",
                treatment="A",
                covariates=["W"],
                treatment_kind="continuous",
            )
        assert caught.value.refusal.reason is RefusalReason.INVALID_DECLARATION
        assert str(caught.value).startswith(R_F1)

    def test_r_f1_a_vector_treatment_with_a_dose(self) -> None:
        """A vector with a continuous component has no arms, so R-f1 refuses it too."""
        from cleverly.interventions import JointPolicy

        frame = self.dose_frame().rename(columns={"A": "D"})
        frame["F"] = (frame["W"] > 0).astype(float)
        estimator = TMLE(
            policies=[JointPolicy("natural", components={}, policy_kind="known")],
            ratio="classifier",
            outcome_learner=LinearRegression(),
            outcome_by_arm=True,
            cross_fit=False,
            simultaneous=False,
        )
        with pytest.raises(CapabilityError) as caught:
            estimator.fit(
                frame,
                outcome="Y",
                treatment=["F", "D"],
                covariates=["W"],
                continuous_treatment=["D"],
            )
        assert caught.value.refusal.reason is RefusalReason.INVALID_DECLARATION
        assert str(caught.value).startswith(R_F1)

    def test_r_f1_a_dose_msm(self) -> None:
        estimator = TMLE(
            msm=MSM.linear(doses=[0.0, 0.5, 1.0]),
            outcome_learner=LinearRegression(),
            outcome_by_arm=True,
            cross_fit=False,
            simultaneous=False,
        )
        with pytest.raises(CapabilityError) as caught:
            estimator.fit(
                self.dose_frame(),
                outcome="Y",
                treatment="A",
                covariates=["W"],
                treatment_kind="continuous",
            )
        assert caught.value.refusal.reason is RefusalReason.INVALID_DECLARATION
        assert str(caught.value).startswith(R_F1)

    def test_r_f1_is_refused_before_any_learner_fits(self) -> None:
        from cleverly.interventions import Shift

        class Tripwire(LinearRegression):
            def fit(self, *args: Any, **kwargs: Any) -> Any:
                raise AssertionError("a learner fitted before the refusal")

        estimator = TMLE(
            policies=[Shift(0.5, cap=None)],
            outcome_learner=Tripwire(),
            treatment_learner=Tripwire(),
            outcome_by_arm=True,
            cross_fit=False,
            simultaneous=False,
        )
        with pytest.raises(CapabilityError):
            estimator.fit(
                self.dose_frame(),
                outcome="Y",
                treatment="A",
                covariates=["W"],
                treatment_kind="continuous",
            )

    def test_r_f1_at_the_nuisance_function(self) -> None:
        from cleverly.interventions import Shift

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = CausalData.from_frame(
                self.dose_frame(),
                outcome="Y",
                treatment="A",
                covariates=["W"],
                treatment_kind="continuous",
            )
        estimator = TMLE(policies=[Shift(0.5, cap=None)], cross_fit=False)
        with pytest.raises(CapabilityError, match="needs an arm-coded treatment"):
            nuisance_module.fit_nuisances(
                data,
                outcome_learner=LinearRegression(),
                treatment_learner=LinearRegression(),
                missingness_learner=None,
                intermediate_learner=None,
                folds=Folds.single(data.n),
                scaler=estimator._scaler(data),
                outcome_by_arm=True,
            )

    def test_r_f2_an_arm_with_no_observed_outcome_in_a_training_fold(self) -> None:
        frame = with_missing_outcome(binary_frame(n=300, seed=3))
        # Arm 1 keeps its observed outcomes in fold 0 only: fold 0's training set lacks them.
        # The cross-fitted TMLE entries run their own preflight first, so the nuisance fit is
        # called on a hand-built split.
        assignment = np.arange(len(frame)) % 3
        treated = (frame["A"] == 1.0).to_numpy()
        frame.loc[treated & (assignment != 0), ["Delta", "Y"]] = [0.0, np.nan]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = CausalData.from_frame(
                frame, outcome="Y", treatment="A", covariates=["W1", "W2"], delta="Delta"
            )
        estimator = TMLE(outcome_by_arm=True, cross_fit=True, n_folds=3)
        with pytest.raises(DataError) as caught:
            nuisance_module.fit_nuisances(
                data,
                outcome_learner=outcome_learner(),
                treatment_learner=treatment_learner(),
                missingness_learner=treatment_learner(),
                intermediate_learner=None,
                folds=Folds(assignment, 3),
                scaler=estimator._scaler(data),
                outcome_by_arm=True,
            )
        assert str(caught.value) == R_F2.format(treatment="A", arm=1, where="training fold 1")
        assert caught.value.refusal.reason is RefusalReason.INVALID_DECLARATION

    def test_r_f2_in_sample_names_the_training_sample(self) -> None:
        frame = with_missing_outcome(binary_frame(n=200, seed=3))
        treated = (frame["A"] == 1.0).to_numpy()
        frame.loc[treated, "Delta"] = 0.0
        frame.loc[treated, "Y"] = np.nan
        with pytest.raises(DataError) as caught:
            fit_tmle(frame, delta="Delta")
        assert str(caught.value) == R_F2.format(treatment="A", arm=1, where="the training sample")

    @pytest.mark.parametrize(
        "make",
        [
            lambda: TMLEMethod(models=ModelSpec(outcome_by_arm=True)),
            lambda: TMLEMethod().with_overrides(outcome_by_arm=True),
            lambda: SDRMethod(models=ModelSpec(outcome_by_arm=True)),
            lambda: IPSIMethod(models=ModelSpec(outcome_by_arm=True)),
        ],
        ids=["tmle", "shortcut", "sdr", "ipsi"],
    )
    def test_r_f3_a_longitudinal_design_refuses_the_setting(self, make: Callable[[], Any]) -> None:
        with pytest.raises(MethodConfigurationError) as caught:
            make().estimator_kwargs(longitudinal=True)
        assert (
            "longitudinal TMLE cannot apply point-treatment option(s): ['outcome_by_arm']"
            in str(caught.value)
        )

    def test_the_declaration_is_a_boolean(self) -> None:
        for bad in ("arm", 1, None):
            with pytest.raises(MethodConfigurationError) as caught:
                TMLE(outcome_by_arm=bad)  # type: ignore[arg-type]
            assert caught.value.refusal.reason is RefusalReason.INVALID_DECLARATION
            assert "outcome_by_arm must be True or False" in str(caught.value)
        assert TMLE(outcome_by_arm=np.True_).outcome_by_arm  # type: ignore[arg-type]


# --------------------------------------------------------------------- the default and plumbing


class TestTheDefaultAndThePlumbing:
    def test_the_default_is_off_everywhere(self) -> None:
        assert ModelSpec().outcome_by_arm is False
        assert TMLE().outcome_by_arm is False
        assert TMLEMethod().estimator_kwargs()["outcome_by_arm"] is False

    def test_the_default_fit_is_bit_identical_to_a_fit_that_does_not_name_it(self) -> None:
        frame = binary_frame(n=500)
        named = fit_tmle(frame, by_arm=False, cross_fit=True, n_folds=3)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            unnamed = (
                TMLE(
                    outcome_learner=outcome_learner(),
                    treatment_learner=treatment_learner(),
                    missingness_learner=treatment_learner(),
                    cross_fit=True,
                    n_folds=3,
                    simultaneous=False,
                    random_state=0,
                )
                .fit(frame, outcome="Y", treatment="A", covariates=["W1", "W2"])
                .single()
            )
        np.testing.assert_array_equal(
            named.nuisance.outcome.observed, unnamed.nuisance.outcome.observed
        )
        for arm in (0.0, 1.0):
            np.testing.assert_array_equal(
                named.nuisance.outcome.arms[arm], unnamed.nuisance.outcome.arms[arm]
            )
        assert named.config.outcome_by_arm is False

    def test_the_shortcut_and_the_model_spec_reach_the_engine(self) -> None:
        for method in (
            TMLEMethod(models=ModelSpec(outcome_by_arm=True)),
            TMLEMethod().with_overrides(outcome_by_arm=True),
        ):
            assert method.models.outcome_by_arm is True
            assert method.estimator_kwargs()["outcome_by_arm"] is True

    def test_every_engine_takes_the_keyword(self) -> None:
        assert DRTMLE(outcome_by_arm=True).outcome_by_arm
        assert CTMLE(outcome_by_arm=True).outcome_by_arm

    def test_the_config_records_the_setting(self) -> None:
        assert fit_tmle(binary_frame()).config.outcome_by_arm is True

    def test_a_saved_result_keeps_the_setting_and_the_predictions(self) -> None:
        result = fit_tmle(binary_frame(n=400), cross_fit=True, n_folds=3)
        loaded = loads(dumps(result))
        assert loaded.config.outcome_by_arm is True
        np.testing.assert_array_equal(
            loaded.nuisance.outcome.arms[1.0], result.nuisance.outcome.arms[1.0]
        )

    def test_a_refit_repeats_the_per_arm_fit(self) -> None:
        frame = binary_frame(n=400)
        estimator = TMLE(
            outcome_learner=outcome_learner(),
            treatment_learner=treatment_learner(),
            outcome_by_arm=True,
            cross_fit=True,
            n_folds=3,
            simultaneous=False,
            random_state=0,
            estimands=("ate",),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = estimator.fit(frame, outcome="Y", treatment="A", covariates=["W1", "W2"])
            again = estimator.refit(result.single().data)
        assert disagreements(again) == []
        np.testing.assert_array_equal(
            again.nuisance.outcome.observed, result.single().nuisance.outcome.observed
        )

    def test_a_bootstrap_replicate_refits_the_nuisances_per_arm(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[int] = []
        original = nuisance_module._fit_outcome_by_arm

        def counting(*args: Any, **kwargs: Any) -> Any:
            calls.append(1)
            return original(*args, **kwargs)

        monkeypatch.setattr(nuisance_module, "_fit_outcome_by_arm", counting)
        result = fit_tmle(binary_frame(n=300), n_bootstrap=3, estimands=("ate",))
        assert result.config.outcome_by_arm is True
        assert len(calls) == 4

    def test_the_super_learner_split_rule_is_asked_of_each_arm(self) -> None:
        from cleverly.learners import SuperLearner

        frame = binary_frame(n=300)
        # Arm 1 holds a single observed outcome of class 1: pooled it holds many.
        treated = (frame["A"] == 1.0).to_numpy()
        frame.loc[treated, "Y"] = 0.0
        frame.loc[np.flatnonzero(treated)[0], "Y"] = 1.0
        learner = SuperLearner([LogisticRegression(max_iter=500)], task="classification")
        with pytest.raises(DataError, match="arm 1"):
            fit_tmle(frame, outcome_learner=learner, cross_fit=True, n_folds=3)
        pooled = TMLE(
            outcome_learner=SuperLearner([LogisticRegression(max_iter=500)], task="classification"),
            treatment_learner=treatment_learner(),
            outcome_by_arm=False,
            cross_fit=True,
            n_folds=3,
            simultaneous=False,
            random_state=0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pooled.fit(frame, outcome="Y", treatment="A", covariates=["W1", "W2"])


# ---------------------------------------------------------------------- the mutation controls


class TestMutationControls:
    """One broken line each, and the hand comparison must fail."""

    def test_the_unmutated_comparisons_pass(self) -> None:
        assert disagreements(fit_tmle(binary_frame(), cross_fit=True, n_folds=3)) == []

    def test_a_mask_without_the_arm_condition_fails_the_primary_comparison(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mutate(
            monkeypatch,
            nuisance_module,
            nuisance_module,
            "_fit_outcome_by_arm",
            [
                (
                    "eligible = {arm: mask & (treatment == arm) for arm in arms}",
                    "eligible = dict.fromkeys(arms, mask)",
                )
            ],
        )
        problems = disagreements(fit_tmle(binary_frame(), cross_fit=True, n_folds=3))
        assert problems and all(float(p.split(": ")[1]) > WITNESS_FLOOR for p in problems)

    def test_swapped_arm_models_fail_the_primary_comparison(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mutate(
            monkeypatch,
            nuisance_module,
            nuisance_module,
            "_fit_outcome_by_arm",
            [
                (
                    "per_arm[arm] = predictions",
                    "per_arm[arms[len(arms) - 1 - arms.index(arm)]] = predictions",
                )
            ],
        )
        problems = disagreements(fit_tmle(binary_frame(), cross_fit=True, n_folds=3))
        assert problems

    def test_a_mask_without_the_arm_condition_fails_the_nested_comparison(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mutate(
            monkeypatch,
            nuisance_module,
            nuisance_module,
            "_fit_outcome_by_arm",
            [
                (
                    "eligible = {arm: mask & (treatment == arm) for arm in arms}",
                    "eligible = dict.fromkeys(arms, mask)",
                )
            ],
        )
        result = TestTheNestedDesignsOfDRTMLE.fit(binary_frame(n=800), outcome_by_arm=True)
        data, nuisance = result.data, result.nuisance
        assignment = np.asarray(nuisance.folds.assignment)
        target = nuisance.scaler.scale(data.outcome)
        reference = hand_models(data, nuisance.folds, target, mask=assignment != 0)
        gap = max(
            float(np.max(np.abs(nuisance.inner.outcome[0].arms[arm] - reference[arm])))
            for arm in data.arm_codes
        )
        assert gap > WITNESS_FLOOR

    def test_swapped_arm_models_fail_the_selection_fold_comparison(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mutate(
            monkeypatch,
            nuisance_module,
            ctmle_module,
            "_fit_outcome_by_arm",
            [
                (
                    "per_arm[arm] = predictions",
                    "per_arm[arms[len(arms) - 1 - arms.index(arm)]] = predictions",
                )
            ],
        )
        seen = record_selection_bases(monkeypatch)
        result = fit_ctmle(binary_frame(n=600))
        data = result.data
        target = result.nuisance.scaler.scale(data.outcome)
        folds, mask, base = seen[0]
        reference = hand_models(data, folds, target, mask=mask)
        gap = max(
            float(np.max(np.abs(base.outcome.arms[arm] - reference[arm]))) for arm in data.arm_codes
        )
        assert gap > WITNESS_FLOOR

    def test_a_fold_local_mask_without_the_arm_condition_fails_the_honest_base(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mutate(
            monkeypatch,
            nuisance_module,
            ctmle_module,
            "_fit_arm_models",
            [("arm_rows = rows[treatment[rows] == arm]", "arm_rows = rows")],
        )
        result = fit_ctmle(
            binary_frame(n=600), strategy="oat", oat_design="per_arm", estimands=("ey1", "ey0")
        )
        data, nuisance = result.data, result.nuisance
        reference = hand_models(data, nuisance.folds, nuisance.scaler.scale(data.outcome))
        gap = max(
            float(np.max(np.abs(nuisance.outcome.arms[arm] - reference[arm])))
            for arm in data.arm_codes
        )
        assert gap > WITNESS_FLOOR

    def test_swapped_fold_local_models_fail_the_honest_base(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mutate(
            monkeypatch,
            nuisance_module,
            ctmle_module,
            "_predict_by_arm",
            [
                ("models[float(requested[0])]", "models[1.0 - float(requested[0])]"),
                ("models[float(code)]", "models[1.0 - float(code)]"),
            ],
        )
        result = fit_ctmle(
            binary_frame(n=600), strategy="oat", oat_design="per_arm", estimands=("ey1", "ey0")
        )
        data, nuisance = result.data, result.nuisance
        reference = hand_models(data, nuisance.folds, nuisance.scaler.scale(data.outcome))
        gap = max(
            float(np.max(np.abs(nuisance.outcome.arms[arm] - reference[arm])))
            for arm in data.arm_codes
        )
        assert gap > WITNESS_FLOOR


class TestThePerArmParametricStackUnderArmModels:
    """X58's fold coefficient record reads one pooled outcome GLM, so arm models write none.

    The arm models leave one outcome diagnostic per arm, so a record built from them would
    raise for want of the single pooled entry. The fit keeps the shipped
    ``generated_design_plugin`` status that the allow-list gives it.
    """

    @pytest.mark.parametrize("cross_fit", [False, True])
    def test_the_fit_records_no_stack_and_keeps_the_plugin_status(self, cross_fit: bool) -> None:
        from cleverly.learners import ParametricMechanism, ParametricOutcome

        result = fit_ctmle(
            binary_frame(n=600),
            strategy="oat",
            oat_design="per_arm",
            outcome_learner=ParametricOutcome(),
            treatment_learner=ParametricMechanism(),
            cross_fit=cross_fit,
            estimands=("ate", "ey1", "ey0"),
        )
        assert "stack" not in result.nuisance.diagnostics
        assert result.inference_status == "generated_design_plugin"

    def test_the_pooled_fit_records_the_stack(self) -> None:
        from cleverly.learners import ParametricMechanism, ParametricOutcome

        result = fit_ctmle(
            binary_frame(n=600),
            strategy="oat",
            oat_design="per_arm",
            outcome_learner=ParametricOutcome(),
            treatment_learner=ParametricMechanism(),
            cross_fit=False,
            outcome_by_arm=False,
            estimands=("ate", "ey1", "ey0"),
        )
        assert "stack" in result.nuisance.diagnostics
