"""``ModelSpec(hazard_pooling="pooled")``: one censoring regression over every node (X33 j).

The pooled fit stacks the at-risk ``(unit, node)`` rows of every node whose retention is not
declared known, with the node's grid time as the last column.  The tests check the stacked
rows, the fold integrity of each unit, and the predictions against a hand-fit pooled logistic
regression.  Mutation controls that drop the time column or keep the rows after the event
each fail one of them.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import CausalStudy, LongitudinalTreatment, ModelSpec, RegimeMean, TMLEMethod
from cleverly.datasets import make_longitudinal, make_point_survival
from cleverly.exceptions import CapabilityError, MethodConfigurationError, RefusalReason
from cleverly.longitudinal import LTMLE, SDR, LongitudinalResult, pooled_hazards
from cleverly.longitudinal.pooled_hazards import stacked_design

NODES = 4
REGIMENS = {"never": 0, "always": 1}
COLUMNS: dict[str, Any] = {
    "outcome": [f"Y{t}" for t in range(1, NODES + 1)],
    "treatment": "A",
    "baseline": ["ID", "W1", "W2"],
    "censoring": [f"C{t}" for t in range(1, NODES + 1)],
}


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    data, _ = make_point_survival(n=400, seed=3, layout="wide", n_times=NODES)
    return data.assign(ID=np.arange(len(data), dtype=float))[["ID", *data.columns]]


def _logistic() -> LogisticRegression:
    return LogisticRegression(C=1e6, max_iter=5000, tol=1e-12)


def _fit(
    frame: pd.DataFrame, learner: Any, *, folds: int = 1, **settings: Any
) -> LongitudinalResult:
    return LTMLE(
        REGIMENS,
        reference="never",
        n_folds=folds,
        random_state=0,
        outcome_learner=_logistic(),
        pseudo_learner=LinearRegression(),
        treatment_learner=_logistic(),
        censoring_learner=learner,
        simultaneous=False,
        **settings,
    ).fit(frame, **COLUMNS)


class Recorder(BaseEstimator, ClassifierMixin):
    """A logistic regression that records each design and target it is fitted on."""

    calls: ClassVar[list[tuple[np.ndarray, np.ndarray]]] = []

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> Recorder:
        Recorder.calls.append((np.asarray(X, dtype=float), np.asarray(y, dtype=float)))
        self.model_ = _logistic().fit(X, y, sample_weight=sample_weight)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        return self.model_.predict_proba(X)


class Memorizer(BaseEstimator, ClassifierMixin):
    """Predicts retention one on a unit it trained on and one half elsewhere.

    The unit id is the first design column, so an out-of-fold prediction of one shows that a
    unit's own rows trained the model that scored it.
    """

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> Memorizer:
        self.seen_ = set(np.asarray(X, dtype=float)[:, 0].tolist())
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        ids = np.asarray(X, dtype=float)[:, 0]
        one = np.where(np.isin(ids, list(self.seen_)), 1.0, 0.5)
        return np.column_stack([1.0 - one, one])


def _at_risk(result: LongitudinalResult, time: int) -> np.ndarray:
    data = result.data
    return data.uncensored_through(time - 1) & data.event_free_through(time - 1)


def _hand_stack(result: LongitudinalResult) -> tuple[np.ndarray, np.ndarray]:
    data = result.data
    designs, targets = [], []
    for time in range(1, NODES + 1):
        rows = _at_risk(result, time)
        design = data.history_design(time, include_current=True)[rows]
        designs.append(np.column_stack([design, np.full(rows.sum(), float(time))]))
        targets.append(data.uncensored[rows, time - 1].astype(float))
    return np.vstack(designs), np.concatenate(targets)


def _stack_matches(frame: pd.DataFrame) -> bool:
    Recorder.calls = []
    result = _fit(frame, Recorder(), hazard_pooling="pooled")
    ((design, target),) = Recorder.calls
    expected_design, expected_target = _hand_stack(result)
    return (
        design.shape == expected_design.shape
        and np.array_equal(design, expected_design)
        and np.array_equal(target, expected_target)
    )


def _predictions_match(frame: pd.DataFrame) -> bool:
    result = _fit(frame, _logistic(), hazard_pooling="pooled")
    design, target = _hand_stack(result)
    model = _logistic().fit(design, target)
    data = result.data
    for time in range(1, NODES + 1):
        full = np.column_stack(
            [data.history_design(time, include_current=True), np.full(data.n, float(time))]
        )
        expected = model.predict_proba(full)[:, 1]
        rows = _at_risk(result, time)
        if not np.allclose(
            result.mechanism.censoring_observed[time - 1][rows], expected[rows], rtol=1e-6, atol=0
        ):
            return False
    return True


class TestThePooledFit:
    def test_the_stacked_rows_are_the_at_risk_rows(self, frame: pd.DataFrame) -> None:
        assert _stack_matches(frame)

    def test_the_predictions_are_the_hand_fit_pooled_regression(self, frame: pd.DataFrame) -> None:
        assert _predictions_match(frame)

    def test_a_unit_keeps_its_fold_at_every_node(self, frame: pd.DataFrame) -> None:
        result = _fit(frame, Memorizer(), folds=3, hazard_pooling="pooled")
        for time in range(1, NODES + 1):
            rows = _at_risk(result, time)
            np.testing.assert_array_equal(result.mechanism.censoring_observed[time - 1][rows], 0.5)

    def test_the_stacked_design_sets_each_plan_arm(self, frame: pd.DataFrame) -> None:
        result = _fit(frame, _logistic(), hazard_pooling="pooled")
        data = result.data
        plan = next(item for item in result.replay_recipe.plans if item.label == "always")
        stacked = stacked_design(data, (2, 3), plan)
        assert stacked.shape == (
            2 * data.n,
            data.history_design(2, include_current=True).shape[1] + 1,
        )
        np.testing.assert_array_equal(stacked[: data.n, -1], 2.0)
        np.testing.assert_array_equal(stacked[data.n :, -1], 3.0)
        np.testing.assert_array_equal(stacked[:, -2], 1.0)

    def test_pooled_and_per_node_fits_differ(self, frame: pd.DataFrame) -> None:
        node = _fit(frame, _logistic())
        pooled = _fit(frame, _logistic(), hazard_pooling="pooled")
        rows = _at_risk(node, 3)
        assert not np.allclose(
            node.mechanism.censoring_observed[2][rows], pooled.mechanism.censoring_observed[2][rows]
        )
        last = f"always @ t={NODES}"
        assert node.fits[last].psi_scaled != pooled.fits[last].psi_scaled


class TestSDR:
    """A cross-fitted ``SDR`` fit pools each fold's own censoring model on its training rows."""

    def test_each_fold_mechanism_pools_its_training_rows(self, frame: pd.DataFrame) -> None:
        Recorder.calls = []
        result = SDR(
            REGIMENS,
            reference="never",
            n_folds=2,
            random_state=0,
            outcome_learner=_logistic(),
            pseudo_learner=LinearRegression(),
            treatment_learner=_logistic(),
            censoring_learner=Recorder(),
            simultaneous=False,
            hazard_pooling="pooled",
        ).fit(frame, **COLUMNS)
        design, target = _hand_stack(result)
        ids = design[:, 0].astype(int)
        # The out-of-fold mechanism fits once per fold, and each fold mechanism once more.
        assert len(Recorder.calls) == 2 * result.folds.n_folds
        stacks = {}
        for fold, (train, _) in enumerate(result.folds):
            rows = np.isin(ids, train)
            stacks[fold] = (design[rows], target[rows])
        folds_seen = []
        for fitted, fitted_target in Recorder.calls:
            (fold,) = (
                fold
                for fold, (expected, expected_target) in stacks.items()
                if np.array_equal(fitted, expected)
                and np.array_equal(fitted_target, expected_target)
            )
            folds_seen.append(fold)
        assert sorted(folds_seen) == [0, 0, 1, 1]


class TestMutationControls:
    def test_dropping_the_time_column_fails_the_hand_fit(
        self, frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert _predictions_match(frame)
        monkeypatch.setattr(pooled_hazards, "node_time", lambda data, time: 0.0)
        assert not _predictions_match(frame)

    def test_keeping_the_rows_after_the_event_fails_the_stack(
        self, frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert _stack_matches(frame)
        original = pooled_hazards.fit_pooled_censoring

        def keep_post_event(data: Any, plans: Any, nodes: Any, at_risk: Any, **kwargs: Any) -> Any:
            widened = {time: data.uncensored_through(time - 1) for time in at_risk}
            return original(data, plans, nodes, widened, **kwargs)

        from cleverly.longitudinal import sequential

        monkeypatch.setattr(sequential, "fit_pooled_censoring", keep_post_event)
        assert not _stack_matches(frame)


class TestRefusals:
    def test_an_unknown_pooling(self) -> None:
        with pytest.raises(ValueError, match="hazard_pooling must be 'node' or 'pooled'"):
            ModelSpec(hazard_pooling="cause")
        with pytest.raises(ValueError, match="hazard_pooling must be"):
            LTMLE(REGIMENS, hazard_pooling="both")

    def test_a_decision_at_every_node(self) -> None:
        frame, _ = make_longitudinal(300, seed=0)
        with pytest.raises(CapabilityError, match="a treatment decision at each of its 2 nodes"):
            LTMLE({"always": 1}, n_folds=1, hazard_pooling="pooled").fit(
                frame,
                outcome="Y",
                treatment=["A1", "A2"],
                baseline=["W1", "W2"],
                time_varying=[[], ["L2"]],
                censoring=["C1", "C2"],
            )

    def test_time_varying_covariates(self) -> None:
        frame, _ = make_point_survival(n=300, seed=1, layout="wide", n_times=3, time_varying=True)
        with pytest.raises(CapabilityError, match="time-varying covariates, so its node designs"):
            LTMLE({"always": 1}, n_folds=1, hazard_pooling="pooled").fit(
                frame,
                outcome=["Y1", "Y2", "Y3"],
                treatment="A",
                baseline=["W1", "W2"],
                time_varying=[[], ["L2"], ["L3"]],
                censoring=["C1", "C2", "C3"],
            )

    def test_a_point_treatment_fit(self) -> None:
        method = TMLEMethod(models=ModelSpec(hazard_pooling="pooled"))
        with pytest.raises(
            MethodConfigurationError,
            match=r"cannot apply longitudinal option\(s\): \['hazard_pooling'\]",
        ) as caught:
            method.estimator_kwargs(longitudinal=False)
        assert caught.value.refusal.reason is RefusalReason.INVALID_DECLARATION
        assert caught.value.refusal.alternative == (
            "Remove it, or use a LongitudinalTreatment design."
        )
        assert method.estimator_kwargs(longitudinal=True)["hazard_pooling"] == "pooled"


def test_the_study_path_reaches_the_pooled_fit(frame: pd.DataFrame) -> None:
    design = LongitudinalTreatment(
        outcome=COLUMNS["outcome"],
        treatment="A",
        baseline=COLUMNS["baseline"],
        censoring=COLUMNS["censoring"],
    )
    pooled = (
        CausalStudy(frame, design=design)
        .identify(RegimeMean({"always": 1}))
        .estimate(
            outcome_learner=_logistic(),
            pseudo_learner=LinearRegression(),
            treatment_learner=_logistic(),
            censoring_learner=_logistic(),
            cross_fit=False,
            hazard_pooling="pooled",
        )
    )
    direct = _fit(frame, _logistic(), hazard_pooling="pooled")
    name = f"risk_regimen[always @ t={NODES}]"
    assert pooled.estimates[name].psi == pytest.approx(direct[name].psi, rel=1e-9)
