"""``hazard_pooling="pooled"`` pools each cause's hazard under ``survival_model="hazard"`` (X33 j).

Cause :math:`j` fits its pseudo-hazard :math:`\\rho_j` once over the eligible rows of every
node: at risk at the node, with no event of an earlier cause there.  The pooled design is the
node's design plus the node's grid time.  The tests check the stacked rows, the fold integrity
of each unit, and the hazards against a hand-fit pooled logistic regression.  Mutation controls
that drop the time column or keep the rows after the event each fail one of them.  The
per-node path is pinned against a hand fit at each node.
"""

from __future__ import annotations

import warnings
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

from cleverly.datasets import make_point_survival
from cleverly.learners import GLM
from cleverly.learners.crossfit import Folds
from cleverly.longitudinal import LTMLE, LongitudinalResult, pooled_hazards
from cleverly.longitudinal import hazard as hazard_module
from cleverly.longitudinal.hazard import event_indicators, fit_hazards

NODES = 4
REGIMENS = {"never": 0, "always": 1}
COLUMNS: dict[str, Any] = {
    "outcome": {
        "relapse": [f"R{t}" for t in range(1, NODES + 1)],
        "death": [f"D{t}" for t in range(1, NODES + 1)],
    },
    "treatment": "A",
    "baseline": ["ID", "W1", "W2"],
    "censoring": [f"C{t}" for t in range(1, NODES + 1)],
}


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    data, _ = make_point_survival(
        n=500,
        seed=5,
        layout="wide",
        causes=2,
        n_times=NODES,
        hazard_scale={1: 0.2, 2: 0.4, 3: 0.7},
    )
    return data.assign(ID=np.arange(len(data), dtype=float))[["ID", *data.columns]]


class Recorder(BaseEstimator, ClassifierMixin):
    """A GLM that records each design, target and weight vector it is fitted on."""

    calls: ClassVar[list[tuple[np.ndarray, np.ndarray]]] = []

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> Recorder:
        Recorder.calls.append((np.asarray(X, dtype=float), np.asarray(y, dtype=float)))
        self.model_ = GLM().fit(np.asarray(X, dtype=float), y, sample_weight=sample_weight)
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        one = np.asarray(self.model_.predict_proba(X), dtype=float).reshape(len(X), -1)[:, -1]
        return np.column_stack([1.0 - one, one])


class Memorizer(BaseEstimator, ClassifierMixin):
    """Predicts rate one on a unit it trained on and one half elsewhere (ID is column 0)."""

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> Memorizer:
        self.seen_ = set(np.asarray(X, dtype=float)[:, 0].tolist())
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        ids = np.asarray(X, dtype=float)[:, 0]
        one = np.where(np.isin(ids, list(self.seen_)), 1.0, 0.5)
        return np.column_stack([1.0 - one, one])


def _fit(
    frame: pd.DataFrame, learner: Any, *, folds: int = 1, **settings: Any
) -> LongitudinalResult:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return LTMLE(
            REGIMENS,
            reference="never",
            n_folds=folds,
            random_state=0,
            survival_model="hazard",
            outcome_learner=learner,
            treatment_learner=GLM(),
            censoring_learner=GLM(),
            simultaneous=False,
            **settings,
        ).fit(frame, **COLUMNS)


def _eligible(result: LongitudinalResult) -> list[np.ndarray]:
    """Per cause, the ``(n, T)`` eligible rows: at risk with no earlier-cause event."""
    at_risk, events = event_indicators(result.data)
    earlier = np.zeros_like(at_risk)
    out = []
    for cause in range(events.shape[2]):
        out.append(at_risk & ~earlier)
        earlier = earlier | (events[:, :, cause] > 0)
    return out


def _hand_stack(result: LongitudinalResult, cause: int) -> tuple[np.ndarray, np.ndarray]:
    data = result.data
    _, events = event_indicators(data)
    rows = _eligible(result)[cause]
    designs, targets = [], []
    for time in range(1, NODES + 1):
        keep = rows[:, time - 1]
        design = data.history_design(time, include_current=True)[keep]
        designs.append(np.column_stack([design, np.full(keep.sum(), float(time))]))
        targets.append(events[keep, time - 1, cause])
    return np.vstack(designs), np.concatenate(targets)


def _stack_matches(frame: pd.DataFrame) -> bool:
    Recorder.calls = []
    result = _fit(frame, Recorder(), hazard_pooling="pooled")
    if len(Recorder.calls) != 2:
        return False
    for cause, (design, target) in enumerate(Recorder.calls):
        expected_design, expected_target = _hand_stack(result, cause)
        if design.shape != expected_design.shape:
            return False
        if not (
            np.array_equal(design, expected_design) and np.array_equal(target, expected_target)
        ):
            return False
    return True


def _hand_hazards(result: LongitudinalResult) -> np.ndarray:
    """The pooled hazards from two hand-fit GLMs and the sequential rule."""
    data = result.data
    levels = result.replay_recipe.hazard.levels
    out = np.zeros((data.n, NODES, len(levels), 2))
    taken = np.zeros((data.n, NODES, len(levels)))
    for cause in range(2):
        design, target = _hand_stack(result, cause)
        model = GLM().fit(design, target)
        for position, plan in enumerate(levels):
            for time in range(1, NODES + 1):
                full = np.column_stack(
                    [
                        data.history_design(time, treatment=plan.values, include_current=True),
                        np.full(data.n, float(time)),
                    ]
                )
                rho = np.asarray(model.predict_proba(full), dtype=float).reshape(data.n, -1)[:, -1]
                out[:, time - 1, position, cause] = rho * (1.0 - taken[:, time - 1, position])
        taken = taken + out[:, :, :, cause]
    return out


def _predictions_match(frame: pd.DataFrame) -> bool:
    result = _fit(frame, GLM(), hazard_pooling="pooled")
    assert result.initial_hazards is not None
    return bool(
        np.allclose(result.initial_hazards.hazards, _hand_hazards(result), rtol=0, atol=1e-10)
    )


class TestThePooledFit:
    def test_each_cause_pools_the_eligible_rows_of_every_node(self, frame: pd.DataFrame) -> None:
        Recorder.calls = []
        result = _fit(frame, Recorder(), hazard_pooling="pooled")
        assert len(Recorder.calls) == 2
        for cause, (design, target) in enumerate(Recorder.calls):
            rows = _eligible(result)[cause]
            assert design.shape[0] == int(rows.sum())
            expected_design, expected_target = _hand_stack(result, cause)
            np.testing.assert_array_equal(design, expected_design)
            np.testing.assert_array_equal(target, expected_target)
            # The row identity: the unit id and the node, per stacked row.
            ids = [np.flatnonzero(rows[:, t]) for t in range(NODES)]
            np.testing.assert_array_equal(design[:, 0], np.concatenate(ids).astype(float))
            np.testing.assert_array_equal(
                design[:, -1], np.concatenate([np.full(len(i), t + 1.0) for t, i in enumerate(ids)])
            )

    def test_a_later_cause_leaves_out_the_rows_of_an_earlier_event(
        self, frame: pd.DataFrame
    ) -> None:
        result = _fit(frame, GLM(), hazard_pooling="pooled")
        first, second = _eligible(result)
        assert second.sum() < first.sum()

    def test_the_hazards_are_the_hand_fit_pooled_regression(self, frame: pd.DataFrame) -> None:
        assert _predictions_match(frame)

    def test_a_unit_keeps_its_fold_at_every_node(self, frame: pd.DataFrame) -> None:
        result = _fit(frame, Memorizer(), folds=3, hazard_pooling="pooled")
        assert result.initial_hazards is not None
        # Every prediction is out of fold, so a unit whose rows trained its own scorer at
        # any node would read one.  The first cause's rho is its hazard.
        np.testing.assert_array_equal(result.initial_hazards.hazards[:, :, :, 0], 0.5)

    def test_a_cause_without_an_event_pools_nothing(self, frame: pd.DataFrame) -> None:
        deaths = [f"D{t}" for t in range(1, NODES + 1)]
        silent = frame[~(frame[deaths] == 1).any(axis=1)]
        Recorder.calls = []
        result = _fit(silent, Recorder(), hazard_pooling="pooled")
        assert result.initial_hazards is not None
        assert len(Recorder.calls) == 1
        np.testing.assert_array_equal(result.initial_hazards.hazards[..., 1], 0.0)
        assert result.initial_hazards.no_event_nodes == tuple((t, 1) for t in range(1, NODES + 1))

    def test_the_per_node_path_is_the_hand_fit_at_each_node(self, frame: pd.DataFrame) -> None:
        result = _fit(frame, GLM())
        assert result.initial_hazards is not None
        data = result.data
        levels = result.replay_recipe.hazard.levels
        at_risk, events = event_indicators(data)
        folds = Folds(np.zeros(data.n, dtype=np.int64), 1)
        again = fit_hazards(data, levels, learner=GLM(), folds=folds, hazard_pooling="node")
        np.testing.assert_array_equal(again.hazards, result.initial_hazards.hazards)
        eligible = _eligible(result)
        for time in range(1, NODES + 1):
            rows = eligible[0][:, time - 1]
            model = GLM().fit(
                data.history_design(time, include_current=True)[rows], events[rows, time - 1, 0]
            )
            rho = np.asarray(
                model.predict_proba(
                    data.history_design(time, treatment=levels[1].values, include_current=True)
                ),
                dtype=float,
            ).reshape(data.n, -1)[:, -1]
            np.testing.assert_allclose(
                result.initial_hazards.hazards[:, time - 1, 1, 0], rho, rtol=0, atol=1e-10
            )
        assert at_risk.shape == (data.n, NODES)


class TestTheTimeTrendWitness:
    def test_pooled_and_per_node_hazards_differ_and_the_pooled_fit_solves_its_score(self) -> None:
        sample, _ = make_point_survival(
            n=600, seed=1, layout="wide", n_times=5, hazard_scale={1: 0.3, 2: 0.5}
        )
        columns: dict[str, Any] = {
            "outcome": [f"Y{k}" for k in range(1, 6)],
            "treatment": "A",
            "baseline": ["W1", "W2"],
            "censoring": [f"C{k}" for k in range(1, 6)],
        }
        fits = {}
        for pooling in ("node", "pooled"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fits[pooling] = LTMLE(
                    REGIMENS,
                    n_folds=1,
                    random_state=0,
                    survival_model="hazard",
                    hazard_pooling=pooling,
                    outcome_learner=GLM(),
                    treatment_learner=GLM(),
                    censoring_learner=GLM(),
                ).fit(sample, **columns)
        node, pooled = fits["node"].initial_hazards, fits["pooled"].initial_hazards
        assert node is not None and pooled is not None
        assert np.max(np.abs(node.hazards - pooled.hazards)) > 1e-3
        assert pooled.hazards.shape == node.hazards.shape
        assert pooled.hazards.min() >= 0.0
        assert pooled.hazards.max() < 1.0
        assert all(row.passed for row in fits["pooled"].diagnostics.score_equations().rows)
        assert fits["pooled"].converged


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
        original = event_indicators

        def keep_post_event(data: Any) -> Any:
            _, events = original(data)
            widened = np.column_stack([data.uncensored_through(t) for t in range(NODES)])
            return widened, events

        monkeypatch.setattr(hazard_module, "event_indicators", keep_post_event)
        assert not _stack_matches(frame)
