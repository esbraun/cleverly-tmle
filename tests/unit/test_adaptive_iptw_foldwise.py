r"""The fold-wise reduced regression of a cross-fitted AdaptiveIPTW, against test-side values.

Under cross-fitting, fold ``k``'s reduced regression trains on the rows of the other folds.  At a
training row of fold ``m`` its design is ``g^{(-k,-m)}(W)``, from a mechanism model that saw
neither fold ``k`` nor fold ``m``.  At fold ``k``'s rows it is the production ``g^{(-k)}(W)``.
So no row of fold ``k`` reaches fold ``k``'s regression through a generated design.  The pooled
construction this replaced read ``g^{(-m)}(W)`` at a training row of fold ``m``, and that model
saw fold ``k``.  The X21 run-1 diagnosis measured that leak as a bias of 0.4 SD at n = 2,000 on
a finite law (``X21-red-cell-diagnosis.md`` in the plans repository).

* The round-0 regression at fold ``k``'s rows does not move when fold ``k``'s treatment and
  outcome change, and it equals a test-side fold-wise regression.  A pooled design fails both.
* With the recorded ``epsilon`` history replayed, fold ``k``'s fitted reduced function on a
  fixed design grid does not move in any round when fold ``k``'s treatment and outcome change.
  The covariates ``W`` stay fixed, because they legitimately move ``g^{(-k)}``.
* The shipped estimate and first step equal a test-side fold-wise loop on complete data.
* With an adaptive mechanism learner, the training designs are leave-two-folds-out predictions,
  not the fold model's in-sample predictions at its own training rows.
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier

from cleverly.estimators import AdaptiveIPTW
from cleverly.estimators import iptw as iptw_module
from cleverly.fluctuation import mechanism as mechanism_module
from tests.unit import _adaptive_iptw_support as s

G = np.array([0.30, 0.55, 0.75])
Q = np.array([0.20, 0.45, 0.70])
BOUNDS = (0.01, 0.99)
FOLDS = 3


def _frame(n: int = 600, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    w = rng.integers(0, 3, size=n)
    a = rng.binomial(1, G[w])
    y = rng.binomial(1, Q[w] + 0.1 * a)
    return pd.DataFrame({"Y": y.astype(float), "A": a.astype(float), "W": w.astype(float)})


def _fit(frame: pd.DataFrame, log: s.SolveKeyLog, treatment_learner: Any = None) -> Any:
    estimator = AdaptiveIPTW(
        treatment_learner=s.CellClassifier() if treatment_learner is None else treatment_learner,
        reduced_outcome_learner=LinearRegression(),
        cross_fit=True,
        n_folds=FOLDS,
        stratify_folds="none",
        g_bounds=BOUNDS,
        tol=1e-12,
        max_iter=100,
        simultaneous=False,
        random_state=0,
        estimands=["ey1", "ey0"],
    )
    with log, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return estimator.fit(frame, outcome="Y", treatment="A", covariates=["W"]).single()


def _cell_frequency(frame: pd.DataFrame, train: np.ndarray, arm: float) -> np.ndarray:
    """``P(A = arm | W)`` at every row, from the cell frequencies of the ``train`` rows."""
    w = frame["W"].to_numpy(dtype=int)
    hit = (frame["A"].to_numpy() == arm).astype(float)
    table = np.array([hit[train & (w == level)].mean() for level in range(3)])
    return np.clip(table[w], *BOUNDS)


def _leave_two_out(frame: pd.DataFrame, assignment: np.ndarray, arm: float) -> np.ndarray:
    """``(K, n)``: entry ``k`` is ``g^{(-k)}`` on fold ``k`` and ``g^{(-k,-m)}`` on fold ``m``."""
    out = np.empty((FOLDS, len(frame)))
    for fold in range(FOLDS):
        for other in range(FOLDS):
            rows = assignment == other
            train = (assignment != fold) & (assignment != other)
            if other == fold:
                train = assignment != fold
            out[fold][rows] = _cell_frequency(frame, train, arm)[rows]
    return out


def _indicator_and_target(frame: pd.DataFrame, arm: float) -> tuple[np.ndarray, np.ndarray]:
    chosen = (frame["A"].to_numpy() == arm).astype(float)
    return chosen, chosen * frame["Y"].to_numpy()


def _fold_regression(
    slab: np.ndarray, indicator: np.ndarray, target: np.ndarray, assignment: np.ndarray, fold: int
) -> np.ndarray:
    """Fold ``fold``'s least-squares regression on ``[1, g]`` at its slab entry, all rows."""
    train = (indicator > 0) & (assignment != fold)
    x = np.column_stack([np.ones(train.sum()), slab[fold][train]])
    beta, *_ = np.linalg.lstsq(x, target[train], rcond=None)
    return np.clip(beta[0] + beta[1] * slab[fold], 0.0, 1.0)


class TestRoundZeroIgnoresTheFoldItPredicts:
    @pytest.fixture(scope="class")
    def runs(self) -> tuple[pd.DataFrame, s.SolveKeyLog, pd.DataFrame, s.SolveKeyLog]:
        frame = _frame()
        log = s.SolveKeyLog()
        _fit(frame, log)
        assert log.assignment is not None
        held = log.assignment == 0
        moved = frame.copy()
        moved.loc[held, "A"] = 1.0 - moved.loc[held, "A"]
        moved.loc[held, "Y"] = 1.0 - moved.loc[held, "Y"]
        moved_log = s.SolveKeyLog()
        _fit(moved, moved_log)
        return frame, log, moved, moved_log

    @pytest.mark.parametrize("arm", ["0", "1"])
    def test_fold_zero_regression_does_not_move_with_fold_zero(
        self, runs: tuple[pd.DataFrame, s.SolveKeyLog, pd.DataFrame, s.SolveKeyLog], arm: str
    ) -> None:
        _, log, _, moved_log = runs
        assert log.assignment is not None and moved_log.assignment is not None
        assert np.array_equal(log.assignment, moved_log.assignment)
        held = log.assignment == 0
        before = log.solved[arm].regression_initial
        after = moved_log.solved[arm].regression_initial
        assert np.max(np.abs(before[held] - after[held])) < 1e-12
        # The nonzero witness: the other folds train on fold 0, so their regressions move.
        assert np.max(np.abs(before[~held] - after[~held])) > 1e-3

    @pytest.mark.parametrize("arm", ["0", "1"])
    def test_the_round_zero_regression_equals_the_test_side_fold_wise_fit(
        self, runs: tuple[pd.DataFrame, s.SolveKeyLog, pd.DataFrame, s.SolveKeyLog], arm: str
    ) -> None:
        frame, log, _, _ = runs
        assert log.assignment is not None
        assignment = log.assignment
        slab = _leave_two_out(frame, assignment, float(arm))
        assert np.max(np.abs(log.slabs[arm] - slab)) < 1e-15
        indicator, target = _indicator_and_target(frame, float(arm))
        expected = np.empty(len(frame))
        for fold in range(FOLDS):
            rows = assignment == fold
            expected[rows] = _fold_regression(slab, indicator, target, assignment, fold)[rows]
        shipped = log.solved[arm].regression_initial
        assert np.max(np.abs(shipped - expected)) < 1e-10
        # The pooled design reads g^{(-m)} at a training row of fold m.  It moves the value.
        pooled = slab.copy()
        for fold in range(FOLDS):
            for other in range(FOLDS):
                rows = assignment == other
                pooled[fold][rows] = _cell_frequency(frame, assignment != other, float(arm))[rows]
        leak = np.empty(len(frame))
        for fold in range(FOLDS):
            rows = assignment == fold
            leak[rows] = _fold_regression(pooled, indicator, target, assignment, fold)[rows]
        assert np.max(np.abs(leak - expected)) > 1e-4


class TestTheShippedLoopIsTheFoldWiseLoop:
    @pytest.mark.parametrize("arm", ["0", "1"])
    def test_the_estimate_and_first_step_equal_the_test_side_loop(self, arm: str) -> None:
        frame = _frame()
        log = s.SolveKeyLog()
        result = _fit(frame, log)
        assert log.assignment is not None
        indicator, target = _indicator_and_target(frame, float(arm))
        slab = _leave_two_out(frame, log.assignment, float(arm))
        g0 = slab[log.assignment, np.arange(len(frame))]
        ref = s.tilt_loop_with_fold_regression(
            indicator, target, g0, log.assignment, slab=slab, bounds=BOUNDS
        )
        assert result.estimates[f"ey{arm}"].psi == pytest.approx(ref["psi"], abs=1e-10)
        shipped = result.extra["adaptive_iptw"].epsilon[arm]
        assert shipped[0] == pytest.approx(ref["eps"][0], abs=1e-10)
        assert abs(ref["eps"][0]) > 1e-4
        # The final slab entries moved along each fold's own covariate: the shipped final
        # mechanism is the reference's own-fold entry.
        assert np.max(np.abs(log.solved[arm].mechanism - ref["g"])) < 1e-9


class TestAnAdaptiveLearnerTrainsOnOutOfSampleDesigns:
    def test_the_training_designs_are_leave_two_folds_out_not_in_sample(self) -> None:
        rng = np.random.default_rng(5)
        n = 300
        w = rng.normal(size=n)
        a = rng.binomial(1, 1.0 / (1.0 + np.exp(-w)))
        y = rng.binomial(1, 0.3 + 0.3 * a)
        frame = pd.DataFrame({"Y": y.astype(float), "A": a.astype(float), "W": w})
        log = s.SolveKeyLog()
        _fit(frame, log, treatment_learner=KNeighborsClassifier(n_neighbors=3))
        assert log.assignment is not None
        assignment = log.assignment
        x = w[:, None]
        for fold in range(FOLDS):
            for other in range(FOLDS):
                if other == fold:
                    continue
                rows = assignment == other
                train = (assignment != fold) & (assignment != other)
                model = KNeighborsClassifier(n_neighbors=3).fit(x[train], a[train])
                expected = np.clip(model.predict_proba(x[rows])[:, 1], *BOUNDS)
                assert np.max(np.abs(log.slabs["1"][fold][rows] - expected)) < 1e-12
            # The fold model's in-sample predictions at its training rows are a different
            # design: each training row is one of its own three neighbours.
            training = assignment != fold
            own = KNeighborsClassifier(n_neighbors=3).fit(x[training], a[training])
            in_sample = np.clip(own.predict_proba(x[training])[:, 1], *BOUNDS)
            assert np.mean(np.abs(in_sample - log.slabs["1"][fold][training]) > 1e-9) > 0.3


#: The fixed design grid the replay test reads each fitted reduced function on.
GRID = np.linspace(0.05, 0.95, 19)[:, None]
ROUNDS = 4


class GridRecorder(LinearRegression):
    """A reduced learner that records each fitted function on :data:`GRID`, in fit order."""

    log: ClassVar[list[np.ndarray]] = []

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> GridRecorder:
        super().fit(X, y, sample_weight=sample_weight)
        GridRecorder.log.append(np.asarray(self.predict(GRID), dtype=float))
        return self


def _replayed_run(
    frame: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    replay: list[np.ndarray] | None,
    *,
    estimands: tuple[str, ...] = ("ey1",),
    fit_kwargs: dict[str, Any] | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """One ``ey1`` fit for :data:`ROUNDS` rounds.  With ``replay``, each step takes that epsilon.

    Returns the grid values of every fitted reduced function and the epsilon of every step.
    """
    solve = mechanism_module.solve_bounded_mechanism
    steps: list[np.ndarray] = []

    def replayed(indicator: Any, g: Any, h: Any, weights: Any, *, bounds: Any) -> Any:
        solved = solve(indicator, g, h, weights, bounds=bounds)
        if replay is None:
            steps.append(np.asarray(solved.epsilon, dtype=float))
            return solved
        epsilon = replay[len(steps)]
        steps.append(epsilon)
        moved = mechanism_module.apply_mechanism_tilt(g, h, epsilon, bounds=bounds)
        return dataclasses.replace(solved, epsilon=epsilon, propensity=moved)

    monkeypatch.setattr(iptw_module, "solve_bounded_mechanism", replayed)
    GridRecorder.log = []
    estimator = AdaptiveIPTW(
        treatment_learner=s.CellClassifier(),
        missingness_learner=s.CellClassifier(),
        reduced_outcome_learner=GridRecorder(),
        cross_fit=True,
        n_folds=FOLDS,
        stratify_folds="none",
        g_bounds=BOUNDS,
        tol=1e-300,
        max_iter=ROUNDS,
        simultaneous=False,
        random_state=0,
        estimands=list(estimands),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        estimator.fit(frame, outcome="Y", treatment="A", covariates=["W"], **(fit_kwargs or {}))
    monkeypatch.undo()
    return list(GridRecorder.log), steps


class TestFoldFunctionsUnderAReplayedEpsilon:
    def test_fold_zero_s_function_ignores_fold_zero_in_every_round(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        frame = _frame()
        log = s.SolveKeyLog()
        _fit(frame, log)
        assert log.assignment is not None
        held = log.assignment == 0
        before, steps = _replayed_run(frame, monkeypatch, None)
        # The fit solves both arm keys, one after the other, each for ROUNDS rounds.
        keys = len(steps) // ROUNDS
        assert keys == 2 and len(steps) == keys * ROUNDS
        assert len(before) == keys * FOLDS * (ROUNDS + 1)
        moved = frame.copy()
        moved.loc[held, "A"] = 1.0 - moved.loc[held, "A"]
        moved.loc[held, "Y"] = 1.0 - moved.loc[held, "Y"]
        after, replayed = _replayed_run(moved, monkeypatch, steps)
        assert len(after) == len(before)
        assert all(np.array_equal(a, b) for a, b in zip(steps, replayed, strict=True))
        # Each key fits fold 0, 1, 2 in each round: round 0, then one round per step.
        for index, (old, new) in enumerate(zip(before, after, strict=True)):
            if index % FOLDS == 0:
                assert np.max(np.abs(old - new)) < 1e-12, index // FOLDS
            else:
                # The nonzero witness: folds 1 and 2 train on fold 0's rows.
                assert np.max(np.abs(old - new)) > 1e-3, (index // FOLDS, index % FOLDS)
        # The steps move the functions, so the later rounds test more than round 0.
        assert np.max(np.abs(before[FOLDS * ROUNDS] - before[0])) > 1e-4


class TestNaturalCourseSlabIgnoresItsOwnFold:
    """The natural-course key's design is the response probability at the realised arm.

    Its slab is built from the leave-two-folds-out response models.  Replacing the slab with
    the production probability puts fold ``k``'s own Delta back into fold ``k``'s regression.
    """

    def test_fold_zero_s_function_ignores_fold_zero_delta_and_y_in_every_round(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        frame = _frame()
        rng = np.random.default_rng(5)
        frame["Delta"] = rng.binomial(1, np.array([0.6, 0.75, 0.9])[frame["W"].astype(int)])
        frame["Delta"] = frame["Delta"].astype(float)
        kwargs = {"delta": "Delta"}
        log = s.SolveKeyLog()
        with log, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            AdaptiveIPTW(
                treatment_learner=s.CellClassifier(),
                missingness_learner=s.CellClassifier(),
                reduced_outcome_learner=LinearRegression(),
                cross_fit=True,
                n_folds=FOLDS,
                stratify_folds="none",
                simultaneous=False,
                random_state=0,
                estimands=["ey_obs"],
            ).fit(frame, outcome="Y", treatment="A", covariates=["W"], **kwargs)
        assert log.assignment is not None
        held = log.assignment == 0
        before, steps = _replayed_run(
            frame, monkeypatch, None, estimands=("ey_obs",), fit_kwargs=kwargs
        )
        assert len(steps) == ROUNDS and len(before) == FOLDS * (ROUNDS + 1)
        moved = frame.copy()
        for column in ("Delta", "Y"):
            moved.loc[held, column] = 1.0 - moved.loc[held, column]
        after, replayed = _replayed_run(
            moved, monkeypatch, steps, estimands=("ey_obs",), fit_kwargs=kwargs
        )
        assert len(after) == len(before)
        assert all(np.array_equal(a, b) for a, b in zip(steps, replayed, strict=True))
        for index, (old, new) in enumerate(zip(before, after, strict=True)):
            if index % FOLDS == 0:
                assert np.max(np.abs(old - new)) < 1e-12, index // FOLDS
            else:
                # The nonzero witness: folds 1 and 2 train on fold 0's rows.
                assert np.max(np.abs(old - new)) > 1e-3, (index // FOLDS, index % FOLDS)
