r"""AdaptiveIPTW with a missing outcome and a missing treatment, against test-side values.

The law is the exact frame ``discrete_law_composite.TWO.frame()``: 2,048 rows that are the law,
with ``A`` missing where ``DeltaA = 0`` and ``Y`` missing where ``Delta = 0``. The composite
indicator of arm ``a`` is ``DeltaA * Delta * 1{A = a}``. The tests compute the composite
mechanism and the targeting loop in numpy from the law tables and compare them with the values
the shipped loop reads and returns.

* W6: the in-sample composite key, with fixed wrong factors so that the score is not zero.
* W17: the cross-fitted composite key, with saturated factor learners refitted on each fold.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from cleverly.estimators import AdaptiveIPTW
from cleverly.estimators.composite import composite_bounds
from tests import discrete_law_composite as dl
from tests.studies.mar_arm_indexed_laws import LawResponse, LawTreatment
from tests.unit import _adaptive_iptw_support as s

LAW = dl.LAWS["two"]
G_BOUNDS = (0.01, 0.99)
NUISANCE_BOUND = 0.01
KEYS = ("0", "1")


def _fit(frame: pd.DataFrame, log: s.SolveKeyLog, **settings: Any) -> Any:
    base: dict[str, Any] = {
        "reduced_outcome_learner": LinearRegression(),
        "g_bounds": G_BOUNDS,
        "nuisance_bound": NUISANCE_BOUND,
        "tol": 1e-12,
        "max_iter": 100,
        "simultaneous": False,
        "random_state": 0,
        "estimands": ["ey1", "ey0"],
        "cross_fit": False,
    }
    base.update(settings)
    with log, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return (
            AdaptiveIPTW(**base)
            .fit(
                frame,
                outcome="Y",
                treatment="A",
                covariates=["W"],
                delta="Delta",
                treatment_delta="DeltaA",
            )
            .single()
        )


def _indicator_and_target(frame: pd.DataFrame, arm: float) -> tuple[np.ndarray, np.ndarray]:
    recorded = frame["DeltaA"].to_numpy() == 1.0
    observed = frame["Delta"].to_numpy() == 1.0
    chosen = recorded & observed & (frame["A"].to_numpy() == arm)
    outcome = np.where(observed, frame["Y"].to_numpy(), 0.0)
    return chosen.astype(float), np.where(chosen, outcome, 0.0)


# --------------------------------------------------------------------------------------- W6


class TestW6CompositeInSample:
    @pytest.fixture(scope="class")
    def run(self) -> tuple[Any, s.SolveKeyLog]:
        oracle = LAW.law()
        log = s.SolveKeyLog()
        result = _fit(
            LAW.frame(),
            log,
            treatment_learner=LawTreatment(oracle, g=LAW.wrong_g),
            missingness_learner=LawResponse(oracle, recorded=LAW.wrong_recorded),
        )
        return result, log

    def _wrong_composite(self, frame: pd.DataFrame) -> np.ndarray:
        """``P(DeltaA = 1 | w) P(A = a | DeltaA = 1, w) pi(a, w)`` from the wrong tables."""
        w = frame["W"].to_numpy(dtype=int)
        return (LAW.wrong_recorded[:, None] * LAW.wrong_g * LAW.pi)[w]

    @pytest.mark.parametrize("column", [0, 1])
    def test_the_round_zero_mechanism_is_the_composite_product(
        self, run: tuple[Any, s.SolveKeyLog], column: int
    ) -> None:
        _, log = run
        expected = self._wrong_composite(LAW.frame())[:, column]
        assert np.max(np.abs(log.mechanisms[str(column)] - expected)) < 1e-15

    def test_the_treatment_is_recorded_with_a_probability_below_one(self) -> None:
        """A recorded probability of one on every row would hide the factor (M7)."""
        assert float(np.min(LAW.wrong_recorded)) < 1.0 - 1e-3
        assert float(np.min(LAW.recorded)) < 1.0 - 1e-3

    @pytest.mark.parametrize("column", [0, 1])
    def test_the_shipped_value_equals_the_test_side_loop(
        self, run: tuple[Any, s.SolveKeyLog], column: int
    ) -> None:
        result, _ = run
        frame = LAW.frame()
        indicator, target = _indicator_and_target(frame, float(column))
        g0 = self._wrong_composite(frame)[:, column]
        ref = s.tilt_loop_with_fold_regression(indicator, target, g0, None)
        name = f"ey{column}"
        assert result.estimates[name].psi == pytest.approx(ref["psi"], abs=1e-10)
        shipped = result.extra["adaptive_iptw"].epsilon[str(column)]
        assert shipped[0] == pytest.approx(ref["eps"][0], abs=1e-10)
        # The nonzero witness: the first step is not zero, so the refit is exercised.
        assert abs(ref["eps"][0]) > 1e-3

    def test_rows_with_a_missing_outcome_and_the_arm_would_change_the_regression(self) -> None:
        frame = LAW.frame()
        for arm in (0.0, 1.0):
            rows = (frame["Delta"] == 0.0) & (frame["A"] == arm)
            assert rows.sum() > 0
        # An indicator that ignored Delta would count those rows: the value moves.
        recorded = frame["DeltaA"].to_numpy() == 1.0
        for column in (0, 1):
            indicator, target = _indicator_and_target(frame, float(column))
            wrong = (recorded & (frame["A"].to_numpy() == float(column))).astype(float)
            g0 = self._wrong_composite(frame)[:, column]
            right = s.tilt_loop_with_fold_regression(indicator, target, g0, None)
            drift = s.tilt_loop_with_fold_regression(wrong, target * wrong, g0, None)
            assert abs(right["psi"] - drift["psi"]) > 1e-3


# -------------------------------------------------------------------------------------- W17


def _factor_composite(frame: pd.DataFrame, train: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """The clipped composite of ``rows`` from saturated factors fitted on the ``train`` rows."""
    w = frame["W"].to_numpy(dtype=int)
    recorded = frame["DeltaA"].to_numpy() == 1.0
    arm = frame["A"].to_numpy()
    observed = frame["Delta"].to_numpy() == 1.0
    out = np.empty((len(frame), 2))
    for level in range(3):
        cell = train & (w == level)
        p_recorded = float(np.mean(recorded[cell]))
        for column in (0, 1):
            on_arm = cell & recorded
            p_arm = float(np.mean(arm[on_arm] == float(column)))
            responders = on_arm & (arm == float(column))
            p_response = float(np.mean(observed[responders]))
            out[rows & (w == level), column] = p_recorded * p_arm * p_response
    lower, upper = composite_bounds(G_BOUNDS, NUISANCE_BOUND, 2)
    return np.clip(out, lower, upper)


class TestW17CompositeCrossFitted:
    @pytest.fixture(scope="class")
    def run(self) -> tuple[Any, s.SolveKeyLog]:
        log = s.SolveKeyLog()
        result = _fit(
            LAW.frame(),
            log,
            treatment_learner=s.CellClassifier(),
            missingness_learner=s.CellClassifier(),
            cross_fit=True,
            n_folds=2,
        )
        return result, log

    def _oof_composite(self, log: s.SolveKeyLog) -> np.ndarray:
        frame = LAW.frame()
        assignment = log.assignment
        assert assignment is not None
        out = np.empty((len(frame), 2))
        for fold in (0, 1):
            held = assignment == fold
            out[held] = _factor_composite(frame, ~held, held)[held]
        return out

    def test_every_fold_holds_every_cell(self, run: tuple[Any, s.SolveKeyLog]) -> None:
        _, log = run
        frame = LAW.frame()
        assert log.assignment is not None
        cells = frame[["W", "A", "DeltaA", "Delta"]].fillna(-1).astype(float)
        keys = cells.apply(tuple, axis=1)
        for fold in (0, 1):
            assert set(keys[log.assignment == fold]) == set(keys)

    @pytest.mark.parametrize("column", [0, 1])
    def test_the_composite_the_loop_reads_is_the_out_of_fold_product(
        self, run: tuple[Any, s.SolveKeyLog], column: int
    ) -> None:
        _, log = run
        oof = self._oof_composite(log)[:, column]
        assert np.max(np.abs(log.mechanisms[str(column)] - oof)) < 1e-15
        # The nonzero witness: a composite from full-sample fits differs on some cell.
        full = _factor_composite(
            LAW.frame(), np.ones(len(oof), dtype=bool), np.ones(len(oof), dtype=bool)
        )[:, column]
        assert np.max(np.abs(oof - full)) > 1e-3

    @pytest.mark.parametrize("column", [0, 1])
    def test_the_shipped_value_equals_the_test_side_cross_fitted_loop(
        self, run: tuple[Any, s.SolveKeyLog], column: int
    ) -> None:
        result, log = run
        frame = LAW.frame()
        indicator, target = _indicator_and_target(frame, float(column))
        g0 = self._oof_composite(log)[:, column]
        ref = s.tilt_loop_with_fold_regression(indicator, target, g0, log.assignment)
        assert result.estimates[f"ey{column}"].psi == pytest.approx(ref["psi"], abs=1e-10)
        shipped = result.extra["adaptive_iptw"].epsilon[str(column)]
        assert shipped[0] == pytest.approx(ref["eps"][0], abs=1e-10)
        assert abs(ref["eps"][0]) > 1e-3

    def test_the_score_meets_the_tolerance_and_the_status_is_the_plugin(
        self, run: tuple[Any, s.SolveKeyLog]
    ) -> None:
        result, _ = run
        record = result.extra["adaptive_iptw"]
        assert record.stop == "tolerance"
        for key in KEYS:
            assert abs(record.score[key]) < 1e-9
        assert result.inference_status == "estimated_mechanism_plugin"
