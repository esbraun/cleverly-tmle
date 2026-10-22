"""The longitudinal negative controls (X82 part C2).

Three forms read a control that the treatment cannot move. The negative-control outcome
refits the panel with the control in place of the outcome. Its pre-treatment form refuses a
control that the adjustment set already explains. The future-treatment test regresses a
control on a later treatment, adjusted for that treatment's declared history ``H_{t+1}``.

The witnesses are nonzero by construction. An unmeasured cause of the later treatment and
of the control moves the future-treatment estimate, and the same control without that cause
does not. A control driven by ``L_2``, which drives ``A_2``, reads 0 only because the
history holds ``L_2``. The refusals are each read on the capability row and on the call.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.data import CausalData
from cleverly.estimators import TMLE
from cleverly.exceptions import CapabilityError, RefusalReason
from cleverly.longitudinal import LTMLE, Repeated
from cleverly.validation import refute
from cleverly.validation.refute import _future_history, refute_refusal

CONTRAST = "ate_regimen[never vs always]"


def _expit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _panel(n: int = 600, seed: int = 0, *, hidden: float = 0.0) -> pd.DataFrame:
    """A two-node panel. ``U`` is unmeasured and moves ``A2`` with weight ``hidden``."""
    rng = np.random.default_rng(seed)
    w = rng.normal(size=n)
    u = rng.normal(size=n)
    a1 = rng.binomial(1, _expit(0.5 * w)).astype(float)
    l2 = rng.normal(size=n) + 0.5 * w + 0.5 * a1
    a2 = rng.binomial(1, _expit(0.8 * l2 + hidden * u)).astype(float)
    y1 = rng.binomial(1, _expit(-0.5 + 0.4 * a1 + 0.3 * w)).astype(float)
    y2 = rng.binomial(1, _expit(-0.5 + 0.4 * a2 + 0.3 * l2 + 0.3 * y1)).astype(float)
    return pd.DataFrame({"W": w, "U": u, "A1": a1, "Y1": y1, "L2": l2, "A2": a2, "Y2": y2})


def _ltmle(n_folds: int = 1) -> LTMLE:
    return LTMLE(
        {"always": 1, "never": 0},
        outcome_learner=LogisticRegression(max_iter=1000),
        pseudo_learner=LinearRegression(),
        treatment_learner=LogisticRegression(max_iter=1000),
        n_folds=n_folds,
        simultaneous=False,
        random_state=3,
    )


def _fit(frame: pd.DataFrame, *, repeated: bool = False, n_folds: int = 1) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _ltmle(n_folds).fit(
            frame,
            outcome=Repeated(["Y1", "Y2"]) if repeated else "Y2",
            treatment=["A1", "A2"],
            baseline=["W"],
            time_varying=[[], ["L2"]] if repeated else [[], ["Y1", "L2"]],
        )


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return _panel()


@pytest.fixture(scope="module")
def fitted(frame: pd.DataFrame) -> Any:
    return _fit(frame)


def _run(result: Any, estimand: str = CONTRAST, **request: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return refute(result, estimand=estimand, random_state=1, **request)


def _refused(result: Any, estimand: str = CONTRAST, **request: Any) -> CapabilityError:
    """The row and the call refuse ``request`` with one message."""
    refusal = refute_refusal(result, estimand=estimand, **request)
    assert refusal is not None
    with pytest.raises(CapabilityError) as error:
        _run(result, estimand, **request)
    assert error.value.refusal.message() == refusal.message()
    return error.value


# ----------------------------------------------------------- the negative-control outcome


class TestNegativeControlOutcome:
    def test_an_end_of_study_control_refits_in_place_of_the_outcome(
        self, fitted: Any, frame: pd.DataFrame
    ) -> None:
        rng = np.random.default_rng(5)
        control = rng.binomial(1, 0.4, size=len(frame)).astype(float)
        report = _run(fitted, tests=("negative_control_outcome",), negative_control_outcome=control)
        test = report["negative_control_outcome"]
        assert test.passed is None
        assert test.rung == "L0"
        assert abs(test.statistic) < 3.0

    def test_a_control_the_treatment_moves_reads_nonzero(
        self, fitted: Any, frame: pd.DataFrame
    ) -> None:
        # A deliberate bad control: the second treatment causes it, so the refit finds the
        # effect and the reading is far from 0.
        rng = np.random.default_rng(6)
        moved = rng.binomial(1, _expit(-1.0 + 2.5 * frame["A2"].to_numpy())).astype(float)
        report = _run(fitted, tests=("negative_control_outcome",), negative_control_outcome=moved)
        assert abs(report["negative_control_outcome"].statistic) > 4.0

    def test_a_repeated_control_carries_one_value_per_node(self, frame: pd.DataFrame) -> None:
        result = _fit(frame, repeated=True)
        rng = np.random.default_rng(7)
        control = rng.binomial(1, 0.4, size=(len(frame), 2)).astype(float)
        estimand = "ate_regimen[never vs always @ t=2]"
        report = _run(
            result,
            estimand,
            tests=("negative_control_outcome",),
            negative_control_outcome=control,
        )
        assert np.isfinite(report["negative_control_outcome"].values[0])
        replaced = result.data.with_outcome(control, name="control")
        np.testing.assert_array_equal(replaced.repeated, control)
        # The copies of the primary outcome in the history do not move.
        for node in (1, 2):
            np.testing.assert_array_equal(
                replaced.covariate_history(node), result.data.covariate_history(node)
            )

    def test_a_continuous_control_on_a_cross_fitted_panel_takes_its_observed_range(
        self, frame: pd.DataFrame
    ) -> None:
        # The fit's outcome learner is a classifier and the panel is cross-fitted. The
        # control refit regresses with the pseudo-outcome learner and bounds the control to
        # its observed range, and the report prints those bounds.
        result = _fit(frame, n_folds=2)
        control = np.random.default_rng(14).normal(size=len(frame))
        report = _run(result, tests=("negative_control_outcome",), negative_control_outcome=control)
        detail = report["negative_control_outcome"].detail
        assert f"[{control.min():.5g}, {control.max():.5g}]" in detail
        assert result.estimator.q_bounds is None

    def test_a_binary_control_takes_no_bounds(self, frame: pd.DataFrame) -> None:
        result = _fit(frame, n_folds=2)
        control = np.random.default_rng(15).binomial(1, 0.4, size=len(frame)).astype(float)
        report = _run(result, tests=("negative_control_outcome",), negative_control_outcome=control)
        assert "observed range" not in report["negative_control_outcome"].detail

    def test_a_survival_fit_refuses_the_control(self) -> None:
        from tests.unit.test_longitudinal_refutation import _ltmle as _survival_ltmle
        from tests.unit.test_longitudinal_refutation import _survival_panel

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = _survival_ltmle(n_folds=1).fit(
                _survival_panel(),
                outcome=["Y1", "Y2"],
                treatment=["A1", "A2"],
                baseline=["W"],
                time_varying=[[], ["L2"]],
                censoring=["C1", "C2"],
            )
        error = _refused(
            result,
            estimand=next(key for key in result.estimates if key.startswith("ate")),
            tests=("negative_control_outcome",),
            negative_control_outcome=np.zeros(result.data.n),
        )
        assert error.refusal.reason is RefusalReason.IMPLEMENTATION_GAP
        assert error.refusal.roadmap == "X82"


class TestPreTreatmentControl:
    def test_a_confounded_control_with_its_own_variation_runs(
        self, fitted: Any, frame: pd.DataFrame
    ) -> None:
        # The witness that the floor is not a blanket refusal: the control depends on W and
        # keeps its own noise, so the residual share is near 1/2 and the test runs.
        rng = np.random.default_rng(8)
        control = frame["W"].to_numpy() + rng.normal(size=len(frame))
        control = (control > 0).astype(float)
        report = _run(
            fitted,
            tests=("negative_control_outcome",),
            negative_control_outcome=control,
            negative_control_kind="pre_treatment",
        )
        assert "pre-treatment" in report["negative_control_outcome"].scope_sentence

    def test_an_adjustment_column_is_refused(self, fitted: Any, frame: pd.DataFrame) -> None:
        error = _refused(
            fitted,
            tests=("negative_control_outcome",),
            negative_control_outcome=frame["W"].to_numpy(),
            negative_control_kind="pre_treatment",
        )
        assert error.refusal.reason is RefusalReason.INVALID_DECLARATION
        assert "'W'" in error.refusal.message()

    def test_a_control_below_the_residual_floor_is_refused(
        self, fitted: Any, frame: pd.DataFrame
    ) -> None:
        rng = np.random.default_rng(9)
        almost = 2.0 * frame["W"].to_numpy() + 1e-3 * rng.normal(size=len(frame))
        error = _refused(
            fitted,
            tests=("negative_control_outcome",),
            negative_control_outcome=almost,
            negative_control_kind="pre_treatment",
        )
        assert "below the floor 0.01" in error.refusal.message()
        # The floor is the argument: a floor of 0 admits the same control.
        assert (
            refute_refusal(
                fitted,
                estimand=CONTRAST,
                tests=("negative_control_outcome",),
                negative_control_outcome=almost,
                negative_control_kind="pre_treatment",
                negative_control_floor=0.0,
            )
            is None
        )

    def test_the_point_fit_reads_the_same_floor(self) -> None:

        rng = np.random.default_rng(10)
        w = rng.normal(size=(400, 2))
        a = rng.binomial(1, _expit(w[:, 0])).astype(float)
        y = w[:, 0] + a + rng.normal(size=400)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = (
                TMLE(
                    outcome_learner=LinearRegression(),
                    treatment_learner=LogisticRegression(),
                    cross_fit=False,
                )
                .fit(CausalData.from_arrays(y, a, w))
                .single()
            )
        almost = w[:, 1] + 1e-3 * rng.normal(size=400)
        refusal = refute_refusal(
            result,
            estimand="ate",
            tests=("negative_control_outcome",),
            negative_control_outcome=almost,
            negative_control_kind="pre_treatment",
        )
        assert refusal is not None and "below the floor" in refusal.message()
        assert (
            refute_refusal(
                result,
                estimand="ate",
                tests=("negative_control_outcome",),
                negative_control_outcome=almost,
            )
            is None
        )

    def test_a_repeated_outcome_refuses_the_pre_treatment_form(self, frame: pd.DataFrame) -> None:
        result = _fit(frame, repeated=True)
        error = _refused(
            result,
            estimand="ate_regimen[never vs always @ t=2]",
            tests=("negative_control_outcome",),
            negative_control_outcome=np.zeros((len(frame), 2)),
            negative_control_kind="pre_treatment",
        )
        assert "one value per unit" in error.refusal.message()


# -------------------------------------------------------------- the future-treatment test


class TestFutureTreatment:
    def test_an_unmeasured_common_cause_moves_the_estimate(self) -> None:
        # U drives A2 and the control. The history does not hold U, so the estimate is
        # nonzero. The same control on a panel where U leaves A2 alone reads 0.
        confounded = _panel(n=1500, seed=11, hidden=2.0)
        clean = _panel(n=1500, seed=11, hidden=0.0)
        readings = {}
        for name, panel in (("confounded", confounded), ("clean", clean)):
            control = panel["U"].to_numpy()
            report = _run(
                _fit(panel), tests=("future_treatment",), future_treatment_control=control
            )
            readings[name] = report["future_treatment"]
        assert abs(readings["confounded"].statistic) > 6.0
        assert abs(readings["clean"].statistic) < 3.0
        assert readings["confounded"].passed is None
        assert readings["confounded"].rung == "L0"

    def test_the_history_holds_the_next_covariate(self, frame: pd.DataFrame) -> None:
        # L2 drives A2 and the control. Adjusted for H_2 = (W, A1, Y1, L2) the reading is 0.
        # The mutation drops L2 from the history and the same control reads far from 0.
        rng = np.random.default_rng(12)
        control = frame["L2"].to_numpy() + rng.normal(size=len(frame))
        fitted = _fit(frame)
        report = _run(fitted, tests=("future_treatment",), future_treatment_control=control)
        assert abs(report["future_treatment"].statistic) < 3.0
        history, names = _future_history(fitted.data, 2)
        assert "L2" in names
        assert history.shape[1] == len(names)

        keep = [index for index, name in enumerate(names) if name != "L2"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dropped = (
                TMLE(
                    outcome_learner=LinearRegression(),
                    treatment_learner=LogisticRegression(),
                    cross_fit=False,
                )
                .fit(CausalData.from_arrays(control, frame["A2"].to_numpy(), history[:, keep]))
                .single()["ate"]
            )
        assert abs(dropped.psi / dropped.plugin_std_error) > 6.0

    def test_a_continuous_control_keeps_the_folds(self, frame: pd.DataFrame) -> None:
        result = _fit(frame, n_folds=2)
        control = np.random.default_rng(16).normal(size=len(frame))
        report = _run(result, tests=("future_treatment",), future_treatment_control=control)
        detail = report["future_treatment"].detail
        assert "cross-fitted in 2 folds" in detail
        rows = result.data.node_present(2)
        assert f"[{control[rows].min():.5g}, {control[rows].max():.5g}]" in detail

    def test_a_history_column_is_refused(self, fitted: Any, frame: pd.DataFrame) -> None:
        error = _refused(
            fitted,
            tests=("future_treatment",),
            future_treatment_control=frame["L2"].to_numpy(),
        )
        assert error.refusal.reason is RefusalReason.INVALID_DECLARATION
        assert "'L2'" in error.refusal.message()

    def test_a_constant_control_is_refused(self, fitted: Any, frame: pd.DataFrame) -> None:
        error = _refused(
            fitted, tests=("future_treatment",), future_treatment_control=np.ones(len(frame))
        )
        assert "0 by construction" in error.refusal.message()

    def test_the_last_node_has_no_later_treatment(self, fitted: Any, frame: pd.DataFrame) -> None:
        error = _refused(
            fitted,
            tests=("future_treatment",),
            future_treatment_control=np.arange(len(frame), dtype=float),
            future_treatment_node=2,
        )
        assert "no later treatment node" in error.refusal.message()

    def test_a_held_design_has_no_later_treatment(self, frame: pd.DataFrame) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            held = _ltmle().fit(
                frame, outcome="Y2", treatment="A1", baseline=["W"], time_varying=[[], ["L2"]]
            )
        error = _refused(
            held,
            estimand=next(key for key in held.estimates if key.startswith("ate")),
            tests=("future_treatment",),
            future_treatment_control=np.arange(len(frame), dtype=float),
        )
        assert "one treatment decision" in error.refusal.message()

    def test_a_point_fit_has_no_later_node(self) -> None:

        rng = np.random.default_rng(13)
        w = rng.normal(size=(200, 1))
        a = rng.binomial(1, 0.5, size=200).astype(float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = (
                TMLE(
                    outcome_learner=LinearRegression(),
                    treatment_learner=LogisticRegression(),
                    cross_fit=False,
                )
                .fit(CausalData.from_arrays(w[:, 0] + a, a, w))
                .single()
            )
        refusal = refute_refusal(
            result,
            estimand="ate",
            tests=("future_treatment",),
            future_treatment_control=rng.normal(size=200),
        )
        assert refusal is not None and "point-treatment fit" in refusal.message()

    def test_the_test_needs_a_control(self, fitted: Any) -> None:
        with pytest.raises(ValueError, match="future_treatment_control"):
            _run(fitted, tests=("future_treatment",))
