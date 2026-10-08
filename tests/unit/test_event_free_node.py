"""A grid node at which no follower of a regimen had the event has hazard zero.

On a time-to-event input with a fine grid, an arm can reach a node with no event among its
followers.  The maximum-likelihood hazard there is zero, and ``survtmle`` reads it the same
way.  The node fits no learner, its regression is exactly zero, its fluctuation is skipped,
and the nuisance report shows a ``LONGITUDINAL_CONSTANT_TARGET`` omission.  The risk of that
arm at that horizon is zero with a zero influence curve.

The mutation control removes the reading.  The logistic fluctuation then moves the zero into
its bounds, and the risk is no longer zero, so the test fails if the reading is removed.
A pooled working-model fluctuation (``msm=``) would do the same, so it refuses such a cell.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.datasets import make_longitudinal
from cleverly.estimators import _nuisance
from cleverly.exceptions import CapabilityError, DataWarning
from cleverly.longitudinal import LTMLE, LongitudinalData
from cleverly.longitudinal import sequential as sequential_module
from cleverly.longitudinal.estimator import constant_node_parameters
from cleverly.msm import MSM
from cleverly.validation.longitudinal import LONGITUDINAL_CONSTANT_TARGET


def _data(n: int = 600, seed: int = 1) -> LongitudinalData:
    """Arm 0 has no event at grid nodes 1 and 2, and events at nodes 3 and 4."""
    rng = np.random.default_rng(seed)
    w = rng.integers(0, 2, n)
    a = rng.binomial(1, 0.5, n)
    time = rng.integers(1, 5, n).astype(float)
    event = rng.binomial(1, 0.6, n)
    event[(a == 0) & (time <= 2)] = 0
    frame = pd.DataFrame({"time": time, "event": event, "A": a, "W": w})
    return LongitudinalData.from_time_to_event(
        frame, time="time", event="event", treatment="A", baseline=["W"], grid=[1, 2, 3, 4]
    )


def _estimator(**kwargs: Any) -> LTMLE:
    return LTMLE(
        {"treated": 1, "control": 0},
        outcome_learner=LogisticRegression(max_iter=1000),
        pseudo_learner=LinearRegression(),
        treatment_learner=LogisticRegression(max_iter=1000),
        censoring_learner=LogisticRegression(max_iter=1000),
        random_state=0,
        **kwargs,
    )


@pytest.mark.parametrize("n_folds", [1, 2])
def test_an_event_free_node_reads_hazard_zero(
    n_folds: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    one_valued: list[int] = []
    real = _nuisance.cross_fit_predictions

    def spy(learner: Any, design: Any, target: Any, *args: Any, **kwargs: Any) -> Any:
        mask = kwargs.get("fit_mask")
        rows = np.asarray(target)[mask] if mask is not None else np.asarray(target)
        if np.unique(rows).size == 1:
            one_valued.append(1)
        return real(learner, design, target, *args, **kwargs)

    monkeypatch.setattr(sequential_module, "cross_fit_predictions", spy)
    result = _estimator(n_folds=n_folds).fit(_data())

    assert not one_valued, "a learner was fitted on a one-valued target"
    for horizon in (1, 2):
        fit = result.fits[f"control @ t={horizon}"]
        assert fit.psi_scaled == 0.0
        np.testing.assert_array_equal(fit.influence_curve_scaled, 0.0)
    # The treated arm had events at node 1, so it is not read as zero.
    assert result.fits["treated @ t=1"].psi_scaled > 0.0
    rows = {
        (row.role, row.time, row.horizon)
        for row in result.diagnostics.nuisance_models().omissions
        if row.reason == LONGITUDINAL_CONSTANT_TARGET
    }
    assert rows == {("outcome", 1, 1), ("outcome", 2, 2), ("pseudo_outcome", 1, 2)}
    assert all(
        row.regimen == "control"
        for row in result.diagnostics.nuisance_models().omissions
        if row.reason == LONGITUDINAL_CONSTANT_TARGET
    )


def test_removing_the_reading_moves_the_risk_off_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutation control: without the reading the bounded fluctuation reports a nonzero risk."""
    monkeypatch.setattr(sequential_module, "constant_target", lambda target, fitted_on: None)
    with pytest.warns(Warning):
        result = _estimator(n_folds=1).fit(_data())
    assert result.fits["control @ t=1"].psi_scaled > 0.0


def test_a_pooled_working_model_refuses_an_event_free_cell() -> None:
    def design(label: Any, horizon: int, frame: Any) -> np.ndarray:
        n = len(frame)
        return np.column_stack(
            [np.ones(n), np.full(n, float(label == "treated")), np.full(n, float(horizon))]
        )

    msm = MSM(design=design, terms=("1", "treated", "horizon"), design_kind="known")
    with pytest.raises(CapabilityError, match="pooled working-model fluctuation"):
        _estimator(n_folds=1, msm=msm).fit(_data())


def test_a_constant_inside_the_unit_interval_is_not_read(monkeypatch: pytest.MonkeyPatch) -> None:
    """An intercept-only regression carried back holds one value in (0, 1) at a node.

    The rule reads 0 and 1 only, so the learner still fits that node, and the fit is the
    learner's fit bit for bit, as it was before the rule.  The registered studies'
    intercept-only cells depend on this: reading their constants moved them by 3e-16.
    """
    from sklearn.dummy import DummyClassifier, DummyRegressor

    one_valued: list[float] = []

    class Recording(DummyRegressor):
        def fit(self, X: Any, y: Any, sample_weight: Any = None) -> Recording:
            values = np.unique(np.asarray(y, dtype=float))
            if values.size == 1:
                one_valued.append(float(values[0]))
            return super().fit(X, y, sample_weight=sample_weight)

    frame, _ = make_longitudinal(n=400, seed=0)

    def fit() -> Any:
        return LTMLE(
            {"always": 1, "never": 0},
            outcome_learner=DummyClassifier(strategy="prior"),
            pseudo_learner=Recording(strategy="mean"),
            treatment_learner=LogisticRegression(max_iter=1000),
            censoring_learner=LogisticRegression(max_iter=1000),
            n_folds=1,
            random_state=0,
        ).fit(
            frame,
            outcome="Y",
            treatment=["A1", "A2"],
            baseline=["W1", "W2"],
            censoring=["C1", "C2"],
            time_varying=[[], ["L2"]],
        )

    reading = fit()
    assert one_valued, "no pseudo node held one value, so this witness tests nothing"
    assert all(0.0 < value < 1.0 for value in one_valued)
    monkeypatch.setattr(sequential_module, "constant_target", lambda target, fitted_on: None)
    learner = fit()
    for name in reading.estimates:
        assert reading.psi(name) == learner.psi(name)
        assert reading[name].std_error == learner[name].std_error


#: The control arm's parameters whose recursion reads a constant node, at horizons 1 and 2.
CONSTANT_NODE = (
    "risk_regimen[control @ t=1]",
    "risk_regimen[control @ t=2]",
    "ate_regimen[control vs treated @ t=1]",
    "ate_regimen[control vs treated @ t=2]",
)


def _reported(**kwargs: Any) -> Any:
    with pytest.warns(DataWarning, match="constant 0 or 1"):
        return _estimator(n_folds=1, **kwargs).fit(_data())


def test_a_constant_node_parameter_reports_no_interval() -> None:
    """The witness: a risk of zero has a zero curve, so it reports a diagnostic and no CI.

    The control arm's risk at horizons 1 and 2, and its contrasts there, take
    ``constant_node_plugin``.  Every other parameter keeps the fit's status, the band covers
    those others only, and a derived estimate that reads a flagged parameter is flagged too.
    """
    result = _reported()
    flagged = {
        name for name, estimate in result.estimates.items() if not estimate.supplies_inference
    }
    assert flagged == set(CONSTANT_NODE)
    assert constant_node_parameters(result) == tuple(
        name for name in result.estimates if name in CONSTANT_NODE
    )
    for name in CONSTANT_NODE:
        assert result[name].inference == "constant_node_plugin"
        with pytest.raises(CapabilityError, match="constant target"):
            _ = result[name].ci
    assert result.inference_status == "influence_curve"
    assert result["risk_regimen[control @ t=3]"].supplies_inference
    assert result.simultaneous is not None
    assert set(result.simultaneous.bands).isdisjoint(CONSTANT_NODE)
    assert result.rmst("control", 3, versus="treated").inference == "constant_node_plugin"
    difference = result.contrast(
        lambda values: float(values[1] - values[0]),
        ["risk_regimen[control @ t=1]", "risk_regimen[treated @ t=1]"],
        name="difference",
        gradient=lambda values: np.array([-1.0, 1.0]),
    )
    assert difference.inference == "constant_node_plugin"
    summary = result.summary()
    assert summary.count("not reported") == 2 * len(CONSTANT_NODE)
    assert "constant-node se, a diagnostic and not a standard error" in summary
    curve = result.curve()
    rows = curve[curve["inference"] == "constant_node_plugin"]
    assert set(rows["parameter"]) == set(CONSTANT_NODE)
    assert rows["std_err"].isna().all()


def test_removing_the_stamp_reports_a_zero_width_interval(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutation control: without the status the risk of zero reports the interval [0, 0]."""
    from cleverly.longitudinal import estimator as estimator_module

    monkeypatch.setattr(estimator_module, "constant_nodes", lambda fit: ())
    result = _estimator(n_folds=1).fit(_data())
    estimate = result["risk_regimen[control @ t=1]"]
    assert estimate.supplies_inference
    assert estimate.std_error == 0.0
    assert estimate.ci == (0.0, 0.0)
