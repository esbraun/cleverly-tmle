"""The longitudinal support frame is well defined for every estimator, targeted or not.

``SequentialGComputation`` and ``SequentialIPW`` fit no fluctuation, so a node of either has no
``epsilon`` and no solver verdict.  The frame leaves both columns out, as it does for ``SDR``,
and it never writes a zero or a ``True`` for a fluctuation that did not run.  The checks cover a
binary and a categorical treatment, and they pin the targeted frames so that the fix does not
move them.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import (
    CausalStudy,
    GComputationMethod,
    IPWMethod,
    LongitudinalTreatment,
    LongPanel,
    ModelSpec,
    RegimeContrast,
    Runtime,
    SDRMethod,
    TMLEMethod,
)
from cleverly.datasets import OFFER_WEEKS, make_longitudinal, make_sequential_offers
from cleverly.learners import GLM

LOGGED = {"none": "p_none", "discount": "p_discount", "shipping": "p_shipping"}
COMMON = {"regimen", "time", "n_followed", "max_weight", "effective_n", "share_truncated"}
TARGETING = {"epsilon", "converged"}
SDR_COLUMNS = {
    "target_outside",
    "target_min",
    "target_max",
    "prediction_outside",
    "bound_binding",
}
RUNTIME = Runtime(random_state=1, n_jobs=1)
LOGISTIC = LogisticRegression(max_iter=1000)


def _models() -> ModelSpec:
    return ModelSpec(
        outcome_learner=LOGISTIC,
        pseudo_learner=LinearRegression(),
        treatment_learner=LOGISTIC,
        censoring_learner=LOGISTIC,
    )


def _binary_effect() -> Any:
    frame, _ = make_longitudinal(1500, seed=3)
    design = LongitudinalTreatment(
        outcome="Y",
        treatment=["A1", "A2"],
        baseline=["W1", "W2"],
        time_varying=[[], ["L2"]],
        censoring=["C1", "C2"],
    )
    plans = {"always": (1, 1), "never": (0, 0)}
    return CausalStudy(frame, design=design).identify(RegimeContrast(plans, reference="never"))


def _categorical_effect() -> Any:
    panel, _ = make_sequential_offers(4000, seed=1)
    design = LongPanel(
        unit="customer",
        time="week",
        treatment="offer",
        baseline=["tenure"],
        time_varying=["engaged"],
        censoring="consent",
        outcome="full_price",
        treatment_probabilities=LOGGED,
    )
    plans = {name: (name,) * OFFER_WEEKS for name in ("none", "discount")}
    return CausalStudy(panel, design=design).identify(RegimeContrast(plans, reference="none"))


@pytest.fixture(scope="module", params=["binary", "categorical"])
def effect(request: pytest.FixtureRequest) -> Any:
    return _binary_effect() if request.param == "binary" else _categorical_effect()


def _assignment(frame: pd.DataFrame) -> str:
    return "assigned_shares" if "assigned_shares" in frame else "share_assigned_1"


def _baseline_methods() -> dict[str, Any]:
    glm = ModelSpec(
        outcome_learner=GLM(),
        pseudo_learner=GLM(),
        treatment_learner=GLM(),
        censoring_learner=GLM(),
    )
    ipw = ModelSpec(treatment_learner=GLM(), censoring_learner=GLM())
    return {
        "gcomp": GComputationMethod(models=glm, runtime=RUNTIME),
        "ipw": IPWMethod(models=ipw, runtime=RUNTIME),
    }


@pytest.mark.parametrize("name", ["gcomp", "ipw"])
def test_an_untargeted_baseline_has_a_frame_without_targeting_columns(
    effect: Any, name: str
) -> None:
    """The frame of a baseline fit builds, and holds no fluctuation column."""
    result = effect.estimate(method=_baseline_methods()[name])
    support = result.diagnostics.support()
    frame = support.to_frame()
    assert set(frame.columns) == COMMON | {_assignment(frame)}
    assert not TARGETING & set(frame.columns)
    assert len(frame) == len(support.rows) > 0
    assert all(row.epsilon == () and row.converged is None for row in support.rows)
    # The columns that remain are the rows' own values.
    assert frame["n_followed"].tolist() == [row.n_followed for row in support.rows]
    assert frame["max_weight"].tolist() == [row.max_weight for row in support.rows]
    assert frame["effective_n"].tolist() == [row.effective_n for row in support.rows]


def test_a_targeted_fit_keeps_its_fluctuation_columns(effect: Any) -> None:
    """``LTMLE`` reports the fitted ``epsilon`` and the solver verdict of every node."""
    result = effect.estimate(method=TMLEMethod(models=_models(), runtime=RUNTIME))
    support = result.diagnostics.support()
    frame = support.to_frame()
    assert set(frame.columns) >= TARGETING
    assert not SDR_COLUMNS & set(frame.columns)
    assert frame["epsilon"].tolist() == [row.epsilon[0] for row in support.rows]
    # A nonzero witness: a frame that dropped or zeroed the column would fail here.
    assert all(math.isfinite(value) and value != 0.0 for value in frame["epsilon"])
    assert frame["converged"].tolist() == [True] * len(frame)
    assert all(row.converged is True for row in support.rows)


def test_an_sdr_fit_keeps_its_transform_columns(effect: Any) -> None:
    """``SDR`` fits no fluctuation: five transform columns take the targeting columns' place."""
    result = effect.estimate(method=SDRMethod(models=_models(), runtime=RUNTIME))
    frame = result.diagnostics.support().to_frame()
    assert set(frame.columns) >= SDR_COLUMNS
    assert not TARGETING & set(frame.columns)


def test_a_row_without_a_fluctuation_reads_missing_and_not_zero() -> None:
    """A frame that mixes the two kinds of row never invents a value for the untargeted one.

    No fit produces a mixed frame today.  The mutation control builds one from a targeted fit,
    so a frame that wrote ``0.0`` or ``True`` for the bare row fails.
    """
    result = _binary_effect().estimate(method=TMLEMethod(models=_models(), runtime=RUNTIME))
    support = result.diagnostics.support()
    bare = dataclasses.replace(support.rows[0], epsilon=(), converged=None)
    mixed = dataclasses.replace(support, rows=(bare, *support.rows[1:]))
    frame = mixed.to_frame()
    assert math.isnan(frame["epsilon"].iloc[0])
    assert pd.isna(frame["converged"].iloc[0])
    assert frame["epsilon"].iloc[1:].tolist() == [row.epsilon[0] for row in support.rows[1:]]
    assert frame["converged"].iloc[1:].tolist() == [True] * (len(frame) - 1)
