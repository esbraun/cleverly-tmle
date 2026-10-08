"""Witnesses and mutation controls for the held design.

An exact-law identity is blind to a term that vanishes at the truth.  Each test here either
uses a nonzero witness, such as a misspecified learner, or applies a deliberate mutation by
monkeypatch and checks that the identity it should break does break.  No source file is
edited.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.datasets import make_point_survival
from cleverly.longitudinal import LTMLE, DynamicRegimen, LongitudinalData
from cleverly.longitudinal import data as data_module
from cleverly.longitudinal import regimen as regimen_module
from tests.discrete_law_point_survival import (
    CellMeans,
    CellProbabilities,
    static,
    survival_law,
)


def test_the_held_factor_is_one_where_a_misspecified_learner_would_not_give_one() -> None:
    """With a penalised treatment learner the copied fit's later g_t is not one.

    The held fit's cumulative product is then exactly the node-1 factor times the censoring
    factors, and the two fits differ.
    """
    frame, _ = make_point_survival(n=1500, seed=11, layout="wide", n_times=3)
    frame["A1"] = frame["A"]
    alive = np.ones(len(frame), dtype=bool)
    for node in (2, 3):
        alive &= (frame[f"C{node - 1}"] == 1) & (frame[f"Y{node - 1}"] == 0)
        frame[f"A{node}"] = np.where(alive, frame["A"], np.nan)
    learners = {
        "outcome_learner": LinearRegression(),
        "treatment_learner": LogisticRegression(C=0.01, max_iter=1000),
        "censoring_learner": LogisticRegression(max_iter=1000),
    }
    columns = {
        "outcome": ["Y1", "Y2", "Y3"],
        "baseline": ["W1", "W2"],
        "censoring": ["C1", "C2", "C3"],
    }
    held = LTMLE({"arm1": 1, "arm0": 0}, n_folds=1, **learners).fit(frame, treatment="A", **columns)
    copied = LTMLE({"arm1": 1, "arm0": 0}, n_folds=1, **learners).fit(
        frame, treatment=["A1", "A2", "A3"], **columns
    )
    mechanism = held.mechanism
    for label in ("arm1", "arm0"):
        product = mechanism.treatment[0][label].copy()
        raw = []
        for node in range(3):
            product = product * mechanism.censoring[node][label]
            raw.append(product.copy())
        fit = next(f for f in held.fits.values() if f.regimen.label == label and f.horizon == 3)
        assert np.array_equal(fit.cumulative_unbounded, np.column_stack(raw))
        twin = next(f for f in copied.fits.values() if f.regimen.label == label and f.horizon == 3)
        later = copied.mechanism.treatment[1][label][twin.steps[1].trained_on]
        assert np.max(np.abs(later - 1.0)) > 0.05
    gaps = [abs(held[name].psi - copied[name].psi) for name in held.estimates]
    assert max(gaps) > 1e-4


def _last_column(frame: Any) -> Any:
    """Arm 1 where the history's last column is positive: ``W`` at node 1, ``L2`` at node 2."""
    return (np.asarray(frame.iloc[:, -1], dtype=float) > 0).astype(int)


def test_a_rule_is_evaluated_once_on_the_decision_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    law = survival_law(with_l2=True)
    rule = DynamicRegimen("rule", (_last_column,), rule_kind="known")

    def fit() -> Any:
        return LTMLE(
            {"rule": rule},
            n_folds=1,
            outcome_learner=CellMeans(),
            treatment_learner=CellProbabilities(),
            censoring_learner=CellMeans(),
        ).fit(law.frame(), **law.fit_columns())

    by_w = lambda probs, w: tuple(1.0 if a == (1 if w > 0 else 0) else 0.0 for a in range(3))  # noqa: E731
    once = fit()
    truth = law.functional(law.probs, by_w, 3)
    assert once["risk_regimen[rule @ t=3]"].psi == pytest.approx(truth, rel=1e-12)
    # The mutation: broadcast the rule to every node and resolve it there, so node 2
    # re-reads it on L2, as the shipped broadcast of one rule over T nodes does.
    monkeypatch.setattr(LongitudinalData, "n_decisions", property(lambda self: self.n_times))

    def broadcast(label: str, plan: Any, n_times: int) -> Any:
        if isinstance(plan, DynamicRegimen) and len(plan.plan) == 1:
            return DynamicRegimen(label, plan.plan * law.n_times, rule_kind="known")
        return plan

    monkeypatch.setattr(regimen_module, "_held_plan", broadcast)
    again = fit()
    assert abs(again["risk_regimen[rule @ t=3]"].psi - truth) > 1e-3


def test_a_swapped_cause_map_misses_the_asymmetric_law(monkeypatch: pytest.MonkeyPatch) -> None:
    law = survival_law(causes=2, censor_first=False)
    frame = law.long_frame((1.0, 2.0, 3.0))

    def fit() -> Any:
        data = LongitudinalData.from_time_to_event(
            frame,
            time="time",
            event="event",
            treatment="A",
            baseline=["W"],
            causes={1: "relapse", 2: "death"},
        )
        return LTMLE(
            {"a1": 1},
            n_folds=1,
            outcome_learner=CellMeans(),
            treatment_learner=CellProbabilities(),
            censoring_learner=CellMeans(),
        ).fit(data)

    truth = law.functional(law.probs, static(1), 3, cause=1)
    assert fit()["cif_regimen[a1, relapse @ t=3]"].psi == pytest.approx(truth, rel=1e-12)
    original = data_module._event_causes

    def swapped(codes: Any, name: str, causes: Any) -> Any:
        labels, order, competing = original(codes, name, causes)
        return labels, tuple(reversed(order)), competing

    monkeypatch.setattr(data_module, "_event_causes", swapped)
    assert abs(fit()["cif_regimen[a1, relapse @ t=3]"].psi - truth) > 0.01


class _ShrunkThirdArm(CellProbabilities):
    """Cell probabilities with the third arm's mass moved to the first."""

    def predict_proba(self, X: Any) -> np.ndarray:
        out = super().predict_proba(X).copy()
        moved = 0.1 * out[:, 2]
        out[:, 2] -= moved
        out[:, 0] += moved
        return out


def test_the_third_arm_mechanism_enters_its_curve() -> None:
    law = survival_law()
    rows = law.first_row_of()

    def curve(learner: Any) -> np.ndarray:
        result = LTMLE(
            {"a2": 2},
            n_folds=1,
            outcome_learner=CellMeans(),
            treatment_learner=learner,
            censoring_learner=CellMeans(),
        ).fit(law.frame(), **law.fit_columns())
        return np.asarray(result["risk_regimen[a2 @ t=3]"].influence_curve[rows])

    eif = law.eif(static(2), 3)
    np.testing.assert_allclose(curve(CellProbabilities()), eif, rtol=1e-10, atol=1e-12)
    assert np.max(np.abs(curve(_ShrunkThirdArm()) - eif)) > 1e-3
