"""Known stochastic policies (F1) and modified treatment policies (X12) at a held node 1.

A held design resolves every regimen on its one decision.  A policy or an MTP therefore
acts at node 1 only, and every later node is an identity node with ratio one: Díaz,
Williams, Hoffman and Schenck (2023), Definition 1 with :math:`d_t(a_t, h_t) = a_t` for
``t >= 2``.  In ``lmtp`` it is K copied treatment columns with the identity policy after the
first, not ``trt`` of length one, which re-applies a policy at every node.

The checks run on the exact survival law with censoring:

* a point-mass policy equals the rule it equals, bit for bit;
* a stochastic policy's estimate is the g-formula and its curve the Gateaux derivative;
* an MTP's estimate is the g-formula of the shifted law and its curve the Gateaux
  derivative, and it equals the copied-column fit with identity policies after node 1;
* a mutation that makes the held nodes policy nodes, so that the numerator becomes
  :math:`q^K`, fails the Gateaux tests.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.interventions import ModifiedPolicy, Shift, Stochastic
from cleverly.longitudinal import LTMLE, DynamicRegimen
from cleverly.longitudinal import regimen as regimen_module
from tests.discrete_law_point_survival import (
    CellMeans,
    CellProbabilities,
    known,
    shifted,
    static,
    survival_law,
)

LAW = survival_law()
HORIZONS = (1, 2, 3)
#: ``q(a | w)``, indexed ``[w, a]``.
POLICY = np.array([[0.25, 0.5, 0.25], [0.5, 0.25, 0.25]])
#: The MTP's level map: one step up, capped at the top level.
UP = (1, 2, 2)


def _density(frame: Any) -> np.ndarray:
    w = np.asarray(frame["W"], dtype=int)
    return POLICY[w]


def _up(a: Any, h: Any) -> Any:
    return np.minimum(np.asarray(a, dtype=float) + 1.0, 2.0)


STOCHASTIC = Stochastic(_density, "q", density_kind="known")
POINT_MASS = Stochastic(
    lambda frame: np.tile([0.0, 1.0, 0.0], (len(frame), 1)), "one", density_kind="known"
)
MTP = ModifiedPolicy("up", apply=_up, policy_kind="known")
NATURAL = Shift(0.0, cap=None)


def _fit(regimens: dict[str, Any], *, held: bool = True) -> Any:
    return LTMLE(
        regimens,
        n_folds=1,
        outcome_learner=CellMeans(),
        treatment_learner=CellProbabilities(),
        censoring_learner=CellMeans(),
    ).fit(LAW.frame(), **LAW.fit_columns(held=held))


def _check_gateaux(result: Any, label: str, assign: Any) -> None:
    rows = LAW.first_row_of()
    for horizon in HORIZONS:
        estimate = result[f"risk_regimen[{label} @ t={horizon}]"]
        truth = LAW.functional(LAW.probs, assign, horizon)
        assert estimate.psi == pytest.approx(truth, rel=1e-12, abs=1e-12)
        np.testing.assert_allclose(
            estimate.influence_curve[rows], LAW.eif(assign, horizon), rtol=1e-9, atol=1e-11
        )


def test_a_point_mass_policy_is_the_rule_it_equals() -> None:
    policy = _fit({"one": POINT_MASS})
    rule = _fit({"one": 1})
    for name in rule.estimates:
        assert policy[name].psi == rule[name].psi
        assert np.array_equal(policy[name].influence_curve, rule[name].influence_curve)
    assert policy.config.policy_point_mass_nodes == (("one", (1,)),)


def test_a_stochastic_policy_at_node_one_is_the_g_formula() -> None:
    result = _fit({"q": STOCHASTIC})
    _check_gateaux(result, "q", known(POLICY))
    # It differs from every static arm, so it is a parameter no arm reaches.
    for arm in range(3):
        assert (
            abs(result["risk_regimen[q @ t=3]"].psi - LAW.functional(LAW.probs, static(arm), 3))
            > 0.005
        )


def test_an_mtp_at_node_one_is_the_g_formula_of_the_shifted_law() -> None:
    result = _fit({"up": MTP})
    _check_gateaux(result, "up", shifted(LAW, UP))
    natural = LAW.functional(LAW.probs, shifted(LAW, (0, 1, 2)), 3)
    assert abs(result["risk_regimen[up @ t=3]"].psi - natural) > 0.005


def test_an_mtp_at_node_one_equals_identity_policies_on_copied_columns() -> None:
    held = _fit({"up": MTP})
    copied = _fit({"up": DynamicRegimen("up", (MTP, NATURAL, NATURAL))}, held=False)
    for name in held.estimates:
        assert held[name].psi == pytest.approx(copied[name].psi, rel=1e-12, abs=1e-12)
        np.testing.assert_allclose(
            held[name].influence_curve, copied[name].influence_curve, rtol=1e-10, atol=1e-12
        )


def _policy_at_every_node(monkeypatch: pytest.MonkeyPatch) -> None:
    """The mutation: copy node 1's policy onto the held nodes, so the numerator is q^K."""
    original = regimen_module._resolve_plan

    def mutated(regimen: Any, data: Any, *, collapse: bool) -> Any:
        plan = original(regimen, data, collapse=collapse)
        if not plan.policy:
            return plan
        from dataclasses import replace

        policy = tuple(plan.policy[0] for _ in plan.policy)
        return replace(plan, policy=policy)

    monkeypatch.setattr(regimen_module, "_resolve_plan", mutated)


def test_a_held_node_that_is_a_policy_node_misses_the_truth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _policy_at_every_node(monkeypatch)
    result = _fit({"q": STOCHASTIC})
    rows = LAW.first_row_of()
    estimate = result["risk_regimen[q @ t=3]"]
    gap = abs(estimate.psi - LAW.functional(LAW.probs, known(POLICY), 3))
    curve = np.max(np.abs(estimate.influence_curve[rows] - LAW.eif(known(POLICY), 3)))
    assert max(gap, curve) > 1e-3


class _FirstCopy:
    """Fit on the first of each set of identical columns, as R's ``glm`` drops aliases.

    A copied-column design repeats the dose in every treatment column, so a penalized fit
    spreads its coefficient over the copies and predicts differently from the held design.
    Dropping the copies makes the two designs one regression.
    """

    def __init__(self, base: Any) -> None:
        self.base = base

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {"base": self.base}

    def set_params(self, **params: Any) -> _FirstCopy:
        self.base = params.get("base", self.base)
        return self

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> _FirstCopy:
        X = np.asarray(X, dtype=float)
        _, first = np.unique(X, axis=1, return_index=True)
        self.keep_ = np.sort(first)
        self.model_ = clone(self.base).fit(X[:, self.keep_], y, sample_weight=sample_weight)
        if hasattr(self.model_, "classes_"):
            self.classes_ = self.model_.classes_
        return self

    def predict(self, X: Any) -> Any:
        return self.model_.predict(np.asarray(X, dtype=float)[:, self.keep_])

    def predict_proba(self, X: Any) -> Any:
        return self.model_.predict_proba(np.asarray(X, dtype=float)[:, self.keep_])


def _dose_panel(n: int = 500, seed: int = 3) -> pd.DataFrame:
    """A three-node survival panel with a continuous baseline dose ``D``, copied as D1..D3."""
    rng = np.random.default_rng(seed)
    w = rng.integers(0, 2, n).astype(float)
    d = rng.uniform(0.0, 2.0, n) + 0.5 * w
    frame: dict[str, Any] = {"W": w, "D": d}
    alive = np.ones(n, dtype=bool)
    for k in (1, 2, 3):
        frame[f"D{k}"] = np.where(alive, d, np.nan)
        stay = np.where(alive, rng.binomial(1, 0.92, n).astype(float), np.nan)
        frame[f"C{k}"] = stay
        event = rng.binomial(1, 1.0 / (1.0 + np.exp(2.0 - 0.6 * d + 0.4 * w)), n)
        frame[f"Y{k}"] = np.where(alive & (stay == 1.0), event.astype(float), np.nan)
        alive = alive & (stay == 1.0) & (event == 0)
    return pd.DataFrame(frame)


def test_a_continuous_shift_at_node_one_equals_identity_shifts_on_copied_columns() -> None:
    """The held dose fits one density, at node 1, and equals the copied-column plan.

    ``Shift(0)`` at a copied node has the ratio ``g(a) / g(a)``, exactly one, so the two
    fits agree once the copied columns enter each regression once.
    """
    settings: dict[str, Any] = {
        "n_folds": 1,
        "outcome_learner": _FirstCopy(LogisticRegression(max_iter=1000)),
        "pseudo_learner": _FirstCopy(LinearRegression()),
        "treatment_learner": LogisticRegression(max_iter=1000),
        "censoring_learner": _FirstCopy(LogisticRegression(max_iter=1000)),
        "simultaneous": False,
    }
    columns = {"baseline": ["W"], "censoring": ["C1", "C2", "C3"], "outcome": ["Y1", "Y2", "Y3"]}
    frame = _dose_panel()
    up = Shift(0.5, cap=None)
    held = LTMLE({"up": up}, **settings).fit(
        frame, treatment="D", continuous_treatment=["D"], **columns
    )
    copied = LTMLE({"up": DynamicRegimen("up", (up, NATURAL, NATURAL))}, **settings).fit(
        frame, treatment=["D1", "D2", "D3"], continuous_treatment=["D1", "D2", "D3"], **columns
    )
    assert sorted(held.mechanism.densities) == [1]
    assert sorted(copied.mechanism.densities) == [1, 2, 3]
    for name in held.estimates:
        assert held[name].psi == pytest.approx(copied[name].psi, rel=1e-12, abs=1e-12)
        np.testing.assert_allclose(
            held[name].influence_curve, copied[name].influence_curve, rtol=1e-10, atol=1e-12
        )
    # The shift moves the risk, so the identity is not one between two natural courses.
    natural = LTMLE({"natural": NATURAL}, **settings).fit(
        frame, treatment="D", continuous_treatment=["D"], **columns
    )
    moved = held.psi("risk_regimen[up @ t=3]") - natural.psi("risk_regimen[natural @ t=3]")
    assert abs(moved) > 0.01
