r"""Is each fold's weighted learned-rule curve the Gateaux derivative of the tilted fixed-rule value?

Under ``weights=`` the learned-rule target is the value in the weighted population. Conditional
on the training rows, fold :math:`v` evaluates one fixed rule :math:`d`, so the fold's
functional is :math:`P \mapsto \Psi_d(P_w)` with :math:`dP_w = w\,dP / E_P[w]`. Its derivative
is taken along a contamination of :math:`P`, the law the rows are drawn from, and it is
:math:`(w / E_P[w])\,D_d(P_w)`. The fit's fold curve is
:math:`(w / \bar w_v)\{H(Y - \bar Q^*) + \bar Q^*(d, W) - \psi_v\}`.

The fit runs on the learned-rule variant of :mod:`tests.discrete_law`, as in
``test_influence_gateaux_learned_rule.py``, with a wrong outcome oracle whose blip signs are the
rule and the treatment oracle of the tilted law. Fold :math:`v` reports the plug-in of the law
:math:`T_v` whose covariate shares are the fold's weighted shares, whose mechanism is the
oracle's and whose regression is :math:`\bar Q^*`. The law the fold's rows come from is
:math:`T_v` untilted.

* A weight on ``W`` alone: the untilted law's covariate shares are the fold's own, so
  :math:`E_{P_v}[w] = \bar w_v` and the fold curve equals the derivative at every support point.
* A weight on ``W`` and ``A``: the tilt moves the mechanism, and :math:`E_{P_v}[w]` equals the
  fold's mean weight only when the fold's treated shares equal the oracle's. The check then
  compares :math:`\bar w_v` times the fold curve with :math:`E_{P_v}[w]` times the derivative,
  which is :math:`w\,D_d(P_w)` on both sides. The mechanism of :math:`D_d(P_w)` is the tilted
  one, which differs from the untilted one at every cell.

Two controls must fail: a curve without the weight factor, and a curve normalised by the
weight mean of all rows rather than the fold's.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from cleverly.estimators import TMLE
from cleverly.interventions import LearnedRule
from tests import discrete_law as law
from tests.conftest import OracleOutcome, OracleTreatment

NAME = f"ey_learned_rule[{law.LEARNED_RULE_LABEL}]"
COUNTS = law.cell_counts(q=law.LEARNED_RULE_Q)
PROBS = COUNTS / law.N

#: The wrong regression of ``test_influence_gateaux_learned_rule.py``, with the rule's blip signs.
WRONG_Q = np.array([[0.45, 0.60], [0.55, 0.25], [0.55, 0.85]])

#: Weight functions of ``(w, a, y)``: one on ``W`` alone, one that also tilts the mechanism.
WEIGHTS: dict[str, Callable[[int, int, int], float]] = {
    "covariate": lambda w, a, y: 1.0 + 0.6 * w,
    "covariate_and_treatment": lambda w, a, y: 1.0 + 0.6 * w + 0.7 * a,
}


def _law(q: np.ndarray, g: np.ndarray) -> np.ndarray:
    arm = np.stack([1.0 - g, g], axis=1)
    outcome = np.stack([1.0 - q, q], axis=2)
    return law.P_W[:, None, None] * arm[:, :, None] * outcome


def _cells() -> np.ndarray:
    return np.repeat(np.arange(len(law.SUPPORT)), [COUNTS[w, a, y] for w, a, y in law.SUPPORT])


def _cell_weights(name: str) -> np.ndarray:
    return law.cell_weights(WEIGHTS[name])


def _tilted_g(name: str) -> np.ndarray:
    tilted = law.tilt(PROBS, _cell_weights(name))
    return tilted.sum(axis=2)[:, 1] / tilted.sum(axis=(1, 2))


def _fit(name: str) -> Any:
    g = _tilted_g(name)
    estimator = TMLE(
        learned_rule=LearnedRule(),
        outcome_learner=OracleOutcome(law.DiscreteLaw(_law(WRONG_Q, g))),
        treatment_learner=OracleTreatment(law.DiscreteLaw(_law(WRONG_Q, g))),
        cross_fit=True,
        cv_evaluation=True,
        n_folds=4,
        simultaneous=False,
        random_state=3,
    )
    frame = law.frame(COUNTS).assign(weight=_cell_weights(name)[_cells()])
    return estimator.fit(frame, outcome="Y", treatment="A", weights="weight").single()


@pytest.fixture(scope="module")
def fits() -> dict[str, Any]:
    return {name: _fit(name) for name in WEIGHTS}


def _tilted_functional(weights: np.ndarray) -> Callable[[Any], Any]:
    return lambda p: law.functional(law.tilt(p, weights), NAME)


def _untilt(tilted: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """The law whose tilt by ``weights`` is ``tilted``."""
    cells = np.zeros_like(tilted)
    for index, cell in enumerate(law.SUPPORT):
        cells[cell] = weights[index]
    raw = tilted / cells
    return raw / raw.sum()


def _folds(fit: Any, name: str) -> list[dict[str, Any]]:
    """For each fold: its rows' support points, its unscaled curve, its law and mean weight."""
    weights = _cell_weights(name)
    n = fit.data.n
    cells = _cells()
    w = np.rint(fit.data.covariates[:, 0]).astype(int)
    row_weight = weights[cells]
    targeted = fit.fluctuations["regime"].targeted
    q_star = np.array(
        [
            [float(targeted.arms[float(a)][np.flatnonzero(w == cell)[0]]) for a in (0, 1)]
            for cell in range(3)
        ]
    )
    g = _tilted_g(name)
    arm = np.stack([1.0 - g, g], axis=1)
    outcome = np.stack([1.0 - q_star, q_star], axis=2)
    curve = np.asarray(fit.estimates[NAME].influence_curve, dtype=float)
    folds = [np.asarray(test) for _, test in fit.nuisance.folds]
    out = []
    for test in folds:
        share = np.bincount(w[test], weights=row_weight[test], minlength=3)
        share = share / share.sum()
        tilted = share[:, None, None] * arm[:, :, None] * outcome
        untilted = _untilt(tilted, weights)
        out.append(
            {
                "cells": cells[test],
                "raw": curve[test] * len(folds) * test.size / n,
                "law": untilted,
                "expected_weight": float(
                    sum(untilted[cell] * weights[i] for i, cell in enumerate(law.SUPPORT))
                ),
                "fold_weight": float(np.mean(row_weight[test])),
                "all_weight": float(np.mean(row_weight)),
                "weights": weights,
            }
        )
    return out


def _derivative(fold: dict[str, Any]) -> np.ndarray:
    return law.contamination_eif(_tilted_functional(fold["weights"]), fold["law"], law.SUPPORT)


def test_the_oracle_learns_the_fixed_rule_in_every_fold(fits: dict[str, Any]) -> None:
    for fit in fits.values():
        w = np.rint(fit.data.covariates[:, 0]).astype(int)
        np.testing.assert_array_equal(fit.nuisance.regimes.values[:, :, 0], law.LEARNED_RULE[w])
        assert float(np.max(np.abs(fit.fluctuations["regime"].epsilon))) > 1e-3


def test_the_treatment_weight_moves_the_mechanism() -> None:
    """The nonzero witness for the second weight: its tilt changes ``g`` at every cell."""
    assert np.allclose(_tilted_g("covariate"), PROBS.sum(axis=2)[:, 1] / PROBS.sum(axis=(1, 2)))
    moved = _tilted_g("covariate_and_treatment") - _tilted_g("covariate")
    assert float(np.min(np.abs(moved))) > 2e-2


def test_a_covariate_weight_gives_the_derivative_at_every_point(fits: dict[str, Any]) -> None:
    for fold in _folds(fits["covariate"], "covariate"):
        assert fold["expected_weight"] == pytest.approx(fold["fold_weight"], rel=1e-12)
        derivative = _derivative(fold)
        for point in np.unique(fold["cells"]):
            np.testing.assert_allclose(
                fold["raw"][fold["cells"] == point], derivative[point], atol=1e-10, rtol=0
            )


def test_a_treatment_weight_gives_the_derivative_up_to_the_fold_mean(
    fits: dict[str, Any],
) -> None:
    gaps = []
    for fold in _folds(fits["covariate_and_treatment"], "covariate_and_treatment"):
        derivative = _derivative(fold) * fold["expected_weight"]
        gaps.append(abs(fold["expected_weight"] - fold["fold_weight"]))
        for point in np.unique(fold["cells"]):
            np.testing.assert_allclose(
                fold["raw"][fold["cells"] == point] * fold["fold_weight"],
                derivative[point],
                atol=1e-10,
                rtol=0,
            )
    # The fold constant is not the identity here, which is why the check scales both sides.
    assert max(gaps) > 1e-4


@pytest.mark.parametrize("name", list(WEIGHTS))
def test_control_a_curve_without_the_weight_is_caught(fits: dict[str, Any], name: str) -> None:
    gaps = []
    for fold in _folds(fits[name], name):
        derivative = _derivative(fold)
        weights = fold["weights"]
        for point in np.unique(fold["cells"]):
            weightless = fold["raw"][fold["cells"] == point] * fold["fold_weight"] / weights[point]
            gaps.append(float(np.max(np.abs(weightless - derivative[point]))))
    assert max(gaps) > 1e-2


def test_control_a_global_weight_mean_is_caught(fits: dict[str, Any]) -> None:
    gaps = []
    for fold in _folds(fits["covariate"], "covariate"):
        derivative = _derivative(fold)
        for point in np.unique(fold["cells"]):
            global_form = fold["raw"][fold["cells"] == point] * (
                fold["fold_weight"] / fold["all_weight"]
            )
            gaps.append(float(np.max(np.abs(global_form - derivative[point]))))
    assert max(gaps) > 1e-4
