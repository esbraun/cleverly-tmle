r"""Is each fold's known-regime and contrast curve the Gateaux derivative of its functional?

The fit of ``tests/unit/test_influence_gateaux_learned_rule.py`` beside ``Static(1)`` and
``Static(0)`` (X11 (c)). Each known regime is one more column of the regime set on the same
folds, and one pooled fluctuation with one coefficient per column gives one targeted
regression :math:`\bar Q^*`, the same in every fold. Fold :math:`v` reports the plug-in of the
law with its own covariate shares, the oracle mechanism and :math:`\bar Q^*`.

:func:`tests.discrete_law.functional` writes the value of ``Static(a)`` as the arm mean
:math:`\psi_a`, and the contrast as :math:`\Psi_d - \psi_a` at the fixed rule
:data:`tests.discrete_law.LEARNED_RULE`. The Gateaux derivative of each, by complex step at the
fold's law, must equal the fold's curve at every support point. The control reads each
contrast's curve against the derivative of the learned value alone, which drops the known
regime's term.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from cleverly.estimators import TMLE
from cleverly.interventions import LearnedRule, Static
from tests import discrete_law as law
from tests.conftest import OracleOutcome, OracleTreatment
from tests.unit.test_influence_gateaux_learned_rule import COUNTS, PROBS, WRONG_Q, _cells, _law

LEARNED = f"ey_learned_rule[{law.LEARNED_RULE_LABEL}]"
KNOWN = law.PER_ARM_NAMES["ey_known_regime"]
CONTRASTS = law.PER_ARM_NAMES["ate_learned_rule"]


@pytest.fixture(scope="module")
def fit() -> Any:
    estimator = TMLE(
        learned_rule=LearnedRule(),
        interventions=[Static(1), Static(0)],
        outcome_learner=OracleOutcome(law.DiscreteLaw(_law(WRONG_Q))),
        treatment_learner=OracleTreatment(law.DiscreteLaw(PROBS)),
        cross_fit=True,
        cv_evaluation=True,
        n_folds=4,
        simultaneous=False,
        random_state=3,
    )
    return estimator.fit(law.frame(COUNTS), outcome="Y", treatment="A").single()


def test_the_fit_reports_every_oracle_name(fit: Any) -> None:
    assert {LEARNED, *KNOWN, *CONTRASTS} == set(fit.estimates)


def test_every_column_s_update_moves(fit: Any) -> None:
    """Each column's coefficient is nonzero, so no curve is read at the initial fit."""
    assert float(np.min(np.abs(fit.fluctuations["regime"].epsilon))) > 1e-2


def test_the_oracle_branches_are_the_arm_means_and_their_difference() -> None:
    """At :data:`law.PROBS` a known value is the static arm's mean, and a contrast the
    learned value minus it, with the matching curves."""
    for arm, known, contrast in zip((1, 0), KNOWN, CONTRASTS, strict=True):
        assert law.TRUTH[known] == pytest.approx(law.TRUTH[f"ey{arm}"], abs=1e-15)
        assert law.TRUTH[contrast] == pytest.approx(
            law.TRUTH[LEARNED] - law.TRUTH[known], abs=1e-15
        )
        np.testing.assert_allclose(law.eif(known), law.eif(f"ey{arm}"), atol=1e-14, rtol=0)
        np.testing.assert_allclose(
            law.eif(contrast), law.eif(LEARNED) - law.eif(known), atol=1e-14, rtol=0
        )
    # The contrast against Static(1) and the one against Static(0) differ, so neither branch
    # can stand in for the other.
    assert abs(law.TRUTH[CONTRASTS[0]] - law.TRUTH[CONTRASTS[1]]) > 1e-2


def _fold_curves(fit: Any, name: str) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """For each fold: the support point of each validation row, the fold's unscaled curve of
    ``name`` at those rows, and the law whose plug-in the fold reports."""
    n = fit.data.n
    cells = _cells()
    w = np.rint(fit.data.covariates[:, 0]).astype(int)
    targeted = fit.fluctuations["regime"].targeted
    q_star = np.array(
        [
            [float(targeted.arms[float(a)][np.flatnonzero(w == cell)[0]]) for a in (0, 1)]
            for cell in range(3)
        ]
    )
    g = PROBS.sum(axis=2)[:, 1] / PROBS.sum(axis=(1, 2))
    arm = np.stack([1.0 - g, g], axis=1)
    outcome = np.stack([1.0 - q_star, q_star], axis=2)
    curve = np.asarray(fit.estimates[name].influence_curve, dtype=float)
    folds = [np.asarray(test) for _, test in fit.nuisance.folds]
    pieces = []
    for test in folds:
        share = np.bincount(w[test], minlength=3) / test.size
        probs = share[:, None, None] * arm[:, :, None] * outcome
        pieces.append((cells[test], curve[test] * len(folds) * test.size / n, probs))
    return pieces


@pytest.mark.parametrize("name", [*KNOWN, *CONTRASTS])
def test_each_fold_curve_is_the_gateaux_derivative_at_the_fold_law(fit: Any, name: str) -> None:
    for cells, raw, probs in _fold_curves(fit, name):
        for point in np.unique(cells):
            expected = law.gateaux(name, int(point), probs=probs)
            np.testing.assert_allclose(raw[cells == point], expected, atol=1e-10, rtol=0)


@pytest.mark.parametrize("name", CONTRASTS)
def test_control_a_contrast_without_its_known_term_is_caught(fit: Any, name: str) -> None:
    """The learned value's derivative alone misses the contrast's curve by far more than the
    tolerance above, so the known regime's term is seen."""
    gaps = [
        np.max(np.abs(raw[cells == point] - law.gateaux(LEARNED, int(point), probs=probs)))
        for cells, raw, probs in _fold_curves(fit, name)
        for point in np.unique(cells)
    ]
    assert float(np.max(gaps)) > 1e-2
