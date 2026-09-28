r"""Is each fold's learned-rule curve the Gateaux derivative of the fixed-rule value?

The learned-rule value is data-adaptive, so no one functional of :math:`P_0` has its
influence curve.  Conditional on the training rows, fold :math:`v` evaluates one fixed rule
:math:`d`, and van der Laan and Luedtke (2015), Theorem 6, read each fold's curve as the
curve of :math:`\Psi_d`.  :func:`tests.discrete_law.functional` writes :math:`\Psi_d`
longhand at the rule :data:`tests.discrete_law.LEARNED_RULE`.

The fit here runs on a variant of the discrete law whose true blip has that rule's sign,
with oracle nuisances, so every fold learns that rule.  The folds are random, so the
pooled update moves: :math:`\varepsilon` is not zero.  Fold :math:`v` then reports the
plug-in of the law with its own covariate shares, the oracle mechanism and the targeted
regression :math:`\bar Q^*`.  The Gateaux derivative of :math:`\Psi_d` at that law, by
complex step, must equal the fold's curve at every support point.
"""

from __future__ import annotations

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

#: A wrong outcome regression with the blip signs of :data:`law.LEARNED_RULE`:
#: ``+0.15``, ``-0.3`` and ``+0.3``. Its score at the law is not zero, so the update moves.
WRONG_Q = np.array([[0.45, 0.60], [0.55, 0.25], [0.55, 0.85]])


def _law(q: np.ndarray) -> np.ndarray:
    """The cell probabilities of the variant law's ``W`` and ``A`` with the regression ``q``."""
    arm = np.stack([1.0 - law.G, law.G], axis=1)
    outcome = np.stack([1.0 - q, q], axis=2)
    return law.P_W[:, None, None] * arm[:, :, None] * outcome


@pytest.fixture(scope="module")
def fit() -> Any:
    estimator = TMLE(
        learned_rule=LearnedRule(),
        outcome_learner=OracleOutcome(law.DiscreteLaw(_law(WRONG_Q))),
        treatment_learner=OracleTreatment(law.DiscreteLaw(PROBS)),
        cross_fit=True,
        cv_evaluation=True,
        n_folds=4,
        simultaneous=False,
        random_state=3,
    )
    return estimator.fit(law.frame(COUNTS), outcome="Y", treatment="A").single()


def _cells() -> np.ndarray:
    """The support-point index of each row of the variant sample."""
    return np.repeat(np.arange(len(law.SUPPORT)), [COUNTS[w, a, y] for w, a, y in law.SUPPORT])


def test_the_oracle_learns_the_fixed_rule_in_every_fold(fit: Any) -> None:
    w = np.rint(fit.data.covariates[:, 0]).astype(int)
    np.testing.assert_array_equal(fit.nuisance.regimes.values[:, :, 0], law.LEARNED_RULE[w])


def test_the_pooled_update_moves(fit: Any) -> None:
    assert float(np.max(np.abs(fit.fluctuations["regime"].epsilon))) > 1e-2


def test_the_fixed_rule_value_is_a_regime_value() -> None:
    """At :data:`law.PROBS` the oracle branch equals the regime mean of the same rule."""
    assert law.TRUTH[NAME] == pytest.approx(law.TRUTH["ey_regime[rule]"], abs=1e-15)
    np.testing.assert_allclose(law.eif(NAME), law.eif("ey_regime[rule]"), atol=1e-14, rtol=0)


def test_each_fold_curve_is_the_gateaux_derivative_at_the_fold_law(fit: Any) -> None:
    n = fit.data.n
    cells = _cells()
    w = np.rint(fit.data.covariates[:, 0]).astype(int)
    targeted = fit.fluctuations["regime"].targeted
    # Qbar*(a, w), which one pooled epsilon makes the same in every fold.
    q_star = np.array(
        [
            [float(targeted.arms[float(a)][np.flatnonzero(w == cell)[0]]) for a in (0, 1)]
            for cell in range(3)
        ]
    )
    g = PROBS.sum(axis=2)[:, 1] / PROBS.sum(axis=(1, 2))
    curve = np.asarray(fit.estimates[NAME].influence_curve, dtype=float)
    folds = [np.asarray(test) for _, test in fit.nuisance.folds]
    for test in folds:
        share = np.bincount(w[test], minlength=3) / test.size
        arm = np.stack([1.0 - g, g], axis=1)
        outcome = np.stack([1.0 - q_star, q_star], axis=2)
        probs = share[:, None, None] * arm[:, :, None] * outcome
        raw = curve[test] * len(folds) * test.size / n
        for point in np.unique(cells[test]):
            expected = law.gateaux(NAME, int(point), probs=probs)
            np.testing.assert_allclose(raw[cells[test] == point], expected, atol=1e-10, rtol=0)


def test_control_a_wrong_rule_is_caught(fit: Any) -> None:
    """The regime ``never`` differs from the fixed rule at two cells, and its derivative
    differs from the fold curve by far more than the tolerance above."""
    assert float(np.max(np.abs(law.eif("ey_regime[never]") - law.eif(NAME)))) > 1e-2
