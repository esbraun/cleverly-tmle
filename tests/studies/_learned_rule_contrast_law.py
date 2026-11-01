r"""The truth of a learned-rule fit beside a known static regime (X11 PR G, part (c)).

``TMLE(learned_rule=..., interventions=[Static(a)])`` reports the learned-rule value, the known
regime's value :math:`\Psi_{a}(P_0) = E_0\{\bar Q_0(a, W)\}` and their contrast
:math:`\tilde\psi_{0n} - \Psi_a(P_0)`.  The first term is data-adaptive: it is the fold average
of the values of the rules the fit learned, which the RM30 harness
:func:`tests.studies._learned_rule_law.harness` refits and integrates.  The second is fixed: it
is the same trapezoid rule on the RM30 grid of ``W1``, for each ``W2``, applied to the rule that
assigns ``a`` everywhere.

The module reads the RM30 laws and the harness of :mod:`tests.studies._learned_rule_law`
without editing them.  The ``learned-rule-contrast`` study is not declared yet; its design reads
these helpers.  ``tests/unit/test_learned_rule_contrast_truth.py`` checks each against a closed
form and against Monte Carlo draws.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from cleverly.estimators import TMLE
from cleverly.interventions import LearnedRule, Static
from tests.studies import _learned_rule_law as base


def estimator(learner: Any, fold_seed: int, arms: tuple[int, ...] = (1,)) -> TMLE:
    """The RM30 estimator beside ``Static(a)`` for each ``a`` of ``arms``.

    The settings are those of :func:`tests.studies._learned_rule_law.estimator`, and every
    other setting is the constructor's default, ``simultaneous=True`` among them.
    """
    return TMLE(
        learned_rule=LearnedRule(),
        interventions=[Static(arm) for arm in arms],
        n_folds=base.N_FOLDS,
        cv_evaluation=True,
        targeting_scheme="pooled",
        g_bounds="auto",
        outcome_learner=learner,
        treatment_learner=base.treatment_learner(),
        random_state=fold_seed,
    )


def fit(frame: pd.DataFrame, learner: Any, fold_seed: int, arms: tuple[int, ...] = (1,)) -> Any:
    """Fit :func:`estimator` to one sample and return the single result."""
    return (
        estimator(learner, fold_seed, arms)
        .fit(frame, outcome="Y", treatment="A", covariates=list(base.COVARIATES))
        .single()
    )


def static_value(arm: float, law: str) -> float:
    r"""``Psi_a(P0) = E Qbar0(a, W)``, the value of the known regime ``Static(arm)``.

    Parameters
    ----------
    arm : float
        The arm the regime assigns everywhere, 0 or 1.
    law : str
        A law of :data:`tests.studies._learned_rule_law.BLIPS`.

    Returns
    -------
    float
        The value, by the trapezoid rule of
        :func:`tests.studies._learned_rule_law.rule_value` on the RM30 grid.

    Raises
    ------
    ValueError
        If ``arm`` is neither 0 nor 1.
    """
    if arm not in (0.0, 1.0):
        raise ValueError(f"a static regime of this law assigns arm 0 or 1; got {arm}")
    grid = base.fixed_rule_grid(lambda w1, _w2: np.full(w1.shape, bool(arm)))
    value, _ = base.rule_value(grid, law)
    return value


@dataclass(frozen=True)
class ContrastTruth:
    """The truth of one contrast fit against one known static regime.

    Parameters
    ----------
    learned : tests.studies._learned_rule_law.Truth
        The harness truth of the learned-rule value, :math:`\\tilde\\psi_{0n}`.
    known : float
        :math:`\\Psi_a(P_0)`, the fixed value of the known regime.
    contrast : float
        The truth of the reported contrast: ``learned.truth - known``, or its negative when
        ``reference=`` names the learned rule.
    """

    learned: base.Truth
    known: float
    contrast: float


def contrast_truth(
    result: Any,
    learner: Any,
    law: str,
    arm: float,
    *,
    column: int = 1,
    reference_is_rule: bool = False,
) -> ContrastTruth:
    """Return the truth of ``ate_learned_rule`` for a fit beside ``Static(arm)``.

    Parameters
    ----------
    result : TMLEResult
        A learned-rule fit of one split with known regimes, on a draw of ``law``.
    learner : estimator
        The fit's outcome learner, which the harness refits on each fold's training rows.
    law : str
        The law the fit's data were drawn from.
    arm : float
        The arm of the known static regime, 0 or 1.
    column : int, default=1
        The regime-set column of that regime: 1 for the first regime of ``interventions=``.
    reference_is_rule : bool, default=False
        Whether ``reference=`` names the learned rule, which reports the regime minus the rule.

    Returns
    -------
    ContrastTruth
        The learned-rule truth, the known value and the contrast truth.

    Raises
    ------
    ValueError
        If the fit's column ``column`` does not assign ``arm`` on every row.
    tests.studies._learned_rule_law.HarnessMismatch
        If a refitted fold rule differs from the fit's rule on a validation row.
    """
    values = np.asarray(result.nuisance.regimes.values, dtype=float)
    if not np.all(values[:, int(arm), column] == 1.0):
        raise ValueError(f"column {column} of the fit's regime set is not Static({arm:g})")
    learned = base.harness(result, learner, law)
    known = static_value(arm, law)
    difference = learned.truth - known
    return ContrastTruth(
        learned=learned, known=known, contrast=-difference if reference_is_rule else difference
    )
