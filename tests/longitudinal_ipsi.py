r"""A longhand of Kennedy's cross-fitted IPSI estimator on a two-node censored law.

:func:`longhand_ipsi` is Algorithm 1 of Kennedy (2019), with the dropout weights of Kim,
Kennedy and Naimi (2021, Appendix A, Algorithm 1), written with numpy and the scikit-learn
estimators the test hands it.  It shares no code with :mod:`cleverly.longitudinal.ipsi`.

Fold :math:`k` fits every nuisance on its training rows.  It builds the node-1 target from
its own node-2 fit, and its held-out rows take the influence function from its models.  The
designs are the engine's, column for column, so one scikit-learn estimator fitted on the
same rows returns the same predictions:

=========================  ==========================  ===========================
nuisance                   design                      rows
=========================  ==========================  ===========================
:math:`\pi_1`              ``[W]``                     every training row
:math:`\omega_1`           ``[W, A1]``                 every training row
:math:`\pi_2`              ``[W, L2, A1]``             training rows with ``C1 = 1``
:math:`\omega_2`           ``[W, L2, A1, A2]``         training rows with ``C1 = 1``
:math:`m_2`                ``[W, L2, A1, A2]``         training rows with ``C2 = 1``
:math:`m_1`                ``[W, A1]``                 training rows with ``C1 = 1``
=========================  ==========================  ===========================

``construction="stitched"`` builds the node-1 target from the out-of-fold node-2 fit at
every row instead (mutation M9).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone

COLUMNS: dict[str, Any] = {
    "outcome": "Y",
    "treatment": ["A1", "A2"],
    "baseline": ["W"],
    "time_varying": [[], ["L2"]],
    "censoring": ["C1", "C2"],
}


@dataclass(frozen=True)
class Longhand:
    """One longhand fit.

    Parameters
    ----------
    psi : float
        The pooled mean of :math:`\\varphi`.
    phi : numpy.ndarray
        Each row's :math:`\\varphi`, from the fold that holds it out.
    target : numpy.ndarray
        Each row's node-1 target :math:`\\bar m_2`, from the fold that holds it out.
    mechanism_term : numpy.ndarray
        Each row's mechanism terms, summed over the nodes.
    fold_means : dict of int to float
        The mean of :math:`\\varphi` within each fold.
    """

    psi: float
    phi: np.ndarray
    target: np.ndarray
    mechanism_term: np.ndarray
    fold_means: dict[int, float]


def _tilt(delta: float, pi: np.ndarray) -> np.ndarray:
    return delta * pi / (delta * pi + 1.0 - pi)


def longhand_ipsi(
    frame: pd.DataFrame,
    deltas: tuple[float, float],
    assignment: np.ndarray,
    *,
    treatment_learner: Any,
    censoring_learner: Any,
    outcome_learner: Any,
    g_bounds: tuple[float, float] = (0.01, 1.0),
    construction: str = "training",
) -> Longhand:
    r"""Kennedy's estimate of the odds tilt ``deltas`` on the two-node law, by hand.

    Parameters
    ----------
    frame : pandas.DataFrame
        A sample of a two-node law, with its columns.
    deltas : tuple of float
        The odds multiplier at each node.
    assignment : numpy.ndarray
        The outer fold of each row.  One fold fits in sample.
    treatment_learner, censoring_learner, outcome_learner : estimator
        The scikit-learn estimators.  Each nuisance fits a fresh clone.
    g_bounds : tuple of float
        The bound pair of the cumulative censoring probability.
    construction : {"training", "stitched"}
        ``"training"`` is the shipped training-fold recursion.  ``"stitched"`` fits the
        node-1 regression on the out-of-fold node-2 predictions (M9).

    Returns
    -------
    Longhand
        The estimate and its parts.
    """
    w = frame["W"].to_numpy(dtype=float)
    a1 = frame["A1"].to_numpy(dtype=float)
    c1 = frame["C1"].to_numpy(dtype=float)
    l2 = np.nan_to_num(frame["L2"].to_numpy(dtype=float))
    a2 = np.nan_to_num(frame["A2"].to_numpy(dtype=float))
    c2 = np.nan_to_num(frame["C2"].to_numpy(dtype=float))
    y = np.nan_to_num(frame["Y"].to_numpy(dtype=float))
    n = len(frame)
    seen1 = c1 == 1.0
    seen2 = seen1 & (c2 == 1.0)
    design = {
        "pi1": w[:, None],
        "omega1": np.column_stack([w, a1]),
        "pi2": np.column_stack([w, l2, a1]),
        "omega2": np.column_stack([w, l2, a1, a2]),
    }
    folds = np.unique(assignment)

    def probability(learner: Any, x: np.ndarray, target: np.ndarray, rows: np.ndarray) -> Any:
        model = clone(learner).fit(x[rows], target[rows])
        return model.predict_proba(x)[:, 1]

    def regression(x: np.ndarray, target: np.ndarray, rows: np.ndarray) -> Any:
        return clone(outcome_learner).fit(x[rows], target[rows])

    def node2(train: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(p2, m2 by arm, m2 at the observed arm)`` fitted on ``train``."""
        p2 = probability(treatment_learner, design["pi2"], a2, train & seen1)
        model = regression(np.column_stack([w, l2, a1, a2]), y, train & seen2)
        by_arm = np.column_stack(
            [model.predict(np.column_stack([w, l2, a1, np.full(n, arm)])) for arm in (0.0, 1.0)]
        )
        return p2, by_arm, model.predict(design["omega2"])

    def carried2(p2: np.ndarray, by_arm: np.ndarray) -> np.ndarray:
        q2 = _tilt(deltas[1], p2)
        return q2 * by_arm[:, 1] + (1.0 - q2) * by_arm[:, 0]

    phi = np.zeros(n)
    target_out = np.zeros(n)
    mechanism_out = np.zeros(n)
    fold_means: dict[int, float] = {}
    for fold in folds:
        held = assignment == fold
        train = ~held if folds.size > 1 else np.ones(n, dtype=bool)
        p1 = probability(treatment_learner, design["pi1"], a1, train)
        o1 = probability(censoring_learner, design["omega1"], c1, train)
        o2 = probability(censoring_learner, design["omega2"], c2, train & seen1)
        p2, by_arm2, observed2 = node2(train)
        target = carried2(p2, by_arm2)
        if construction == "stitched":
            for other in folds:
                rows = assignment == other
                p2_k, by_arm_k, _ = node2(~rows)
                target[rows] = carried2(p2_k, by_arm_k)[rows]
        model1 = regression(np.column_stack([w, a1]), target, train & seen1)
        by_arm1 = np.column_stack(
            [model1.predict(np.column_stack([w, np.full(n, arm)])) for arm in (0.0, 1.0)]
        )
        observed1 = model1.predict(np.column_stack([w, a1]))
        q1 = _tilt(deltas[0], p1)
        start = q1 * by_arm1[:, 1] + (1.0 - q1) * by_arm1[:, 0]
        d1, d2 = (delta * p + 1.0 - p for delta, p in zip(deltas, (p1, p2), strict=True))
        r1 = (deltas[0] * a1 + 1.0 - a1) / d1
        r2 = (deltas[1] * a2 + 1.0 - a2) / d2
        lower, upper = g_bounds
        weight1 = np.where(seen1, r1 / np.clip(o1, lower, upper), 0.0)
        weight2 = np.where(seen2, r1 * r2 / np.clip(o1 * o2, lower, upper), 0.0)
        mechanism = deltas[0] * (by_arm1[:, 1] - by_arm1[:, 0]) * (a1 - p1) / d1**2
        mechanism = mechanism + np.where(
            seen1, weight1 * deltas[1] * (by_arm2[:, 1] - by_arm2[:, 0]) * (a2 - p2) / d2**2, 0.0
        )
        value = (
            start
            + np.where(seen1, weight1 * (target - observed1), 0.0)
            + np.where(seen2, weight2 * (y - observed2), 0.0)
            + mechanism
        )
        phi[held] = value[held]
        target_out[held] = target[held]
        mechanism_out[held] = mechanism[held]
        fold_means[int(fold)] = float(np.mean(value[held]))
    return Longhand(
        psi=float(np.mean(phi)),
        phi=phi,
        target=target_out,
        mechanism_term=mechanism_out,
        fold_means=fold_means,
    )
