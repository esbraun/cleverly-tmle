r"""A longhand of the cross-fitted IPSI estimator on a two-node, three-level censored law.

:func:`longhand_ipsi_categorical` is Algorithm 1 of Kennedy (2019) with the dropout weights of
Kim, Kennedy and Naimi (2021, Appendix A, Algorithm 1) and the categorical tilt
:math:`q_t(a) = \delta_{t,a}\pi_t(a)/D_t`, written with numpy and the scikit-learn estimators
the test hands it.  It shares no code with :mod:`cleverly.longitudinal.ipsi`.

Fold :math:`k` fits every nuisance on its training rows.  It builds the node-1 target from its
own node-2 fit, and its held-out rows take the influence function from its models.  The designs
are the engine's, column for column: a treatment block is the drop-first indicator block of the
node's sorted level codes, so one scikit-learn estimator fitted on the same rows returns the
same predictions.

=========================  ===========================================  =====================
nuisance                   design                                       rows
=========================  ===========================================  =====================
:math:`\pi_1`              ``[W]``                                      every training row
:math:`\omega_1`           ``[W, ind(A1)]``                             every training row
:math:`\pi_2`              ``[W, L2, ind(A1)]``                         training rows, ``C1=1``
:math:`\omega_2`           ``[W, L2, ind(A1), ind(A2)]``                training rows, ``C1=1``
:math:`m_2`                ``[W, L2, ind(A1), ind(A2)]``                training rows, ``C2=1``
:math:`m_1`                ``[W, ind(A1)]``                             training rows, ``C1=1``
=========================  ===========================================  =====================

The factors are given on the **sorted** level axis, the axis the engine's arm codes use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone

#: The labels as the data layer sorts them: code 0, 1, 2.
SORTED_LABELS: tuple[str, ...] = ("high", "low", "standard")


def sorted_factors(factors: dict[str, float]) -> np.ndarray:
    """The factor of every level on the sorted axis; a level not named keeps ``1``."""
    return np.array([factors.get(label, 1.0) for label in SORTED_LABELS], dtype=float)


def codes_of(column: pd.Series) -> np.ndarray:
    """Sorted level codes of a label column; ``nan`` where the node was not reached."""
    lookup = {label: float(code) for code, label in enumerate(SORTED_LABELS)}
    return np.array([lookup.get(value, np.nan) for value in column], dtype=float)


def indicators(codes: np.ndarray) -> np.ndarray:
    """Drop-first indicators of the sorted codes; a missing code reads as the reference."""
    filled = np.nan_to_num(codes, nan=0.0)
    return np.column_stack([(filled == level).astype(float) for level in range(1, 3)])


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
    """

    psi: float
    phi: np.ndarray
    target: np.ndarray
    mechanism_term: np.ndarray


def _tilt(g: np.ndarray, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = g @ f
    return g * f[None, :] / d[:, None], d


def longhand_ipsi_categorical(
    frame: pd.DataFrame,
    factors: tuple[np.ndarray, np.ndarray],
    assignment: np.ndarray,
    *,
    treatment_learner: Any,
    censoring_learner: Any,
    outcome_learner: Any,
    g_bounds: tuple[float, float] = (0.01, 1.0),
) -> Longhand:
    r"""The estimate of the categorical odds tilt ``factors`` on the censored law, by hand.

    Parameters
    ----------
    frame : pandas.DataFrame
        A sample of the end-of-study law, with its columns and string arm labels.
    factors : tuple of numpy.ndarray
        The ``(3,)`` factors at each node, on the sorted level axis.
    assignment : numpy.ndarray
        The outer fold of each row.  One fold fits in sample.
    treatment_learner, censoring_learner, outcome_learner : estimator
        The scikit-learn estimators.  Each nuisance fits a fresh clone.
    g_bounds : tuple of float
        The bound pair of the cumulative censoring probability.

    Returns
    -------
    Longhand
        The estimate and its parts.
    """
    w = frame["W"].to_numpy(dtype=float)
    a1 = codes_of(frame["A1"])
    a2 = codes_of(frame["A2"])
    c1 = frame["C1"].to_numpy(dtype=float)
    l2 = np.nan_to_num(frame["L2"].to_numpy(dtype=float))
    c2 = np.nan_to_num(frame["C2"].to_numpy(dtype=float))
    y = np.nan_to_num(frame["Y"].to_numpy(dtype=float))
    n = len(frame)
    seen1 = c1 == 1.0
    seen2 = seen1 & (c2 == 1.0)
    i1, i2 = indicators(a1), indicators(a2)
    design = {
        "pi1": w[:, None],
        "omega1": np.column_stack([w, i1]),
        "pi2": np.column_stack([w, l2, i1]),
        "omega2": np.column_stack([w, l2, i1, i2]),
    }
    folds = np.unique(assignment)
    f1, f2 = factors

    def classify(learner: Any, x: np.ndarray, target: np.ndarray, rows: np.ndarray) -> Any:
        model = clone(learner).fit(x[rows], target[rows])
        return model.predict_proba(x)

    def regression(x: np.ndarray, target: np.ndarray, rows: np.ndarray) -> Any:
        return clone(outcome_learner).fit(x[rows], target[rows])

    def by_arm_of(model: Any, parts: list[np.ndarray], block: int) -> np.ndarray:
        """Predictions with the current block set to each level; ``block`` is its position."""
        out = []
        for level in range(3):
            current = np.full(n, float(level))
            columns = [*parts]
            columns.insert(block, indicators(current))
            out.append(model.predict(np.column_stack(columns)))
        return np.column_stack(out)

    def node2(train: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(pi2, m2 by arm, m2 at the observed arm)`` fitted on ``train``."""
        pi2 = classify(treatment_learner, design["pi2"], a2, train & seen1)
        model = regression(np.column_stack([w, l2, i1, i2]), y, train & seen2)
        by_arm = by_arm_of(model, [w[:, None], l2[:, None], i1], 3)
        return pi2, by_arm, model.predict(design["omega2"])

    phi = np.zeros(n)
    target_out = np.zeros(n)
    mechanism_out = np.zeros(n)
    for fold in folds:
        held = assignment == fold
        train = ~held if folds.size > 1 else np.ones(n, dtype=bool)
        pi1 = classify(treatment_learner, design["pi1"], a1, train)
        o1 = classify(censoring_learner, design["omega1"], c1, train)[:, 1]
        o2 = classify(censoring_learner, design["omega2"], c2, train & seen1)[:, 1]
        pi2, by_arm2, observed2 = node2(train)
        q2, d2 = _tilt(pi2, f2)
        target = (q2 * by_arm2).sum(axis=1)
        model1 = regression(np.column_stack([w, i1]), target, train & seen1)
        by_arm1 = np.column_stack(
            [
                model1.predict(np.column_stack([w, indicators(np.full(n, float(level)))]))
                for level in range(3)
            ]
        )
        observed1 = model1.predict(np.column_stack([w, i1]))
        q1, d1 = _tilt(pi1, f1)
        start = (q1 * by_arm1).sum(axis=1)
        code1 = np.nan_to_num(a1, nan=0.0).astype(int)
        code2 = np.nan_to_num(a2, nan=0.0).astype(int)
        r1 = f1[code1] / d1
        r2 = f2[code2] / d2
        lower, upper = g_bounds
        weight1 = np.where(seen1, r1 / np.clip(o1, lower, upper), 0.0)
        weight2 = np.where(seen2, r1 * r2 / np.clip(o1 * o2, lower, upper), 0.0)
        rows = np.arange(n)
        mechanism = f1[code1] / d1 * (by_arm1[rows, code1] - start)
        mechanism = mechanism + np.where(
            seen1, weight1 * f2[code2] / d2 * (by_arm2[rows, code2] - target), 0.0
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
    return Longhand(
        psi=float(np.mean(phi)), phi=phi, target=target_out, mechanism_term=mechanism_out
    )
