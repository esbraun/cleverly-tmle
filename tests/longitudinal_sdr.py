r"""Shared helpers for the SDR tests: per-node learners and a longhand recursion.

:class:`NodeLearner` hands each node of a two-node fit its own learner, chosen by the width
of the design that node regresses on.  It raises on a width it does not know, so a design
change fails loudly rather than handing a node the wrong learner.

:func:`longhand_sdr` is the training-fold SDR recursion of Díaz, Williams, Hoffman and
Schenck (2023, Equation (4)) on the two-node law of ``tests/discrete_law_longitudinal.py``,
written with numpy alone.  Every nuisance is a cell mean, which is closed form and so
reproduces the engine's saturated fits to the last bit or two.  The longhand fits its own
mechanism and its own regressions per fold from the raw columns.  It shares no code with
:mod:`cleverly.longitudinal.sdr`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone

from . import discrete_law_longitudinal as law

COLUMNS: dict[str, Any] = {
    "outcome": "Y",
    "treatment": ["A1", "A2"],
    "baseline": ["W"],
    "time_varying": [[], ["L2"]],
    "censoring": ["C1", "C2"],
}

#: The engine's regression design widths on the two-node law: ``[W]`` at node 1 and
#: ``[W, L2]`` at node 2.  Treatment reads ``[W]`` and ``[W, A1, L2]``; censoring reads
#: ``[W, A1]`` and ``[W, A1, L2, A2]``.
OUTCOME_WIDTH = {1: 1, 2: 2}
TREATMENT_WIDTH = {1: 1, 2: 3}
CENSORING_WIDTH = {1: 2, 2: 4}


class NodeLearner(BaseEstimator):
    """Dispatch a fit to the learner registered for the design's width.

    Parameters
    ----------
    by_width : mapping of int to estimator
        The learner for each design width.
    columns : mapping of int to tuple of int or None
        The design columns each width's learner reads.  ``None`` reads every column.
    """

    def __init__(
        self,
        by_width: Mapping[int, Any] | None = None,
        columns: Mapping[int, tuple[int, ...]] | None = None,
    ) -> None:
        self.by_width = by_width
        self.columns = columns

    def _select(self, X: Any) -> np.ndarray:
        matrix = np.asarray(X, dtype=float)
        chosen = (self.columns or {}).get(matrix.shape[1])
        return matrix if chosen is None else matrix[:, list(chosen)]

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> NodeLearner:
        width = np.asarray(X).shape[1]
        if width not in (self.by_width or {}):
            raise ValueError(f"NodeLearner has no learner for a design of width {width}")
        model = clone(self.by_width[width])  # type: ignore[index]
        if sample_weight is None:
            model.fit(self._select(X), y)
        else:
            model.fit(self._select(X), y, sample_weight=sample_weight)
        self.model_ = model
        self.classes_ = np.asarray(getattr(model, "classes_", [0.0, 1.0]), dtype=float)
        return self

    def predict(self, X: Any) -> np.ndarray:
        return np.asarray(self.model_.predict(self._select(X)), dtype=float)

    def predict_proba(self, X: Any) -> np.ndarray:
        if hasattr(self.model_, "predict_proba"):
            return np.asarray(self.model_.predict_proba(self._select(X)), dtype=float)
        p = self.predict(X)
        return np.column_stack([1.0 - p, p])


def sample(n: int, seed: int) -> pd.DataFrame:
    """``n`` iid draws from the two-node law: rows of its exact realisation, with replacement."""
    return law.frame().sample(n=n, replace=True, random_state=seed).reset_index(drop=True)


def _cell_mean(keys: np.ndarray, target: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """The mean of ``target`` over ``rows`` within each distinct row of ``keys``.

    A cell no training row reaches takes the training mean, as ``CellMeans`` does.
    """
    out = np.full(keys.shape[0], float(np.mean(target[rows])) if rows.any() else 0.0)
    if not rows.any():
        return out
    cells, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    for code in range(cells.shape[0]):
        member = inverse == code
        fitted = member & rows
        if fitted.any():
            out[member] = float(np.mean(target[fitted]))
    return out


def _bound(values: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
    return np.clip(values, bounds[0], bounds[1])


@dataclass(frozen=True)
class Longhand:
    """One longhand fit.

    Parameters
    ----------
    psi : float
        The estimate.
    curve : numpy.ndarray
        The influence curve, ``phi_1 - psi``.
    phi1 : numpy.ndarray
        Each row's first-node transform, from the fold that holds it out.
    fold_phi2 : dict of int to numpy.ndarray
        Each fold's second-node transform on every row.
    """

    psi: float
    curve: np.ndarray
    phi1: np.ndarray
    fold_phi2: dict[int, np.ndarray]


def longhand_sdr(
    frame: pd.DataFrame,
    arms: tuple[int, int],
    assignment: np.ndarray,
    *,
    g_bounds: tuple[float, float] = (0.01, 1.0),
    construction: str = "training",
    bound_rule: str = "partial",
) -> Longhand:
    r"""The SDR estimate of a static plan on the two-node law, by hand.

    Parameters
    ----------
    frame : pandas.DataFrame
        A sample of the law, with its columns.
    arms : tuple of int
        The plan's arm at each node.
    assignment : numpy.ndarray
        The outer fold of each row.  One fold fits in sample.
    g_bounds : tuple of float
        The bound pair on each partial path probability.
    construction : {"training", "stitched"}
        ``"training"`` is the shipped training-fold recursion: fold :math:`k` fits every
        nuisance on its training rows and composes every transform from its own fits.
        ``"stitched"`` is Step 2 of Section 5.3 as printed: each row's transform reads
        the fits of the fold that holds it out.
    bound_rule : {"partial", "per_node", "prefix"}
        ``"partial"`` is the shipped rule: ``g_bounds`` bounds each partial path probability
        :math:`\prod_{k=t}^{s} p_k`.  ``"per_node"`` bounds each factor before the product,
        and ``"prefix"`` divides the bounded prefix from node 1 by the bounded prefix before
        :math:`t`.  The two alternatives are the mutation M10.

    Returns
    -------
    Longhand
        The estimate, the curve and the transforms.
    """
    w = frame["W"].to_numpy(dtype=float)
    a1 = frame["A1"].to_numpy(dtype=float)
    c1 = frame["C1"].to_numpy(dtype=float)
    l2 = np.nan_to_num(frame["L2"].to_numpy(dtype=float))
    a2 = np.nan_to_num(frame["A2"].to_numpy(dtype=float), nan=-1.0)
    c2 = np.nan_to_num(frame["C2"].to_numpy(dtype=float))
    y = np.nan_to_num(frame["Y"].to_numpy(dtype=float))
    n = len(frame)
    plan1 = np.full(n, float(arms[0]))
    plan2 = np.full(n, float(arms[1]))
    follow1 = (a1 == arms[0]) & (c1 == 1.0)
    follow2 = follow1 & (a2 == arms[1]) & (c2 == 1.0)
    observed2 = c1 == 1.0
    folds = np.unique(assignment)

    def mechanism(train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(p1, p2)``: the per-node factors at the plan's arms, fitted on ``train``."""
        g1 = _cell_mean(w[:, None], (a1 == 1.0).astype(float), train)
        g1_arm = g1 if arms[0] == 1 else 1.0 - g1
        # Each censoring factor is read at the plan's arm.
        cens1_arm = _predict_cells(np.column_stack([w, a1]), c1, train, np.column_stack([w, plan1]))
        rows2 = train & observed2
        keys2 = np.column_stack([w, plan1, l2])
        g2_arm = _predict_cells(
            np.column_stack([w, a1, l2]), (a2 == 1.0).astype(float), rows2, keys2
        )
        g2_arm = g2_arm if arms[1] == 1 else 1.0 - g2_arm
        cens2_arm = _predict_cells(
            np.column_stack([w, a1, l2, a2]),
            c2,
            rows2,
            np.column_stack([w, plan1, l2, plan2]),
        )
        return g1_arm * cens1_arm, g2_arm * cens2_arm

    def regressions(
        train: np.ndarray, p1: np.ndarray, p2: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(m2, phi2, m1)`` fitted on ``train`` with the factors ``p1, p2``."""
        m2 = _cell_mean(np.column_stack([w, l2]), y, train & follow2)
        w22 = np.where(follow2, 1.0 / weight_2(p1, p2), 0.0)
        phi2 = np.where(follow1, m2 + w22 * (y - m2), 0.0)
        m1 = _cell_mean(w[:, None], phi2, train & follow1)
        return m2, phi2, m1

    def weight_2(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """The bounded denominator of :math:`W_{2,2}`."""
        if bound_rule == "partial":
            return _bound(p2, g_bounds)
        if bound_rule == "per_node":
            return _bound(p2, g_bounds)
        return _bound(p1 * p2, g_bounds) / _bound(p1, g_bounds)

    def weight_12(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """The bounded denominator of :math:`W_{1,2}`."""
        if bound_rule == "per_node":
            return _bound(p1, g_bounds) * _bound(p2, g_bounds)
        return _bound(p1 * p2, g_bounds)

    def first(m1: np.ndarray, m2: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        w11 = np.where(follow1, 1.0 / _bound(p1, g_bounds), 0.0)
        w12 = np.where(follow2, 1.0 / weight_12(p1, p2), 0.0)
        return m1 + w11 * (np.where(follow1, m2, 0.0) - m1) + w12 * (y - m2)

    phi1 = np.zeros(n)
    fold_phi2: dict[int, np.ndarray] = {}
    if construction == "training":
        for fold in folds:
            held = assignment == fold
            train = ~held if folds.size > 1 else np.ones(n, dtype=bool)
            p1, p2 = mechanism(train)
            m2, phi2, m1 = regressions(train, p1, p2)
            fold_phi2[int(fold)] = phi2
            phi1[held] = first(m1, m2, p1, p2)[held]
    elif construction == "stitched":
        p1 = np.zeros(n)
        p2 = np.zeros(n)
        m2 = np.zeros(n)
        for fold in folds:
            held = assignment == fold
            p1_k, p2_k = mechanism(~held)
            p1[held], p2[held] = p1_k[held], p2_k[held]
            m2[held] = _cell_mean(np.column_stack([w, l2]), y, ~held & follow2)[held]
        w22 = np.where(follow2, 1.0 / weight_2(p1, p2), 0.0)
        phi2 = np.where(follow1, m2 + w22 * (y - m2), 0.0)
        m1 = np.zeros(n)
        for fold in folds:
            held = assignment == fold
            m1[held] = _cell_mean(w[:, None], phi2, ~held & follow1)[held]
            fold_phi2[int(fold)] = phi2
        phi1 = first(m1, m2, p1, p2)
    else:  # pragma: no cover - a test names one of the two
        raise ValueError(construction)
    psi = float(np.mean(phi1))
    return Longhand(psi=psi, curve=phi1 - psi, phi1=phi1, fold_phi2=fold_phi2)


def _predict_cells(
    keys: np.ndarray, target: np.ndarray, rows: np.ndarray, at: np.ndarray
) -> np.ndarray:
    """Cell means of ``target`` over ``rows``, read at the key rows ``at``."""
    fallback = float(np.mean(target[rows])) if rows.any() else 0.0
    out = np.full(at.shape[0], fallback)
    if not rows.any():
        return out
    cells, inverse = np.unique(keys[rows], axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    sums = np.bincount(inverse, weights=target[rows], minlength=cells.shape[0])
    sizes = np.bincount(inverse, minlength=cells.shape[0])
    means = sums / sizes
    for code, cell in enumerate(cells):
        out[np.all(at == cell, axis=1)] = means[code]
    return out
