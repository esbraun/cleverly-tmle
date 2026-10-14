r"""A modified treatment policy's density ratio, estimated by classification.

Díaz, Williams, Hoffman and Schenck (2023, Section 5.4) estimate the ratio
:math:`r(a, h) = g^d(a \mid h) / g(a \mid h)` without a density.  Stack two copies of the
sample: the observed rows :math:`(H_i, A_i)` with label :math:`\lambda = 0`, and the rows
the policy assigns, :math:`(H_i, d(A_i, H_i))`, with label :math:`\lambda = 1`.  Both
copies hold :math:`H` with its observed law, so the conditional odds of the label are the
ratio:

.. math::

    \frac{P(\lambda = 1 \mid a, h)}{P(\lambda = 0 \mid a, h)}
      = \frac{g^d(a \mid h)}{g(a \mid h)} .

Any binary classifier estimates :math:`u = P(\lambda = 1 \mid a, h)`, and
:math:`\hat r = \hat u / (1 - \hat u)`.  A randomized policy stacks one assigned copy per
randomizer value, each weighted by the value's known probability, so the assigned block
still has total weight one per unit.

**The split is over units, never over stacked rows.**  A unit's two copies share
:math:`H_i`, and a fold that trained on one copy and predicted the other would put the
unit's own row in the fit that predicts it.  So each outer fold trains on both copies of its
training units, and predicts for its held-out units, and the Super Learner's inner folds
are grouped by unit as well (Section 5.4 states the same requirement).

The ratio is not bounded.  A predicted :math:`\hat u` of exactly one would make it infinite,
so :math:`\hat u` is read below :math:`1 - 10^{-12}`, which is a guard against a division by
zero and not a truncation of any finite ratio this package reports.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .._typing import FloatArray, IntArray, Learner
from ..exceptions import DataError
from ..utils.parallel import map_parallel
from ._fitting import fit_learner, predict_mean
from .crossfit import Folds

__all__ = ["classifier_ratio"]

#: The largest classifier probability read, so ``u / (1 - u)`` stays finite.
_ODDS_GUARD = 1.0 - 1e-12


def classifier_ratio(
    learner: Learner,
    covariates: FloatArray,
    observed: FloatArray,
    assigned: Sequence[tuple[float, FloatArray]],
    weights: FloatArray,
    folds: Folds,
    *,
    evaluate_at: Sequence[FloatArray],
    fit_mask: Any = None,
    groups: IntArray | None = None,
    n_jobs: int = 1,
    atoms: Sequence[float] = (),
    assigned_by_fold: Sequence[Sequence[tuple[float, FloatArray]]] | None = None,
) -> list[FloatArray]:
    """Cross-fit :math:`\\hat r = \\hat u / (1 - \\hat u)` and evaluate it at given doses.

    Parameters
    ----------
    learner : Learner
        The binary classifier of the stacked label.
    covariates : ndarray
        ``(n, p)`` the history :math:`H`.  The treatment columns are prepended to it.
    observed : ndarray
        ``(n,)`` observed dose, or the ``(n, d)`` design of a vector treatment: the
        :math:`\\lambda = 0` block.
    assigned : sequence of tuple
        ``(probability, dose)`` per randomizer value, the :math:`\\lambda = 1` blocks, each
        dose of the shape of ``observed``.  A deterministic policy passes one pair with
        probability one.
    weights : ndarray
        ``(n,)`` observation weights.
    folds : Folds
        The outer split of the units.
    evaluate_at : sequence of ndarray
        Doses of the shape of ``observed`` at which to return the ratio, one array per
        request.
    fit_mask : ndarray or None
        ``(n,)`` the units the classifier may train on.  ``None`` is every unit.
    groups : ndarray or None
        Cluster codes.  The inner folds keep a cluster whole; without clusters they keep a
        unit's copies together.
    n_jobs : int
        Parallel workers across the outer folds.
    atoms : sequence of float
        The declared atoms of a dose of one column.  Every block of the stack then holds
        the dose and the indicator of each atom, ``[a, 1{a = a_1}, ..., 1{a = a_K}]``, so
        the odds of the label can jump at an atom.  The odds are then the ratio with
        respect to Lebesgue plus counting measure on the atoms, which Section 5.4 allows
        for any common dominating measure.
    assigned_by_fold : sequence or None
        Per outer fold, the ``assigned`` blocks that fold's classifier stacks, in place of
        ``assigned``: the map of a tuned cap fitted on that fold's training rows.  ``None``
        stacks ``assigned`` in every fold.

    Returns
    -------
    list of ndarray
        One ``(n,)`` ratio per entry of ``evaluate_at``, each row predicted by the fold that
        held the row's unit out.
    """
    h = np.asarray(covariates, dtype=float)
    if h.ndim == 1:
        h = h.reshape(-1, 1)
    n = h.shape[0]
    declared = tuple(float(atom) for atom in atoms)
    raw = _treatment_block(observed, n, None, "observed")
    width = raw.shape[1]
    a = _with_atoms(raw, declared)
    w = np.asarray(weights, dtype=float).reshape(-1)
    mask = np.ones(n, dtype=bool) if fit_mask is None else np.asarray(fit_mask, dtype=bool)
    units = np.arange(n, dtype=np.int64) if groups is None else np.asarray(groups)

    def stack(pairs: Sequence[tuple[float, FloatArray]]) -> list[tuple[float, float, FloatArray]]:
        return [(0.0, 1.0, a)] + [
            (
                1.0,
                float(p),
                _with_atoms(_treatment_block(d, n, width, "an assigned dose"), declared),
            )
            for p, d in pairs
        ]

    blocks = stack(assigned)
    fold_blocks = None if assigned_by_fold is None else [stack(pairs) for pairs in assigned_by_fold]
    requests = [
        _with_atoms(_treatment_block(dose, n, width, "evaluate_at"), declared)
        for dose in evaluate_at
    ]

    def fit_on(rows: IntArray, stacked: list[tuple[float, float, FloatArray]] = blocks) -> Learner:
        keep = rows[mask[rows]]
        design = np.vstack([np.column_stack([dose[keep], h[keep]]) for _, _, dose in stacked])
        target = np.concatenate([np.full(keep.size, label) for label, _, _ in stacked])
        weight = np.concatenate([w[keep] * share for _, share, _ in stacked])
        unit = np.concatenate([units[keep] for _ in stacked])
        return fit_learner(learner, design, target, weight, groups=unit, warn_unweighted=False)

    def ratio_at(model: Learner, rows: IntArray) -> list[FloatArray]:
        out = []
        for dose in requests:
            values = dose[rows]
            u = predict_mean(model, np.column_stack([values, h[rows]]), "classification")
            u = np.minimum(np.clip(u, 0.0, 1.0), _ODDS_GUARD)
            out.append(np.asarray(u / (1.0 - u), dtype=float))
        return out

    results = [np.zeros(n) for _ in requests]
    if folds.is_single:
        everyone = np.arange(n, dtype=np.int64)
        for target, values in zip(results, ratio_at(fit_on(everyone), everyone), strict=True):
            target[:] = values
        return results

    def run_fold(fold: int, train: IntArray, test: IntArray) -> tuple[IntArray, list[FloatArray]]:
        stacked = blocks if fold_blocks is None else fold_blocks[fold]
        return test, ratio_at(fit_on(train, stacked), test)

    for test, parts in map_parallel(
        run_fold,
        [(fold, train, test) for fold, (train, test) in enumerate(folds)],
        n_jobs=n_jobs,
    ):
        for target, column in zip(results, parts, strict=True):
            target[test] = column
    return results


def _with_atoms(block: FloatArray, atoms: tuple[float, ...]) -> FloatArray:
    """A one-column dose block with the indicator of each declared atom appended."""
    if not atoms:
        return block
    if block.shape[1] != 1:
        raise DataError("declared atoms apply to a dose of one column")
    dose = block[:, 0]
    columns = [dose]
    for atom in atoms:
        tolerance = 1e-12 * max(1.0, abs(atom))
        columns.append((np.abs(dose - atom) <= tolerance).astype(float))
    return np.column_stack(columns)


def _treatment_block(values: Any, n: int, width: int | None, what: str) -> FloatArray:
    """A dose as an ``(n, d)`` block: a vector of ``n`` doses is one column.

    Two dimensions are read as they are, so a vector treatment's design is never flattened
    into ``n * d`` rows.
    """
    block = np.asarray(values, dtype=float)
    if block.ndim == 1:
        block = block.reshape(-1, 1)
    if block.ndim != 2 or block.shape[0] != n:
        raise DataError(
            f"{what} has shape {np.shape(values)}; the classifier reads (n,) or (n, d) "
            f"treatment values with n = {n}"
        )
    if width is not None and block.shape[1] != width:
        raise DataError(
            f"{what} has {block.shape[1]} treatment column(s) and the observed treatment has "
            f"{width}; every block of the stack holds the same columns"
        )
    return block
