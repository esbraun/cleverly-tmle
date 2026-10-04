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
) -> list[FloatArray]:
    """Cross-fit :math:`\\hat r = \\hat u / (1 - \\hat u)` and evaluate it at given doses.

    Parameters
    ----------
    learner : Learner
        The binary classifier of the stacked label.
    covariates : ndarray
        ``(n, p)`` the history :math:`H`.  The dose is prepended as column zero.
    observed : ndarray
        ``(n,)`` observed dose, the :math:`\\lambda = 0` block.
    assigned : sequence of tuple
        ``(probability, dose)`` per randomizer value, the :math:`\\lambda = 1` blocks.  A
        deterministic policy passes one pair with probability one.
    weights : ndarray
        ``(n,)`` observation weights.
    folds : Folds
        The outer split of the units.
    evaluate_at : sequence of ndarray
        ``(n,)`` doses at which to return the ratio, one array per request.
    fit_mask : ndarray or None
        ``(n,)`` the units the classifier may train on.  ``None`` is every unit.
    groups : ndarray or None
        Cluster codes.  The inner folds keep a cluster whole; without clusters they keep a
        unit's copies together.
    n_jobs : int
        Parallel workers across the outer folds.

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
    a = np.asarray(observed, dtype=float).reshape(-1)
    w = np.asarray(weights, dtype=float).reshape(-1)
    mask = np.ones(n, dtype=bool) if fit_mask is None else np.asarray(fit_mask, dtype=bool)
    units = np.arange(n, dtype=np.int64) if groups is None else np.asarray(groups)
    blocks = [(0.0, 1.0, a)] + [(1.0, float(p), np.asarray(d, dtype=float)) for p, d in assigned]

    def fit_on(rows: IntArray) -> Learner:
        keep = rows[mask[rows]]
        design = np.vstack([np.column_stack([dose[keep], h[keep]]) for _, _, dose in blocks])
        target = np.concatenate([np.full(keep.size, label) for label, _, _ in blocks])
        weight = np.concatenate([w[keep] * share for _, share, _ in blocks])
        unit = np.concatenate([units[keep] for _ in blocks])
        return fit_learner(learner, design, target, weight, groups=unit, warn_unweighted=False)

    def ratio_at(model: Learner, rows: IntArray) -> list[FloatArray]:
        out = []
        for dose in evaluate_at:
            values = np.asarray(dose, dtype=float).reshape(-1)[rows]
            u = predict_mean(model, np.column_stack([values, h[rows]]), "classification")
            u = np.minimum(np.clip(u, 0.0, 1.0), _ODDS_GUARD)
            out.append(np.asarray(u / (1.0 - u), dtype=float))
        return out

    results = [np.zeros(n) for _ in evaluate_at]
    if folds.is_single:
        everyone = np.arange(n, dtype=np.int64)
        for target, values in zip(results, ratio_at(fit_on(everyone), everyone), strict=True):
            target[:] = values
        return results

    def run_fold(train: IntArray, test: IntArray) -> tuple[IntArray, list[FloatArray]]:
        return test, ratio_at(fit_on(train), test)

    for test, parts in map_parallel(
        run_fold, [(train, test) for train, test in folds], n_jobs=n_jobs
    ):
        for target, column in zip(results, parts, strict=True):
            target[test] = column
    return results
