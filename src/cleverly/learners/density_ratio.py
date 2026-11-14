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

**Step columns.**  A capped shift or a banded policy makes the true ratio jump in the dose: at
the cap, at a band edge, at a declared support bound.  A smooth classifier would round such a
jump off, so on a dose of one column the policy hands the classifier one indicator per place
where its ratio can jump, and every stacked block reads ``[a, atom indicators, steps, H]``.  A
classifier that sets ``policy_steps = "omit"`` reads the design without them.

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

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from .._typing import FloatArray, IntArray, Learner
from ..data.validate import atom_block
from ..exceptions import DataError
from ..utils.parallel import map_parallel
from ._fitting import fit_learner, predict_mean
from .crossfit import Folds

__all__ = ["classifier_ratio", "omits_policy_steps"]

#: The largest classifier probability read, so ``u / (1 - u)`` stays finite.
_ODDS_GUARD = 1.0 - 1e-12

#: A classifier that sets ``policy_steps = "omit"`` receives no step columns.
OMIT_STEPS = "omit"

#: ``steps(dose, rows, fold, train)``: the ``(rows.size, S)`` step columns at the doses
#: ``dose`` of the rows ``rows``, for outer fold ``fold``, with bounds read from ``train``.
PolicySteps = Callable[[FloatArray, IntArray, int, IntArray], FloatArray]


def omits_policy_steps(learner: Any) -> bool:
    """Whether a classifier opts out of the policy's step columns (``policy_steps = "omit"``).

    Parameters
    ----------
    learner : object
        The classifier.

    Returns
    -------
    bool
        ``True`` when the learner reads the design without step columns.
    """
    return getattr(learner, "policy_steps", None) == OMIT_STEPS


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
    steps: PolicySteps | None = None,
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
    steps : callable or None
        The policy's step columns, ``steps(dose, rows, fold, train)``: the indicators at
        which the true ratio can jump in the dose, as ``(rows.size, S)`` columns at the
        doses ``dose`` of the rows ``rows``, for outer fold ``fold`` (0 for a single fold),
        with any empirical bound read from the fold's fitted training rows ``train``.  Every
        block of the stack is then ``[a, atom indicators, steps, H]``: the dose, one
        indicator per declared atom in ``atoms`` order, the step columns, and the history.
        A step column that is constant on a fold's stacked training rows, or equal there to
        an atom indicator, is dropped from that fold's design and its predictions.  A
        learner with ``policy_steps = "omit"`` receives ``[a, atom indicators, H]`` as with
        ``steps=None``.  A vector design takes no ``steps``.

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
    if steps is not None and omits_policy_steps(learner):
        steps = None
    if steps is not None and width != 1:
        raise DataError(
            "policy steps apply to a dose of one column; a vector design carries the "
            "indicators of its components itself"
        )
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

    def fit_on(
        rows: IntArray, fold: int, stacked: list[tuple[float, float, FloatArray]] = blocks
    ) -> _Fitted:
        keep = rows[mask[rows]]
        parts = [dose[keep] for _, _, dose in stacked]
        retained = None
        if steps is not None:
            columns = [_steps_at(steps, dose[keep, 0], keep, fold, keep) for _, _, dose in stacked]
            retained = _retained_steps(np.vstack(columns), np.vstack(parts)[:, width:])
            parts = [
                np.column_stack([part, column[:, retained]])
                for part, column in zip(parts, columns, strict=True)
            ]
        design = np.vstack([np.column_stack([part, h[keep]]) for part in parts])
        target = np.concatenate([np.full(keep.size, label) for label, _, _ in stacked])
        weight = np.concatenate([w[keep] * share for _, share, _ in stacked])
        unit = np.concatenate([units[keep] for _ in stacked])
        model = fit_learner(learner, design, target, weight, groups=unit, warn_unweighted=False)
        return _Fitted(model, fold, keep, retained)

    def ratio_at(fitted: _Fitted, rows: IntArray) -> list[FloatArray]:
        out = []
        for dose in requests:
            values = dose[rows]
            if fitted.retained is not None:
                assert steps is not None
                column = _steps_at(steps, values[:, 0], rows, fitted.fold, fitted.train)
                values = np.column_stack([values, column[:, fitted.retained]])
            u = predict_mean(fitted.model, np.column_stack([values, h[rows]]), "classification")
            u = np.minimum(np.clip(u, 0.0, 1.0), _ODDS_GUARD)
            out.append(np.asarray(u / (1.0 - u), dtype=float))
        return out

    results = [np.zeros(n) for _ in requests]
    if folds.is_single:
        everyone = np.arange(n, dtype=np.int64)
        for target, values in zip(results, ratio_at(fit_on(everyone, 0), everyone), strict=True):
            target[:] = values
        return results

    def run_fold(fold: int, train: IntArray, test: IntArray) -> tuple[IntArray, list[FloatArray]]:
        stacked = blocks if fold_blocks is None else fold_blocks[fold]
        return test, ratio_at(fit_on(train, fold, stacked), test)

    for test, parts in map_parallel(
        run_fold,
        [(fold, train, test) for fold, (train, test) in enumerate(folds)],
        n_jobs=n_jobs,
    ):
        for target, column in zip(results, parts, strict=True):
            target[test] = column
    return results


class _Fitted:
    """One fold's classifier, its fold, its fitted training rows and its retained steps."""

    def __init__(self, model: Learner, fold: int, train: IntArray, retained: Any) -> None:
        self.model = model
        self.fold = fold
        self.train = train
        self.retained = retained


def _steps_at(
    steps: PolicySteps, dose: FloatArray, rows: IntArray, fold: int, train: IntArray
) -> FloatArray:
    """The step columns at ``dose`` on ``rows``, as an ``(rows.size, S)`` float block."""
    block = np.asarray(steps(dose, rows, fold, train), dtype=float)
    return block.reshape(rows.size, -1)


def _retained_steps(columns: FloatArray, atoms: FloatArray) -> IntArray:
    """The step columns a fold keeps: not constant on its stacked training rows, and not
    equal there to an atom indicator.  Their order is the lowered map's."""
    keep = []
    for j in range(columns.shape[1]):
        column = columns[:, j]
        if column.size == 0 or np.all(column == column[0]):
            continue
        if any(np.array_equal(column, atoms[:, k]) for k in range(atoms.shape[1])):
            continue
        keep.append(j)
    return np.asarray(keep, dtype=np.int64)


def _with_atoms(block: FloatArray, atoms: tuple[float, ...]) -> FloatArray:
    """A one-column dose block with the indicator of each declared atom appended.

    A vector treatment passes no ``atoms``: its design already holds each component's
    indicators (:meth:`cleverly.data.CausalData.component_design`), built by the same
    :func:`~cleverly.data.validate.atom_block`.
    """
    if not atoms:
        return block
    if block.shape[1] != 1:
        raise DataError(
            "atoms= applies to a dose of one column; a vector design carries the indicators "
            "of its components' atoms itself"
        )
    return atom_block(block[:, 0], atoms)


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
