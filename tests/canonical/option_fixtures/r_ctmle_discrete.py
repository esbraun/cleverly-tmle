"""A transcription of R ``ctmleDiscrete(preOrder = TRUE)`` for the two ``ctmle`` option fixtures.

The fixtures ``ctmle_patience.json`` and ``ctmle_wg.json`` hold R runs of ``ctmleDiscrete`` (ctmle
0.1.2, commit 18de559).  ``CTMLE`` does not reproduce those runs number for number, because R's
discrete path differs from the package in two ways that the X22 options do not touch:

* R's cross-validation fits each candidate mechanism on every row, the validation rows included
  (``R/functions_discrete.R``, lines 32, 81, 508 and 557).  ``CTMLE`` fits it on the training rows.
* R fluctuates with the one ATE clever covariate ``hAW`` (lines 33 and 82).  ``CTMLE`` targets the
  arm means with one column per arm.

So the comparison has two halves.  This transcription must reproduce every R number of every run:
the scores, the selected index, the estimate and the added terms.  That proves the reading of R,
its patience counter and its ``Wg`` pool included.  The options are then compared on what they
control: the patience counter on ``CTMLE``'s own scores, and the candidate pool.

The transcription covers the arguments the fixtures pass: a binary outcome, ``Q`` given,
``like_type = "loglike"``, ``PEN = FALSE``, ``preOrder = TRUE`` and the default ``stopFactor``.
It keeps R's construction criterion, which adds ``var(D*)`` scaled by the training share whatever
``PEN`` is (lines 40, 70 and 88).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.special import expit, logit

__all__ = ["DiscreteRun", "ctmle_discrete", "patience_stop"]

#: ``stage1`` bounds a given binomial ``Q`` at ``(1 - alpha, alpha)`` with ``alpha = 0.995``.
Q_BOUND = 0.005
#: The Newton steps stop when no coefficient moves by more than this.
NEWTON_TOLERANCE = 1e-14


@dataclass(frozen=True)
class DiscreteRun:
    """One ``ctmleDiscrete`` run, in the fields the fixture JSON records.

    Parameters
    ----------
    best_k : int
        ``cv.res$best_k``, one-based.
    est : float
        ``est``, the estimate of candidate ``best_k``.
    cv_score : tuple of float
        ``cv.res$penlikelihood``, ``inf`` where the patience counter left a candidate unscored.
    terms : tuple of str
        ``candidates$terms``: ``"1"``, then the covariates of the final construction.
    """

    best_k: int
    est: float
    cv_score: tuple[float, ...]
    terms: tuple[str, ...]


def _logistic(y: np.ndarray, x: np.ndarray, offset: np.ndarray | None = None) -> np.ndarray:
    """The unpenalized logistic MLE by Newton steps, R ``glm(family = binomial)`` from zero."""
    offset = np.zeros(y.size) if offset is None else offset
    beta = np.zeros(x.shape[1])
    for _ in range(200):
        mu = expit(offset + x @ beta)
        step = np.linalg.solve((x * (mu * (1.0 - mu))[:, None]).T @ x, x.T @ (y - mu))
        beta = beta + step
        if np.max(np.abs(step)) < NEWTON_TOLERANCE:
            break
    return beta


def _clever(a: np.ndarray, w: np.ndarray, columns: Sequence[int], gbound: float) -> np.ndarray:
    """``Tx_mech``: ``[hAW, h1W, h0W]`` from ``glm(A ~ 1 + W[, columns])`` on every row."""
    design = np.column_stack([np.ones(a.size), w[:, list(columns)]])
    g1 = np.clip(expit(design @ _logistic(a, design)), gbound, 1.0 - gbound)
    h1, h0 = 1.0 / g1, -1.0 / (1.0 - g1)
    return np.column_stack([np.where(a == 1.0, h1, h0), h1, h0])


def _epsilon(y: np.ndarray, q: np.ndarray, h: np.ndarray) -> float:
    """``coef(glm(Y ~ -1 + offset(Q[, "QAW"]) + h[, "hAW"]))``."""
    return float(_logistic(y, h[:, :1], offset=q[:, 0])[0])


def _loss(y: np.ndarray, q: np.ndarray) -> float:
    return float(-np.sum(y * np.log(q) + (1.0 - y) * np.log(1.0 - q)))


def _criterion(y: np.ndarray, q: np.ndarray, h: np.ndarray, share: float) -> float:
    """The construction score: the loss plus ``calc_varIC(Y, Qstar, hAW)[1]`` times the share."""
    curve = (y - q[:, 0]) * h[:, 0] + q[:, 1] - q[:, 2] - (q[:, 1].mean() - q[:, 2].mean())
    return _loss(y, q[:, 0]) + float(np.var(curve, ddof=1)) * share


def _select_terms(
    y: np.ndarray,
    q: np.ndarray,
    a: np.ndarray,
    w: np.ndarray,
    current: list[int],
    remaining: list[int],
    train: np.ndarray,
    first: bool,
    gbound: float,
) -> tuple[list[int], list[float]]:
    """``select_terms`` under ``preOrder = TRUE``: the next clever covariate's terms and epsilons."""
    share = float(train.sum()) / train.size
    epsilons: list[float] = []
    scores = [np.inf]
    if first:
        h = _clever(a, w, current, gbound)[train]
        eps = _epsilon(y, q, h)
        epsilons.append(eps)
        scores = [_criterion(y, expit(q + eps * h), h, share)]
    previous, terms = list(current), list(current)
    counter, improving, total = 0, True, len(remaining)
    while remaining:
        counter += 1
        h = _clever(a, w, [*terms, remaining[0]], gbound)[train]
        eps = _epsilon(y, q, h)
        score = _criterion(y, expit(q + eps * h), h, share)
        score = score if np.isfinite(score) else np.inf
        improving = scores[-1] > score
        previous, terms = terms, [*terms, remaining[0]]
        scores.append(score)
        epsilons.append(eps)
        remaining = remaining[1:]
        if not improving or counter == total:
            break
    if not improving and counter > 1:
        # The last term did not improve the criterion, so it starts the next clever covariate.
        terms, epsilons = previous, epsilons[:-1]
    return terms, epsilons


def patience_stop(score: np.ndarray | Sequence[float], patience: int) -> int:
    """The number of candidates R's patience counter scores, from the whole-path scores.

    ``cv`` (``R/functions_discrete.R``, lines 626 to 646) keeps the best score so far and counts
    the scores above it.  A score at or below it resets the count.  Scoring stops once the count
    reaches ``patience``.

    Parameters
    ----------
    score : sequence of float
        The scores of the whole path, in path order.
    patience : int
        R's ``patience``.

    Returns
    -------
    int
        The number of candidates scored before the stop.
    """
    pointer, count = score[0], 0
    for position, value in enumerate(score, start=1):
        if value > pointer:
            count += 1
        if count >= patience:
            return position
        if value <= pointer:
            pointer, count = value, 0
    return len(score)


def ctmle_discrete(
    y: np.ndarray | Sequence[float],
    a: np.ndarray | Sequence[float],
    pool: np.ndarray,
    names: Sequence[str],
    q0: np.ndarray | Sequence[float],
    q1: np.ndarray | Sequence[float],
    fold: np.ndarray | Sequence[int],
    *,
    gbound: float = 0.025,
    patience: int | None = None,
) -> DiscreteRun:
    """``ctmleDiscrete(Wg = pool, Q = cbind(q0, q1), preOrder = TRUE, folds = ...)``.

    Parameters
    ----------
    y : sequence of float
        The outcome.
    a : sequence of float
        The treatment.
    pool : ndarray of float
        ``Wg[, order]``: the candidate covariates in path order.
    names : sequence of str
        The names of the columns of ``pool``.
    q0 : sequence of float
        The initial outcome regression at arm 0.
    q1 : sequence of float
        The initial outcome regression at arm 1.
    fold : sequence of int
        The fold of each row, from 0.
    gbound : float
        ``gbound``.
    patience : int or None
        ``patience``; ``None`` is ``patience = FALSE``.

    Returns
    -------
    DiscreteRun
        The fields the fixture JSON records.
    """
    y_ = np.asarray(y, dtype=float)
    a_ = np.asarray(a, dtype=float)
    w = np.asarray(pool, dtype=float)
    bounded = [np.clip(np.asarray(v, dtype=float), Q_BOUND, 1.0 - Q_BOUND) for v in (q0, q1)]
    q_start = logit(
        np.column_stack([np.where(a_ == 1.0, bounded[1], bounded[0]), bounded[1], bounded[0]])
    )
    labels = np.asarray(fold, dtype=int)
    folds = [np.flatnonzero(labels == k) for k in range(int(labels.max()) + 1)]
    score = _cross_validate(y_, a_, w, q_start, folds, gbound, patience)
    finite = score[np.isfinite(score)]
    best_k = int(np.flatnonzero(finite == finite.min()).max()) + 1
    width = best_k + 1 if best_k == 1 else best_k
    terms, epsilons, ends = _construct(y_, a_, w, q_start, width - 1, gbound)
    estimate = _final_estimate(a_, w, q_start, terms, epsilons, ends, gbound, best_k)
    return DiscreteRun(
        best_k=best_k,
        est=estimate,
        cv_score=tuple(float(value) for value in score),
        terms=("1", *(names[t] for t in terms)),
    )


def _cross_validate(
    y: np.ndarray,
    a: np.ndarray,
    w: np.ndarray,
    q_start: np.ndarray,
    folds: Sequence[np.ndarray],
    gbound: float,
    patience: int | None,
) -> np.ndarray:
    """``cv``: the summed validation loss of each candidate, ``inf`` where unscored."""
    n, total = y.size, w.shape[1] + 1
    v_count = len(folds)
    q = [q_start.copy() for _ in range(v_count)]
    terms: list[list[int]] = [[] for _ in range(v_count)]
    remaining = [list(range(w.shape[1])) for _ in range(v_count)]
    epsilons: list[list[float]] = [[] for _ in range(v_count)]
    losses: list[np.ndarray] = [np.empty(0) for _ in range(v_count)]
    next_term = [1] * v_count
    step, count, pointer, scored_before = 0, 0, np.inf, 0
    while True:
        step += 1
        likelihood = np.zeros(total)
        for v, test in enumerate(folds):
            train = np.ones(n, dtype=bool)
            train[test] = False
            if next_term[v] <= total:
                chosen, new_eps = _select_terms(
                    y[train], q[v][train], a, w, terms[v], remaining[v], train, step == 1, gbound
                )
                added = [t for t in chosen if t not in terms[v]]
                epsilons[v] += new_eps
                remaining[v] = [t for t in remaining[v] if t not in added]
                terms[v] = chosen
                size = len(chosen) + 1
                if step == 1:
                    losses[v] = np.full(size, np.inf)
                    next_term[v] = 1
                elif losses[v].size + len(added) <= total:
                    losses[v] = np.r_[losses[v], np.full(len(added), np.inf)]
                h = np.empty((n, 3))
                for j in range(next_term[v], size + 1):
                    h = _clever(a, w, chosen[: j - 1], gbound)
                    targeted = expit(q[v] + epsilons[v][j - 1] * h)
                    losses[v][j - 1] = _loss(y[test], targeted[test, 0])
                q[v] = q[v] + epsilons[v][-1] * h
                next_term[v] = max(next_term[v], size + 1)
            end = losses[v].size
            likelihood = np.r_[likelihood[:end] + losses[v], np.full(total - end, np.inf)]
        done = bool(np.max(likelihood) < np.inf)
        if patience is not None:
            scored = int(np.isfinite(likelihood).sum())
            if step == 1:
                pointer, scored_before = likelihood[0], 0
            for j in range(scored_before, scored):
                if likelihood[j] > pointer:
                    count += 1
                if count >= patience:
                    done = True
                if likelihood[j] <= pointer and not done:
                    pointer, count = likelihood[j], 0
            scored_before = scored
        if done:
            return likelihood


def _construct(
    y: np.ndarray, a: np.ndarray, w: np.ndarray, q_start: np.ndarray, limit: int, gbound: float
) -> tuple[list[int], list[float], list[int]]:
    """``construct_candidates`` on every row: the terms, their epsilons and the covariate ends."""
    train = np.ones(y.size, dtype=bool)
    q = q_start.copy()
    terms: list[int] = []
    remaining = list(range(limit))
    epsilons: list[float] = []
    sizes = [1]
    step = 0
    while remaining:
        step += 1
        chosen, new_eps = _select_terms(y, q, a, w, terms, remaining, train, step == 1, gbound)
        added = [t for t in chosen if t not in terms]
        epsilons += new_eps
        remaining = [t for t in remaining if t not in added]
        if step == 1:
            sizes = [1 + len(added)]
        else:
            sizes.append(len(added))
        q = q + epsilons[-1] * _clever(a, w, chosen, gbound)
        terms = chosen
        if limit <= sum(sizes) - 1:
            break
    end = min(len(terms), limit) + 1
    return terms[: end - 1], epsilons[:end], list(np.cumsum(sizes))


def _final_estimate(
    a: np.ndarray,
    w: np.ndarray,
    q_start: np.ndarray,
    terms: Sequence[int],
    epsilons: Sequence[float],
    ends: Sequence[int],
    gbound: float,
    best_k: int,
) -> float:
    """``evaluate_candidates``: the estimate of candidate ``best_k``."""
    q = q_start.copy()
    covariate = 0
    estimate = float("nan")
    for i in range(len(terms) + 1):
        h = _clever(a, w, terms[:i], gbound)
        targeted = expit(q + epsilons[i] * h)
        if i + 1 == best_k:
            estimate = float(targeted[:, 1].mean() - targeted[:, 2].mean())
        if covariate < len(ends) and i + 1 == ends[covariate]:
            covariate += 1
            q = q + epsilons[i] * h
    return estimate
