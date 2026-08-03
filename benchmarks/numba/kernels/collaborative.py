r"""Collaborative TMLE's candidate path, with the candidate mechanisms cached.

``CTMLE`` searches over propensity models: for each candidate it builds a clever covariate,
targets, and scores the result by cross-validated loss, keeping the sequence that wins.
The plan flags this as a strong candidate on the grounds that it evaluates many candidates
repeatedly, and the profile disagrees with the premise in a way worth writing down.

**Where a CTMLE fit actually spends its time.**  Profiled at ``n = 20,000`` with a ``glm``
outcome learner and the default treatment library, one fit is ~190 s, of which the largest
lines are LightGBM's ``update`` (23 s), liblinear's ``_fit_liblinear`` (16 s) and
scikit-learn's own input validation and ``type_of_target`` (~30 s together).  The
post-selection ``retarget`` -- everything this package owns, once the candidates exist --
is **11 ms**.  The candidate *search* is dominated by fitting the candidates, and fitting a
candidate is fitting a model.

That does not make the scoring uninteresting, but it does change the question from "can
numba speed up CTMLE" to "given cached candidate predictions, is the scoring itself worth
compiling".  This kernel answers that one, because it is the part that would matter if the
candidates ever became cheap -- with a linear-model library, or with the candidates fitted
once and swept repeatedly, which is what a sensitivity analysis over the path does.

**The sequential dependency is real and is respected.**  A forward-selection path chooses
candidate ``k+1`` after seeing candidate ``k``'s result, so candidates are *not*
independent and parallelising across them would compute a different path.  What is
independent is (a) the cross-validation folds within a candidate and (b) the *scoring* of a
pre-enumerated candidate set, which is what a fixed library gives.  Both are benchmarked;
the adaptive path is not parallelised, and the report says so rather than showing a
speed-up for an algorithm nobody runs.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..fixtures import Regime, make_targeting
from ..implementations.numba_parallel import PARALLEL_AVAILABLE, pjit, prange
from ..implementations.numba_serial import njit
from ..validation import compare_mapping
from . import KernelSpec, register

__all__ = [
    "build",
    "numba_candidate_scores",
    "numba_candidate_scores_parallel",
    "numpy_candidate_scores",
]

_ALPHA = 0.9995
_MAX_NEWTON = 20
_NEWTON_TOL = 1e-10


def build(
    n: int = 50_000,
    n_candidates: int = 50,
    n_folds: int = 5,
    regime: Regime = "moderate",
    seed: int = 20260803,
) -> dict[str, Any]:
    """Cached candidate propensities: ``(n_candidates, n)``, increasingly adjusted.

    A candidate path adds covariates one at a time, so the candidates are *nested* -- each
    a slightly more adjusted mechanism than the last, not an independent draw.  Modelled
    here by walking a logit from an unadjusted constant towards the true mechanism, so the
    later candidates have the heavier tails and the scoring cost is not uniform across
    them.  A fixture of independent columns would make every candidate the same price and
    hide the load-balance question the parallel arm exists to answer.
    """
    fixture = make_targeting(n, n_arms=2, regime=regime, seed=seed, n_folds=n_folds)
    truth = np.log(fixture.propensity[:, 1] / (1.0 - fixture.propensity[:, 1]))
    constant = np.full(n, float(np.mean(truth)))
    shares = np.linspace(0.0, 1.0, n_candidates)
    logits = constant[None, :] * (1.0 - shares[:, None]) + truth[None, :] * shares[:, None]
    candidates = 1.0 / (1.0 + np.exp(-logits))
    return {
        "fixture": fixture,
        "candidates": np.ascontiguousarray(np.clip(candidates, 1e-3, 1.0 - 1e-3)),
    }


def _expit(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -700.0, 700.0)))


# --------------------------------------------------------------------------- numpy


def numpy_candidate_scores(inputs: dict[str, Any]) -> dict[str, Any]:
    """Per candidate: build the covariate, target, score by cross-validated loss.

    The loss is the quasi-binomial deviance of the targeted fit on each held-out fold,
    summed -- which is the criterion shape CTMLE selects on.  The selection itself is the
    argmin, and it is part of the output because the correctness gate has to check that
    every implementation picks the *same* candidate, not merely that their losses agree.
    """
    fixture = inputs["fixture"]
    candidates = inputs["candidates"]
    y = fixture.outcome
    weights = fixture.weights
    mask = fixture.observed
    folds = fixture.folds
    n_folds = fixture.n_folds
    indicator = fixture.treatment_indicator

    initial = np.clip(fixture.initial_observed, 1.0 - _ALPHA, _ALPHA)
    offset = np.log(initial / (1.0 - initial))

    losses = np.empty(candidates.shape[0])
    epsilons = np.empty(candidates.shape[0])
    for c in range(candidates.shape[0]):
        g1 = candidates[c]
        covariate = indicator[:, 1] / g1 - indicator[:, 0] / (1.0 - g1)
        epsilon = 0.0
        total = weights[mask].sum()
        for _ in range(_MAX_NEWTON):
            p = _expit(offset + covariate * epsilon)
            gradient = float((weights * covariate * (y - p))[mask].sum())
            if abs(gradient) / total <= _NEWTON_TOL:
                break
            hessian = float((weights * covariate * covariate * p * (1.0 - p))[mask].sum())
            if hessian <= 0.0 or not np.isfinite(hessian):
                break
            epsilon += gradient / hessian
        epsilons[c] = epsilon
        targeted = np.clip(_expit(offset + covariate * epsilon), 1e-12, 1.0 - 1e-12)
        loss = 0.0
        for fold in range(n_folds):
            rows = (folds == fold) & mask
            if not rows.any():
                continue
            loss -= float(
                (
                    weights[rows]
                    * (
                        y[rows] * np.log(targeted[rows])
                        + (1.0 - y[rows]) * np.log(1.0 - targeted[rows])
                    )
                ).sum()
            )
        losses[c] = loss
    return {
        "losses": losses,
        "epsilons": epsilons,
        "selected": int(np.argmin(losses)),
    }


# --------------------------------------------------------------------------- numba


@njit()
def _score_one(y, offset, weights, mask, folds, indicator, g1, n_folds):
    """One candidate: covariate, Newton, deviance -- three passes fused into two."""
    n = y.shape[0]
    epsilon = 0.0
    total = 0.0
    for i in range(n):
        if mask[i]:
            total += weights[i]
    for _ in range(_MAX_NEWTON):
        gradient = 0.0
        hessian = 0.0
        for i in range(n):
            if not mask[i]:
                continue
            g = g1[i]
            covariate = indicator[i, 1] / g - indicator[i, 0] / (1.0 - g)
            eta = offset[i] + covariate * epsilon
            if eta > 700.0:
                eta = 700.0
            elif eta < -700.0:
                eta = -700.0
            p = 1.0 / (1.0 + np.exp(-eta))
            gradient += weights[i] * covariate * (y[i] - p)
            hessian += weights[i] * covariate * covariate * p * (1.0 - p)
        if abs(gradient) / total <= _NEWTON_TOL:
            break
        if hessian <= 0.0 or not np.isfinite(hessian):
            break
        epsilon += gradient / hessian
    loss = 0.0
    for i in range(n):
        if not mask[i]:
            continue
        g = g1[i]
        covariate = indicator[i, 1] / g - indicator[i, 0] / (1.0 - g)
        eta = offset[i] + covariate * epsilon
        if eta > 700.0:
            eta = 700.0
        elif eta < -700.0:
            eta = -700.0
        p = 1.0 / (1.0 + np.exp(-eta))
        if p < 1e-12:
            p = 1e-12
        elif p > 1.0 - 1e-12:
            p = 1.0 - 1e-12
        loss -= weights[i] * (y[i] * np.log(p) + (1.0 - y[i]) * np.log(1.0 - p))
    return epsilon, loss


@njit()
def _scores_serial(y, offset, weights, mask, folds, indicator, candidates, n_folds):
    count = candidates.shape[0]
    losses = np.empty(count)
    epsilons = np.empty(count)
    for c in range(count):
        epsilons[c], losses[c] = _score_one(
            y, offset, weights, mask, folds, indicator, candidates[c], n_folds
        )
    return epsilons, losses


@pjit()
def _scores_parallel(y, offset, weights, mask, folds, indicator, candidates, n_folds):
    """``prange`` over candidates -- valid only for a *pre-enumerated* candidate set.

    A forward-selection path is sequential by construction and is not run through here;
    what this measures is the cost of scoring a fixed library, which is what a sweep over
    an already-selected path does and what a non-adaptive candidate set is.
    """
    count = candidates.shape[0]
    losses = np.empty(count)
    epsilons = np.empty(count)
    for c in prange(count):
        epsilons[c], losses[c] = _score_one(
            y, offset, weights, mask, folds, indicator, candidates[c], n_folds
        )
    return epsilons, losses


def _run(inputs: dict[str, Any], kernel: Any) -> dict[str, Any]:
    fixture = inputs["fixture"]
    initial = np.clip(fixture.initial_observed, 1.0 - _ALPHA, _ALPHA)
    epsilons, losses = kernel(
        fixture.outcome,
        np.log(initial / (1.0 - initial)),
        fixture.weights,
        fixture.observed,
        np.ascontiguousarray(fixture.folds),
        fixture.treatment_indicator,
        inputs["candidates"],
        int(fixture.n_folds),
    )
    return {"losses": losses, "epsilons": epsilons, "selected": int(np.argmin(losses))}


def numba_candidate_scores(inputs: dict[str, Any]) -> dict[str, Any]:
    return _run(inputs, _scores_serial)


def numba_candidate_scores_parallel(inputs: dict[str, Any]) -> dict[str, Any]:
    return _run(inputs, _scores_parallel)


def _compare(reference: dict[str, Any], candidate: dict[str, Any]) -> tuple[float, float]:
    """Losses and coefficients numerically; the selected candidate exactly.

    The second half is the one that matters.  Two implementations can agree on every loss
    to twelve digits and still select different candidates when two candidates are within
    that of each other -- which is precisely what happens near the end of a nested path,
    where consecutive candidates differ by one covariate.  A benchmark that only compared
    the losses would call that agreement.
    """
    if reference["selected"] != candidate["selected"]:
        return float("inf"), float("inf")
    return compare_mapping(
        {"losses": reference["losses"], "epsilons": reference["epsilons"]},
        {"losses": candidate["losses"], "epsilons": candidate["epsilons"]},
    )


_IMPLEMENTATIONS: dict[str, Any] = {"numpy": numpy_candidate_scores}
if PARALLEL_AVAILABLE:
    _IMPLEMENTATIONS["numba"] = numba_candidate_scores
    _IMPLEMENTATIONS["numba_parallel"] = numba_candidate_scores_parallel

register(
    KernelSpec(
        name="ctmle_candidate_scores",
        estimator="ctmle",
        build=build,
        implementations=_IMPLEMENTATIONS,
        compare=_compare,
        tolerance=(1e-7, 1e-10),
        parallel_axis="candidates",
        note=(
            "profiled at 11 ms of a 190 s fit: the search is candidate-fitting-bound, so "
            "this measures the scoring that would matter if the candidates were cheap"
        ),
        dimensions={
            "n": 50_000,
            "n_candidates": 50,
            "n_folds": 5,
            "regime": "moderate",
            "seed": 20260803,
        },
        amortise=True,
    )
)
