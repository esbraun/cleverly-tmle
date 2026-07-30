"""The score of the fluctuation submodel, and the scale it is judged against.

These four helpers are shared by every solver and by the estimator that stitches
fold-specific fluctuations back together.  They live in their own module because
they had previously been written out three times -- once in
:mod:`cleverly.fluctuation.iterative`, once again in
:mod:`cleverly.fluctuation.one_step` (which imported the private names from its
sibling), and a third near-verbatim copy in :mod:`cleverly.estimators.tmle`.  Three
implementations of one quantity is three chances for them to drift, and the score is
precisely the quantity the whole method is defined by: TMLE is the estimator that
solves ``P_n D*(hat P) = 0``, and :func:`cleverly.validation.score_check` audits how
close to zero it got.

Everything here takes and returns plain arrays, so this module imports nothing from
its siblings and no import cycle is possible.
"""

from __future__ import annotations

import numpy as np

from .._typing import BoolArray, FloatArray

__all__ = ["quasi_loglik", "relative_score", "score_columns", "score_scale"]


def score_columns(
    y: FloatArray,
    q_star: FloatArray,
    h: FloatArray,
    weights: FloatArray,
    mask: BoolArray,
) -> FloatArray:
    """``mean(w * h * (Y - Q*))`` over observed rows, scaled by the full sample.

    The mean is taken over *all* ``n`` rows, not just the observed ones, because the
    estimating equation carries a ``Delta`` factor: unobserved rows contribute a
    genuine zero rather than being excluded from the average.

    ``h`` is always the *unweighted* clever covariate, even when the fluctuation was
    fit in its weighted form (R's ``target.gwt``).  The weighted form is a
    reparameterisation that solves the same equation, so the score has to be reported
    on the one canonical scale or two fits of the same model would not be comparable.
    """
    residual = np.zeros_like(y)
    residual[mask] = y[mask] - q_star[mask]
    contribution = (weights * residual)[:, None] * h
    return np.asarray(contribution.mean(axis=0), dtype=float)


def score_scale(h: FloatArray, weights: FloatArray, mask: BoolArray) -> FloatArray:
    """Per-column ``mean(|w * h|)``: the largest the score could be.

    The residual ``Y - Q*`` is bounded by one on the ``[0, 1]`` outcome scale, so this
    bounds ``|score|`` and makes the ratio dimensionless.
    """
    contribution = np.zeros_like(h)
    contribution[mask] = np.abs(weights[mask])[:, None] * np.abs(h[mask])
    return np.asarray(contribution.mean(axis=0), dtype=float)


def relative_score(score: FloatArray, scale: FloatArray) -> float:
    """Largest score component relative to its maximum possible magnitude."""
    if score.size == 0:
        return 0.0
    return float(np.max(np.abs(score) / np.maximum(scale, 1e-300)))


def quasi_loglik(y: FloatArray, p: FloatArray, weights: FloatArray) -> float:
    """Weighted binomial quasi-log-likelihood, valid for ``y`` in ``[0, 1]``."""
    q = np.clip(p, 1e-15, 1.0 - 1e-15)
    return float(np.sum(weights * (y * np.log(q) + (1.0 - y) * np.log(1.0 - q))))
