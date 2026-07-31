r"""A longitudinal process with a known truth, and time-varying confounding in it.

The point of the process is the node :math:`L_2`: it is caused by :math:`A_1` and it
causes both :math:`A_2` and :math:`Y`.  A point-treatment analysis has no right answer
available -- adjusting for :math:`L_2` blocks the part of :math:`A_1`'s effect that runs
through it, and not adjusting for it leaves :math:`A_2` confounded -- so this is the
structure that separates a longitudinal estimator from a sequence of cross-sectional
ones.  ``tests/e2e/test_ltmle.py`` checks that separation numerically rather than
asserting it here.

The truth is computed by **quadrature, not simulation**.  Under an intervention the
treatment and censoring mechanisms drop out entirely, so what is left is an expectation
of the outcome regression over three independent standard normals, which
Gauss--Hermite evaluates to machine precision.  A Monte Carlo truth would carry an error
of its own into every coverage assertion built on it.
"""

from __future__ import annotations

from functools import cache
from typing import Any

import numpy as np

from .._typing import FloatArray
from ..utils.bounds import expit
from ..utils.frames import frame_from_dict

__all__ = ["longitudinal_truth", "make_longitudinal"]

#: Coefficients of the outcome regression, ``logit P(Y = 1 | A1, A2, L2, W)``.
_Y = {"intercept": -0.4, "a1": 0.5, "a2": 0.8, "l2": 0.4, "w1": 0.3, "w2": -0.2, "kink": 0.5}

#: ``L2 = l2_w1 * W1 + l2_a1 * A1 + noise``, the time-varying confounder ``A1`` moves.
_L2 = {"w1": 0.6, "a1": 0.9}


def _outcome_probability(
    w1: FloatArray,
    w2: FloatArray,
    l2: FloatArray,
    a1: FloatArray | float,
    a2: FloatArray | float,
) -> FloatArray:
    """``P(Y = 1 | ...)``, at a regimen's arms or at each unit's own.

    The ``tanh`` term keeps the regression off a linear index, so a ``glm`` nuisance
    learner here is misspecified rather than accidentally exact.
    """
    index = (
        _Y["intercept"]
        + _Y["a1"] * a1
        + _Y["a2"] * a2
        + _Y["l2"] * l2
        + _Y["w1"] * w1
        + _Y["w2"] * w2
        + _Y["kink"] * np.tanh(l2)
    )
    return expit(index)


@cache
def longitudinal_truth(a1: float, a2: float, nodes: int = 48) -> float:
    r"""``E[Y_{(a1, a2)}]``, the mean outcome had everyone followed that regimen.

    Under the intervention the mechanism is gone and :math:`L_2` is
    :math:`N(0.6 W_1 + 0.9 a_1, 1)`, so the parameter is an expectation of the outcome
    regression over three independent standard normals -- :math:`W_1`, :math:`W_2` and
    :math:`L_2`'s own noise.  A product Gauss--Hermite rule with ``nodes`` points per
    dimension evaluates it to well under ``1e-10``, which is what makes it usable as a
    *truth* in a coverage study rather than as a second estimate.
    """
    points, weights = np.polynomial.hermite_e.hermegauss(nodes)
    weights = weights / np.sqrt(2.0 * np.pi)
    w1 = points.reshape(-1, 1, 1)
    w2 = points.reshape(1, -1, 1)
    noise = points.reshape(1, 1, -1)
    mass = weights.reshape(-1, 1, 1) * weights.reshape(1, -1, 1) * weights.reshape(1, 1, -1)
    l2 = _L2["w1"] * w1 + _L2["a1"] * a1 + noise
    return float(np.sum(mass * _outcome_probability(w1, w2, l2, a1, a2)))


def make_longitudinal(
    n: int = 2000,
    *,
    seed: int | np.random.Generator | None = None,
    censoring: bool = True,
    cluster_size: int | None = None,
    backend: str = "pandas",
) -> tuple[Any, dict[str, float]]:
    """Two time points, a binary outcome and (optionally) monotone censoring.

    Returns ``(frame, truth)``.  The frame is wide -- a row per unit, a column per node
    -- with columns ``W1``, ``W2``, ``A1``, ``C1``, ``L2``, ``A2``, ``C2``, ``Y`` in time
    order, and the nodes after a unit's censoring time set to ``nan``, which is the form
    :class:`~cleverly.longitudinal.LongitudinalData` requires.

    ``truth`` holds ``ey_regimen[...]`` for the four static regimens and
    ``ate_regimen[always vs never]``, under the labels a fit reports them by, so a test
    can look up the truth with the name it read off the result.

    ``cluster_size`` adds an ``id`` column and, with it, an *unobserved* effect shared
    within each cluster that moves both treatment decisions and the outcome -- so the
    influence curves are correlated within a cluster and ignoring ``id=`` understates the
    standard error.  The same construction as
    :func:`~cleverly.datasets.make_clustered`, and for the same reason: an ``id`` column
    over independent rows makes a cluster-robust variance *equal* the plain one, which
    tests nothing.  The counterfactual means are unchanged -- the shared effect is
    marginalised over and enters neither ``L2`` nor the outcome regression's form -- so
    ``truth`` still holds.
    """
    rng = np.random.default_rng(seed)
    w1 = rng.standard_normal(n)
    w2 = rng.standard_normal(n)

    if cluster_size is None:
        ids = None
        shared = np.zeros(n)
    else:
        ids = np.arange(n) // cluster_size
        # Drawn per cluster and repeated, and deliberately not among the covariates.
        shared = rng.standard_normal(int(ids.max()) + 1)[ids]

    a1 = rng.binomial(1, expit(0.3 * w1 - 0.4 * w2 + 0.8 * shared)).astype(float)
    c1 = (
        rng.binomial(1, expit(2.2 + 0.3 * w1 - 0.3 * a1)).astype(float) if censoring else np.ones(n)
    )
    alive1 = c1 == 1.0

    l2 = _L2["w1"] * w1 + _L2["a1"] * a1 + rng.standard_normal(n)
    a2 = rng.binomial(1, expit(0.5 * l2 + 0.6 * a1 - 0.2 * w2 + 0.8 * shared)).astype(float)
    c2 = rng.binomial(1, expit(2.4 + 0.2 * l2)).astype(float) if censoring else np.ones(n)
    alive2 = alive1 & (c2 == 1.0)

    probability = _outcome_probability(w1, w2, l2, a1, a2)
    if cluster_size is not None:
        # Tilt the outcome by the same shared effect, so the residual -- and with it the
        # influence curve -- carries the within-cluster correlation.
        probability = expit(np.log(probability / (1.0 - probability)) + 0.8 * shared)
    y = rng.binomial(1, probability).astype(float)

    payload = {
        "W1": w1,
        "W2": w2,
        "A1": a1,
        "C1": c1,
        "L2": np.where(alive1, l2, np.nan),
        "A2": np.where(alive1, a2, np.nan),
        "C2": np.where(alive1, c2, np.nan),
        "Y": np.where(alive2, y, np.nan),
    }
    if not censoring:
        del payload["C1"]
        del payload["C2"]
    if ids is not None:
        payload["id"] = ids.astype(float)

    truth = {
        f"ey_regimen[{label}]": longitudinal_truth(plan[0], plan[1])
        for label, plan in (
            ("always", (1.0, 1.0)),
            ("never", (0.0, 0.0)),
            ("early", (1.0, 0.0)),
            ("late", (0.0, 1.0)),
        )
    }
    truth["ate_regimen[always vs never]"] = truth["ey_regimen[always]"] - truth["ey_regimen[never]"]
    return frame_from_dict(payload, backend=backend), truth
