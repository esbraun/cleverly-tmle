r"""The logistic projection of a finite law, and the plug-in second moment it gives.

The omitted-variable bound under ``nu2_estimator="plugin"`` squares the Riesz representer
built from the fitted treatment mechanism.  When that mechanism is a scikit-learn
``LogisticRegression``, the fit is an M-estimator, and the estimate is the plug-in functional

.. math::

    \nu^2(P) = E_{P_w}\bigl[\alpha_b(g_{\gamma(P)})^2\bigr],

where :math:`P_w` is the law tilted by the observation weights, :math:`\gamma(P)` is the
logistic projection of :math:`P_w`, and :math:`\alpha_b` is the representer of the truncated
mechanism.  This module writes that functional from the cell probabilities of a finite law.
It imports no package code, so a Gateaux derivative of it is an independent comparator for
the curve the package reports.

The law is a vector of cell probabilities over a finite support.  Each cell carries a
design row :math:`x`, an arm index and a weight.  The arms are ``0, ..., K - 1``, and the
projection is written in reference form against arm ``0``.  The softmax is invariant to the
reference, and so is the symmetric penalty below, so the reference changes no value.

**Penalty.**  scikit-learn minimises :math:`C \sum_i \tilde w_i \ell_i + \tfrac12
\sum_k \|\beta_k\|^2_S`, with mean-one weights :math:`\tilde w` and the intercept left out of
:math:`S`.  Divided by :math:`C n`, that is :math:`E_{P_w}[\ell] + \theta^\top P \theta /
(2 C n)`, with ``n`` held fixed along the contamination path.  The smallest symmetric norm
with given reference-form differences is :math:`\sum_k \|\beta_k - \bar\beta\|^2`, which is
:math:`\theta^\top \{(I_{K-1} - \mathbf 1 \mathbf 1^\top / K) \otimes S\} \theta`
(:func:`penalty_matrix`).

**Complex step.**  Every operation is arithmetic, :func:`numpy.exp` or a linear solve, and
the Newton iteration runs a fixed number of steps with no convergence test.  The functional
is therefore analytic in the cell probabilities, and a complex step differentiates it to full
precision.  Truncation reads its mask off the real part of the mechanism.  The mask is
locally constant when no cell sits on a bound, so the clipped functional stays analytic
there.
"""

from __future__ import annotations

from typing import Any

import numpy as np

#: Newton steps of the projection, run without a convergence test so that the solve stays
#: analytic.  Newton converges quadratically in the value and in its derivative alike, so
#: the imaginary part is exact once the real part stops moving.
NEWTON_STEPS = 60


def penalty_matrix(n_arms: int, d: int, *, C: float, n: int, intercept: bool = True) -> Any:
    r"""The reference-form ridge matrix :math:`P / (C n)`.

    Parameters
    ----------
    n_arms : int
        The number of arms ``K``.
    d : int
        The width of the design, with the intercept column first when ``intercept``.
    C : float
        scikit-learn's inverse penalty strength; ``numpy.inf`` gives a zero matrix.
    n : int
        The sample size the penalty is divided by.
    intercept : bool
        Whether column 0 is an unpenalized intercept.

    Returns
    -------
    ndarray
        The ``((K - 1) d, (K - 1) d)`` matrix.
    """
    p = (n_arms - 1) * d
    if np.isinf(C):
        return np.zeros((p, p))
    slopes = np.eye(d)
    if intercept:
        slopes[0, 0] = 0.0
    arm = np.eye(n_arms - 1) - np.ones((n_arms - 1, n_arms - 1)) / n_arms
    if n_arms == 2:
        arm = np.ones((1, 1))
    return np.kron(arm, slopes) / (C * n)


def softmax(design: Any, theta: Any) -> Any:
    """The ``(S, K)`` probabilities of the reference-form coefficients, arm 0 first.

    Parameters
    ----------
    design : ndarray
        The ``(S, d)`` design.
    theta : ndarray
        The ``(K - 1, d)`` coefficients against arm 0.

    Returns
    -------
    ndarray
        One row per cell; complex when ``theta`` is.
    """
    eta = np.asarray(design) @ np.asarray(theta).T
    expd = np.exp(np.column_stack([np.zeros(eta.shape[0]), eta]))
    return expd / expd.sum(axis=1, keepdims=True)


def logistic_projection(
    mass: Any,
    design: Any,
    arm: Any,
    n_arms: int,
    *,
    penalty: Any = None,
    steps: int = NEWTON_STEPS,
) -> Any:
    r"""The reference-form logistic projection of a finite law.

    The root of :math:`E_q[x \otimes (e_A - g_\theta(x))_{-0}] = \Pi\theta`, with
    :math:`q` the cell masses normalised to sum to one and :math:`\Pi` the ``penalty``.

    Parameters
    ----------
    mass : ndarray
        ``(S,)`` nonnegative cell masses, the weighted cell probabilities; complex allowed.
    design : ndarray
        ``(S, d)`` design of each cell.
    arm : ndarray
        ``(S,)`` arm index of each cell, in ``0, ..., K - 1``.
    n_arms : int
        The number of arms ``K``.
    penalty : ndarray or None
        The :func:`penalty_matrix`, or ``None`` for none.
    steps : int
        Newton steps from zero.

    Returns
    -------
    ndarray
        ``(K - 1, d)`` coefficients.
    """
    x = np.asarray(design, dtype=float)
    q = np.asarray(mass) / np.sum(mass)
    s, d = x.shape
    k = n_arms - 1
    indicator = np.zeros((s, n_arms))
    indicator[np.arange(s), np.asarray(arm, dtype=int)] = 1.0
    pen = np.zeros((k * d, k * d)) if penalty is None else np.asarray(penalty)
    theta = np.zeros(k * d, dtype=complex if np.iscomplexobj(q) else float)
    for _ in range(steps):
        g = softmax(x, theta.reshape(k, d))[:, 1:]
        residual = indicator[:, 1:] - g
        gradient = np.einsum("s,sm,sj->mj", q, residual, x).reshape(-1) - pen @ theta
        curvature = np.einsum("s,sm,sq,sj,sl->mjql", q, g, -g, x, x)
        for m in range(k):
            curvature[m, :, m, :] += np.einsum("s,s,sj,sl->jl", q, g[:, m], x, x)
        information = curvature.reshape(k * d, k * d) + pen
        theta = theta + np.linalg.solve(information, gradient)
    return theta.reshape(k, d)


def truncate(g: Any, bounds: tuple[float, float] | None) -> tuple[Any, Any]:
    """The mechanism clipped into ``bounds``, and the cells that moved.

    Two arms clip arm 1 and take arm 0 as its complement, and mark both columns of a row
    that moved.  More arms clip cell by cell.  The mask is read off the real part.

    Parameters
    ----------
    g : ndarray
        ``(S, K)`` mechanism.
    bounds : tuple of float or None
        ``(lower, upper)``, or ``None`` for no truncation.

    Returns
    -------
    tuple of ndarray
        The bounded mechanism and the ``(S, K)`` boolean mask of moved cells.
    """
    if bounds is None:
        return g, np.zeros(np.shape(g), dtype=bool)
    lower, upper = bounds
    if g.shape[1] == 2:
        one = g[:, 1]
        below, above = np.real(one) < lower, np.real(one) > upper
        clipped = np.where(below, lower, np.where(above, upper, one))
        moved = below | above
        return np.column_stack([1.0 - clipped, clipped]), np.column_stack([moved, moved])
    below, above = np.real(g) < lower, np.real(g) > upper
    return np.where(below, lower, np.where(above, upper, g)), below | above


def parse(estimand: str) -> tuple[str, int, int | None]:
    """``"ey[a]"``, ``"ate[a vs r]"``, ``"att[a vs r]"`` or ``"atc[a vs r]"`` as its parts.

    Parameters
    ----------
    estimand : str
        The name, on arm indices.

    Returns
    -------
    tuple
        The stem, the arm and the reference arm (``None`` for a mean).
    """
    stem, _, rest = estimand.partition("[")
    parts = [int(part) for part in rest[:-1].split(" vs ")]
    return stem, parts[0], (parts[1] if len(parts) > 1 else None)


def representer(estimand: str, arm: Any, g: Any, share: Any = None) -> Any:
    r"""The Riesz representer of each cell.

    Parameters
    ----------
    estimand : str
        A name :func:`parse` reads.
    arm : ndarray
        ``(S,)`` arm index of each cell.
    g : ndarray
        ``(S, K)`` bounded mechanism.
    share : Any
        The weighted share of the conditioning arm, for ``att`` and ``atc``.

    Returns
    -------
    ndarray
        ``(S,)`` values; the cross terms vanish when squared, since the arms are disjoint.
    """
    stem, a, r = parse(estimand)
    on_a = (np.asarray(arm) == a).astype(float)
    if stem == "ey":
        return on_a / g[:, a]
    assert r is not None
    on_r = (np.asarray(arm) == r).astype(float)
    if stem == "ate":
        return on_a / g[:, a] - on_r / g[:, r]
    if stem == "att":
        return (on_a - on_r * g[:, a] / g[:, r]) / share
    if stem == "atc":
        return (on_a * g[:, r] / g[:, a] - on_r) / share
    raise ValueError(f"no representer is written for {estimand!r}")


def conditioning_arm(estimand: str) -> int | None:
    """The arm an ``att`` or ``atc`` conditions on, or ``None``.

    Parameters
    ----------
    estimand : str
        A name :func:`parse` reads.

    Returns
    -------
    int or None
        The conditioning arm.
    """
    stem, a, r = parse(estimand)
    return a if stem == "att" else r if stem == "atc" else None


def plugin_second_moment(
    probs: Any,
    design: Any,
    arm: Any,
    n_arms: int,
    estimand: str,
    *,
    weights: Any = None,
    penalty: Any = None,
    bounds: tuple[float, float] | None = None,
    mechanism: Any = None,
) -> Any:
    r"""The plug-in :math:`E_{P_w}[\alpha_b(g)^2]` of a finite law.

    Parameters
    ----------
    probs : ndarray
        ``(S,)`` cell probabilities; complex allowed.
    design : ndarray
        ``(S, d)`` design of each cell.
    arm : ndarray
        ``(S,)`` arm index of each cell.
    n_arms : int
        The number of arms.
    estimand : str
        A name :func:`parse` reads.
    weights : ndarray or None
        ``(S,)`` observation weight of each cell, or ``None`` for one.
    penalty : ndarray or None
        The :func:`penalty_matrix` of the projection.
    bounds : tuple of float or None
        The truncation bounds.
    mechanism : ndarray or None
        A fixed ``(S, K)`` mechanism, in place of the projection: the known mechanism
        :math:`g_0`, held fixed along the path.

    Returns
    -------
    Any
        The second moment; complex when ``probs`` is.
    """
    p = np.asarray(probs).reshape(-1)
    w = np.ones(p.shape[0]) if weights is None else np.asarray(weights, dtype=float)
    q = w * p / np.sum(w * p)
    if mechanism is None:
        theta = logistic_projection(q, design, arm, n_arms, penalty=penalty)
        g = softmax(design, theta)
    else:
        g = np.asarray(mechanism, dtype=float)
    bounded, _ = truncate(g, bounds)
    conditioning = conditioning_arm(estimand)
    share = None
    if conditioning is not None:
        share = np.sum(q * (np.asarray(arm) == conditioning))
    alpha = representer(estimand, arm, bounded, share)
    return np.sum(q * alpha**2)
