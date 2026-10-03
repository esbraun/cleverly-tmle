r"""Exact truths and efficient influence functions for the stratified alternating targets.

The laws of the stratified incremental, MSM and DR-TMLE studies, with every truth and every
efficient influence function computed from the law itself.

``L1`` is :mod:`tests.studies.stratified_law` unchanged: ``V`` in {0, 1, 2}, ``W`` in
{0, 1, 2} given ``V``, binary ``A`` and ``Y``, 36 support points.  On it, each parameter is an
analytic function of the 36 cell probabilities, and its efficient influence function is the
Gateaux derivative of that function along the contamination path, taken by complex step
(:func:`tests.discrete_law.contamination_eif`).  No clever covariate and no library code
enters these functions.

=====================================  ===========================================================
parameter                              functional of the law given ``V = s`` (or of the full law)
=====================================  ===========================================================
``ey_ipsi[name]``                      ``sum_w P(w) [q_d Q(1, w) + (1 - q_d) Q(0, w)]`` with
                                       ``q_d = d g / (d g + 1 - g)`` (Kennedy 2019, Equation (1))
``ate_ipsi[name vs natural course]``   the difference from the ``delta = 1`` mean
``msm[term]`` (identity, logit)        the ``h = 1`` least-squares projection of ``Q(a, w)`` on
                                       ``phi = (1, a, W)`` over both arms and the law of ``W``,
                                       on the link's scale with its ``dm/deta`` weight
=====================================  ===========================================================

The marginal incremental mean is the ``P(V = s)``-weighted mixture of the stratum means.  The
marginal MSM is the projection over the law of ``(W, V)``, with ``V`` outside the working model.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from tests import discrete_law
from tests.studies import stratified_law as base

#: The support and its probabilities, from :mod:`tests.studies.stratified_law`.
SUPPORT = base.SUPPORT
PROBS = base.PROBS
STRATA = base.STRATA

#: The three odds multipliers, by the name the package gives each.
DELTAS: dict[str, float] = {"natural course": 1.0, "odds x2": 2.0, "odds x0.5": 0.5}
REFERENCE = "natural course"

#: ``phi = (1, a, W)``, the L1 working model, and the links the studies fit it under.
MSM_TERMS = ("(intercept)", "a", "W")
MSM_LINKS = ("identity", "logit")


def _cells(probs: Any) -> tuple[Any, Any, Any]:
    """``P(V, W)``, ``g(w, v)`` and ``Q(a, w, v)`` from the 36 cell probabilities."""
    p = np.asarray(probs).reshape(3, 3, 2, 2)  # (v, w, a, y)
    p_vw = p.sum(axis=(2, 3))
    p_vwa = p.sum(axis=3)
    g = p_vwa[:, :, 1] / p_vw
    q = p[:, :, :, 1] / p_vwa
    return p_vw, g, q


def ipsi_mean(probs: Any, delta: float, stratum: int | None = None) -> Any:
    """The incremental mean at ``delta``, inside ``stratum`` or over the full law."""
    p_vw, g, q = _cells(probs)
    d = delta * g + 1.0 - g
    mixture = (delta * g * q[:, :, 1] + (1.0 - g) * q[:, :, 0]) / d
    if stratum is None:
        return (p_vw * mixture).sum() / p_vw.sum()
    weights = p_vw[stratum]
    return (weights * mixture[stratum]).sum() / weights.sum()


def msm_beta(probs: Any, link: str, stratum: int | None = None) -> Any:
    """The projection coefficients of ``Q`` on ``(1, a, W)``, inside a stratum or marginal.

    Newton with the exact Jacobian runs a fixed number of steps, so the function stays
    analytic in the probabilities, as :func:`tests.discrete_law.functional` keeps it.
    """
    inverse, slope, curvature = discrete_law.MSM_LINKS[link]
    p_vw, _, q = _cells(probs)
    w = np.broadcast_to(np.arange(3, dtype=float)[None, :, None], (3, 3, 2))
    a = np.broadcast_to(np.array([0.0, 1.0])[None, None, :], (3, 3, 2))
    phi = np.stack([np.ones((3, 3, 2)), a, w], axis=3)  # (v, w, a, term)
    mass = p_vw if stratum is None else np.where(np.arange(3)[:, None] == stratum, p_vw, 0.0)
    beta = np.zeros(len(MSM_TERMS), dtype=np.asarray(probs).dtype)
    for _ in range(discrete_law.MSM_NEWTON_STEPS):
        m = inverse(np.einsum("vwap,p->vwa", phi, beta))
        residual = q - m
        first, second = slope(m), curvature(m)
        score = np.einsum("vwap,vwa,vw->p", phi, first * residual, mass)
        jacobian = np.einsum("vwap,vwaq,vwa,vw->pq", phi, phi, first**2 - residual * second, mass)
        beta = beta + np.linalg.solve(jacobian, score)
    return beta


def _suffix(stratum: int | None) -> str:
    return "" if stratum is None else f"[V={stratum}]"


def ipsi_names(stratum: int | None = None) -> tuple[str, ...]:
    """The five incremental names, marginal or of one stratum."""
    means = tuple(f"ey_ipsi[{name}]{_suffix(stratum)}" for name in DELTAS)
    contrasts = tuple(
        f"ate_ipsi[{name} vs {REFERENCE}]{_suffix(stratum)}" for name in DELTAS if name != REFERENCE
    )
    return means + contrasts


def msm_names(stratum: int | None = None) -> tuple[str, ...]:
    """The three MSM coefficient names, marginal or of one stratum."""
    return tuple(f"msm[{term}]{_suffix(stratum)}" for term in MSM_TERMS)


def _ipsi_functional(name: str) -> Callable[[Any], Any]:
    stem, _, rest = name.partition("[")
    label, _, tail = rest.partition("]")
    stratum = int(tail[3:-1]) if tail.startswith("[V=") else None
    if stem == "ey_ipsi":
        return lambda p: ipsi_mean(p, DELTAS[label], stratum)
    left, right = label.split(" vs ")
    return lambda p: ipsi_mean(p, DELTAS[left], stratum) - ipsi_mean(p, DELTAS[right], stratum)


def _msm_functional(name: str, link: str) -> Callable[[Any], Any]:
    term, _, tail = name[len("msm[") :].partition("]")
    stratum = int(tail[3:-1]) if tail.startswith("[V=") else None
    index = MSM_TERMS.index(term)
    return lambda p: msm_beta(p, link, stratum)[index]


def functional(name: str, link: str | None = None) -> Callable[[Any], Any]:
    """The analytic functional of one parameter; ``link`` selects an MSM's link."""
    if name.startswith("msm["):
        if link is None:
            raise ValueError("an MSM name needs its link")
        return _msm_functional(name, link)
    return _ipsi_functional(name)


def truth(name: str, link: str | None = None) -> float:
    """The true value of one parameter."""
    return float(np.real(functional(name, link)(PROBS)))


def eif(name: str, link: str | None = None) -> np.ndarray:
    """The efficient influence function of one parameter at every support point."""
    return np.asarray(
        discrete_law.contamination_eif(
            functional(name, link), PROBS, [(index,) for index in range(len(SUPPORT))]
        ),
        dtype=float,
    )


def efficiency_sd(name: str, link: str | None = None) -> float:
    """The efficiency-bound SD, ``sqrt(E[D^2])``."""
    return float(np.sqrt(np.sum(PROBS * eif(name, link) ** 2)))


def covariance(names: tuple[str, ...], link: str | None = None) -> np.ndarray:
    """The exact covariance ``E[D_j D_k]`` of the selected curves."""
    curves = np.column_stack([eif(name, link) for name in names])
    return np.asarray((curves * PROBS[:, None]).T @ curves, dtype=float)


def all_ipsi_names() -> tuple[str, ...]:
    """Every incremental name the stratified fit reports: marginal, then stratum by stratum."""
    return ipsi_names() + tuple(name for s in STRATA for name in ipsi_names(s))


def all_msm_names() -> tuple[str, ...]:
    """Every MSM name the stratified fit reports: marginal, then stratum by stratum."""
    return msm_names() + tuple(name for s in STRATA for name in msm_names(s))


TRUTH_IPSI: dict[str, float] = {name: truth(name) for name in all_ipsi_names()}
TRUTH_MSM: dict[str, dict[str, float]] = {
    link: {name: truth(name, link) for name in all_msm_names()} for link in MSM_LINKS
}


def sample(n: int, seed: int) -> Any:
    """Draw ``n`` rows ``(V, W, A, Y)`` of L1."""
    return base.sample(n, seed)


def truths(link: str | None = None) -> Mapping[str, float]:
    """The incremental truths, or one link's MSM truths."""
    return dict(TRUTH_IPSI) if link is None else dict(TRUTH_MSM[link])


# ------------------------------------------------------------------------- L2: a dose

r"""``L2`` keeps the strata and ``W`` of L1 and gives the treatment a continuous dose.

=========  =====================================================================
piece      definition
=========  =====================================================================
``A``      ``N(0.5 W - 0.4 V, 1)`` given ``(W, V)``
``Yc``     ``1 + A + 0.5 W + 0.5 V + 0.5 A V + N(0, 1)``, for the identity link
``Yd``     binary, ``expit(-0.5 + 0.6 A + 0.4 W + 0.5 V - 0.3 A V)``, for the logit link
=========  =====================================================================

The working model is ``phi = (1, a)`` with ``h = 1`` over the declared dose grid
:data:`GRID`, integrated by the trapezoid rule the package uses
(:meth:`cleverly.msm.MSMSet.evaluate`).  Inside a stratum the identity truth is exact, because
``E[Yc | a, V = s]`` is linear in ``a``.  The logit truth is a genuine projection: averaging
``expit`` over ``W`` leaves a curve that is not logit-linear.
"""

#: The declared dose grid of the continuous working model.
GRID = (-1.25, -0.625, 0.0, 0.625, 1.25)


def quadrature(grid: tuple[float, ...] = GRID) -> np.ndarray:
    """The trapezoid weights :meth:`cleverly.msm.MSMSet.evaluate` puts on each dose."""
    doses = np.asarray(grid, dtype=float)
    weights = np.empty(doses.size)
    weights[0] = (doses[1] - doses[0]) / 2.0
    weights[-1] = (doses[-1] - doses[-2]) / 2.0
    weights[1:-1] = (doses[2:] - doses[:-2]) / 2.0
    return weights


def dose_mean(w: Any, v: Any) -> Any:
    """``E[A | W, V]``."""
    return 0.5 * np.asarray(w, dtype=float) - 0.4 * np.asarray(v, dtype=float)


def l2_outcome(link: str, a: Any, w: Any, v: Any) -> Any:
    """``E[Y | A, W, V]`` of the outcome the link reads."""
    a, w, v = (np.asarray(x, dtype=float) for x in (a, w, v))
    if link == "identity":
        return 1.0 + a + 0.5 * w + 0.5 * v + 0.5 * a * v
    return 1.0 / (1.0 + np.exp(-(-0.5 + 0.6 * a + 0.4 * w + 0.5 * v - 0.3 * a * v)))


def l2_sample(n: int, seed: int) -> Any:
    """Draw ``n`` rows ``(V, W, D, Yc, Yd)`` of L2."""
    import pandas as pd

    rng = np.random.default_rng(seed)
    v = rng.choice(3, size=n, p=base.P_V)
    w = np.array([rng.choice(3, p=base.P_W_GIVEN_V[level]) for level in v])
    dose = dose_mean(w, v) + rng.normal(size=n)
    yc = l2_outcome("identity", dose, w, v) + rng.normal(size=n)
    yd = (rng.random(n) < l2_outcome("logit", dose, w, v)).astype(int)
    return pd.DataFrame({"V": v, "W": w, "D": dose, "Yc": yc, "Yd": yd})


def _l2_cells(stratum: int | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The ``(w, v)`` cells of the law, and their probabilities given the scope."""
    v, w = np.meshgrid(np.arange(3.0), np.arange(3.0), indexing="ij")
    mass = base.P_V[:, None] * base.P_W_GIVEN_V
    if stratum is not None:
        mass = np.where(np.arange(3)[:, None] == stratum, mass, 0.0)
    mass = mass / mass.sum()
    return w.ravel(), v.ravel(), mass.ravel()


def _link(link: str) -> tuple[Any, Any, Any]:
    return discrete_law.MSM_LINKS[link]


def l2_beta(link: str, stratum: int | None = None) -> np.ndarray:
    """The grid projection of ``E[Y_d | V]`` on ``(1, d)``, inside a stratum or marginal."""
    inverse, slope, curvature = _link(link)
    w, v, mass = _l2_cells(stratum)
    doses = np.asarray(GRID)
    phi = np.column_stack([np.ones_like(doses), doses])
    weights = quadrature()
    q = np.array(
        [[l2_outcome(link, d, wi, vi) for d in doses] for wi, vi in zip(w, v, strict=True)]
    )
    beta = np.zeros(2)
    for _ in range(discrete_law.MSM_NEWTON_STEPS):
        m = inverse(phi @ beta)
        residual = q - m[None, :]
        first, second = slope(m), curvature(m)
        score = np.einsum("c,cj,jp,j->p", mass, residual * first[None, :], phi, weights)
        jacobian = np.einsum(
            "c,cj,jp,jq,j->pq",
            mass,
            first[None, :] ** 2 - residual * second[None, :],
            phi,
            phi,
            weights,
        )
        beta = beta + np.linalg.solve(jacobian, score)
    return np.asarray(beta, dtype=float)


def l2_efficiency_sd(link: str, stratum: int | None = None, *, points: int = 4001) -> np.ndarray:
    """The SD of the influence curve the package reports for each coefficient, at the law.

    The curve is ``M^-1 [h 1{A in range} phi(A) m'(A) / g(A | W) {Y - Q(A, W)} +
    sum_j w_j phi_j m'_j {Q(d_j, W) - m_j}]``, divided by ``P(V = s)`` on the stratum's rows
    inside a stratum.  ``M`` is the projection Jacobian with its curvature term.  The two terms
    are uncorrelated, so the variance is the sum of an integral over the dose range and a sum
    over the ``(W, V)`` cells.  The integral runs on ``points`` trapezoid nodes.
    """
    inverse, slope, curvature = _link(link)
    w, v, mass = _l2_cells(stratum)
    beta = l2_beta(link, stratum)
    doses = np.asarray(GRID)
    phi = np.column_stack([np.ones_like(doses), doses])
    weights = quadrature()
    m = inverse(phi @ beta)
    first, second = slope(m), curvature(m)
    q = np.array(
        [[l2_outcome(link, d, wi, vi) for d in doses] for wi, vi in zip(w, v, strict=True)]
    )
    residual = q - m[None, :]
    jacobian = np.einsum(
        "c,cj,jp,jq,j->pq",
        mass,
        first[None, :] ** 2 - residual * second[None, :],
        phi,
        phi,
        weights,
    )
    inverse_jacobian = np.linalg.inv(jacobian)
    # The plug-in term, one vector per cell.
    plug = np.einsum("cj,jp,j->cp", residual * first[None, :], phi, weights)
    plug_variance = np.einsum("c,cp,cq->pq", mass, plug, plug)
    # The residual term: an integral over the dose range of phi phi' m'^2 Var(Y | a, W) / g.
    grid = np.linspace(doses[0], doses[-1], points)
    step = grid[1] - grid[0]
    trapezoid = np.full(points, step)
    trapezoid[[0, -1]] = step / 2.0
    phi_a = np.column_stack([np.ones_like(grid), grid])
    m_a = inverse(phi_a @ beta)
    first_a = slope(m_a)
    residual_variance = np.zeros((2, 2))
    for wi, vi, cell in zip(w, v, mass, strict=True):
        if cell == 0.0:
            continue
        density = np.exp(-0.5 * (grid - dose_mean(wi, vi)) ** 2) / np.sqrt(2.0 * np.pi)
        mean = l2_outcome(link, grid, wi, vi)
        variance = np.ones_like(grid) if link == "identity" else mean * (1.0 - mean)
        integrand = (first_a**2) * variance / density
        residual_variance += cell * np.einsum("a,ap,aq->pq", trapezoid * integrand, phi_a, phi_a)
    covariance = inverse_jacobian @ (plug_variance + residual_variance) @ inverse_jacobian.T
    scale = 1.0 if stratum is None else 1.0 / float(base.P_V[stratum])
    return np.sqrt(np.diag(covariance) * scale)


def l2_names(stratum: int | None = None) -> tuple[str, ...]:
    """The two coefficient names of the continuous working model."""
    return tuple(f"msm[{term}]{_suffix(stratum)}" for term in ("(intercept)", "a"))


def l2_truths(link: str) -> dict[str, float]:
    """Every marginal and stratum coefficient of one link."""
    out: dict[str, float] = {}
    for stratum in (None, *STRATA):
        out.update(zip(l2_names(stratum), l2_beta(link, stratum).tolist(), strict=True))
    return out


# ---------------------------------------------------------- L1 with a bounded outcome

#: The beta concentration of the bounded outcome of the identity-MSM pairing, as in
#: :func:`tests.studies.canonical_point_msm.draw_from_seed`.
CONCENTRATION = 24.0


def beta_sample(n: int, seed: int) -> Any:
    """Draw ``n`` rows of L1 with ``Y ~ Beta(24 Q, 24 (1 - Q))`` in place of the binary ``Y``.

    ``E[Y | A, W, V]`` is still ``Q``, so every MSM truth is L1's.
    """
    import pandas as pd

    rng = np.random.default_rng(seed)
    cells = rng.choice(len(SUPPORT), size=n, p=PROBS / PROBS.sum())
    support = np.asarray(SUPPORT, dtype=float)[cells]
    v, w, a = support[:, 0], support[:, 1], support[:, 2]
    mean = np.asarray(base.outcome(a, w, v))
    y = rng.beta(mean * CONCENTRATION, (1.0 - mean) * CONCENTRATION)
    return pd.DataFrame({"V": v.astype(int), "W": w.astype(int), "A": a.astype(int), "Y": y})


def beta_efficiency_sd(name: str, link: str = "identity") -> float:
    """The efficiency-bound SD of an MSM coefficient under the bounded-outcome law.

    On L1 the curve is ``c(A, W, V) (Y - Q) + d(A, W, V)`` at each support point, with
    ``c = D(Y = 1) - D(Y = 0)``.  The bounded law keeps ``Q`` and replaces the outcome
    variance ``Q (1 - Q)`` by ``Q (1 - Q) / (CONCENTRATION + 1)``.
    """
    curve = eif(name, link).reshape(-1, 2)  # (v, w, a) cells, then y = 0, 1
    mass = PROBS.reshape(-1, 2).sum(axis=1)
    q = PROBS.reshape(-1, 2)[:, 1] / mass
    slope = curve[:, 1] - curve[:, 0]
    centre = curve[:, 0] + slope * q
    variance = q * (1.0 - q) / (CONCENTRATION + 1.0)
    return float(np.sqrt(np.sum(mass * (slope**2 * variance + centre**2))))


# ---------------------------------------------------------- the stratified paper law

r"""The stratified paper law of the DR-TMLE study.

The complete-data binary law of Benkeser et al. (2017), Section 5.1, as
:mod:`tests.studies.canonical_drtmle` draws it, with a baseline stratum ``V`` that shifts the
intercepts of both regressions:

=========  =====================================================================
piece      definition
=========  =====================================================================
``V``      0, 1 or 2 with probabilities 0.5, 0.3 and 0.2, independent of ``W``
``W``      ``W1 ~ U(-2, 2)``, ``W2 ~ Bernoulli(0.5)``, ``W12 = W1 W2``
``A``      ``expit(-W1 + 2 W1 W2 + 0.4 - 0.4 V)``
``Y``      ``expit((0.2 + 0.3 V) A - W1 + 2 W1 W2 - 0.3 + 0.4 V)``
=========  =====================================================================

A logistic model on ``(A, W1, W2, W12, V, A V)`` is correct for ``Y`` and one on
``(W1, W2, W12, V)`` for ``A``; dropping ``W12`` misspecifies either, as the shipped study does.
Every truth is a one-dimensional integral over ``W1``, by the shipped quadrature.
"""

PAPER_P_V = (0.5, 0.3, 0.2)


def paper_linear(w1: Any, w2: Any) -> Any:
    return -np.asarray(w1, dtype=float) + 2.0 * np.asarray(w1, dtype=float) * np.asarray(
        w2, dtype=float
    )


def paper_propensity(w1: Any, w2: Any, v: Any) -> Any:
    return 1.0 / (1.0 + np.exp(-(paper_linear(w1, w2) + 0.4 - 0.4 * np.asarray(v, dtype=float))))


def paper_outcome(a: Any, w1: Any, w2: Any, v: Any) -> Any:
    a, v = np.asarray(a, dtype=float), np.asarray(v, dtype=float)
    eta = (0.2 + 0.3 * v) * a + paper_linear(w1, w2) - 0.3 + 0.4 * v
    return 1.0 / (1.0 + np.exp(-eta))


def paper_sample(n: int, seed: int) -> Any:
    """Draw ``n`` rows ``(Y, A, W1, W2, W12, V)`` of the stratified paper law."""
    import pandas as pd

    rng = np.random.default_rng(seed)
    v = rng.choice(3, size=n, p=PAPER_P_V)
    w1 = rng.uniform(-2.0, 2.0, size=n)
    w2 = rng.binomial(1, 0.5, size=n).astype(float)
    a = rng.binomial(1, paper_propensity(w1, w2, v)).astype(float)
    y = rng.binomial(1, paper_outcome(a, w1, w2, v)).astype(float)
    return pd.DataFrame({"Y": y, "A": a, "W1": w1, "W2": w2, "W12": w1 * w2, "V": v})


def _paper_arm_mean(arm: float, stratum: int) -> float:
    from scipy.integrate import quad

    def integrand(w1: float) -> float:
        return 0.5 * float(
            paper_outcome(arm, w1, 0.0, stratum) + paper_outcome(arm, w1, 1.0, stratum)
        )

    value, error = quad(integrand, -2.0, 2.0, epsabs=1e-13, epsrel=1e-13, limit=200)
    if error > 1e-11:  # pragma: no cover - a declaration guard
        raise RuntimeError(f"paper-law quadrature error {error:g} exceeds its audit bar")
    return value / 4.0


def paper_truths() -> dict[str, float]:
    """Every marginal and stratum arm mean and ATE of the stratified paper law."""
    return dict(_paper_truths())


@functools.cache
def _paper_truths() -> dict[str, float]:
    out: dict[str, float] = {}
    marginal = {0.0: 0.0, 1.0: 0.0}
    for stratum, mass in zip(STRATA, PAPER_P_V, strict=True):
        means = {arm: _paper_arm_mean(arm, stratum) for arm in (0.0, 1.0)}
        out[f"ey[0][V={stratum}]"] = means[0.0]
        out[f"ey[1][V={stratum}]"] = means[1.0]
        out[f"ate[V={stratum}]"] = means[1.0] - means[0.0]
        for arm in marginal:
            marginal[arm] += mass * means[arm]
    out["ey[0]"] = marginal[0.0]
    out["ey[1]"] = marginal[1.0]
    out["ate"] = marginal[1.0] - marginal[0.0]
    return out
