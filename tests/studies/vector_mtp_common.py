"""Laws, truths, plans and learners of the ``vector-mtp`` study.

Four laws, each with its truth and efficiency bound computed before any run.  Every dose is a
normal truncated to ``[0, 6]`` with unit scale, every policy keeps the dose inside that range
(each map is capped at 5.5), and every flag probability lies in ``[0.25, 0.75]``, so every
density ratio is bounded, the condition of Díaz, Williams, Hoffman and Schenck (2023),
Theorem 3.

**The point law** (``P``): a flag and a dose.

.. code-block:: text

    W1 ~ Uniform(-1, 1),  W2 ~ Bernoulli(1/2)
    F | W ~ Bernoulli(expit(0.1 + 0.6 W1 - 0.4 W2))
    D | F, W ~ N(2.4 + 0.6 F + 0.4 W1 + 0.3 W2, 1) truncated to [0, 6]
    Y | F, D, W ~ Bernoulli(expit(-1.8 + 0.5 F + 0.5 D - 0.05 D^2 + 0.4 W1 + 0.3 W2))

**The two-node law** (``L``): a flag and a dose at each node.

.. code-block:: text

    W ~ Bernoulli(1/2)
    F1 | W ~ Bernoulli(expit(-0.1 + 0.4 W))
    D1 | F1, W ~ N(2.4 + 0.6 F1 + 0.4 W, 1) truncated to [0, 6]
    L2 | W, F1, D1 ~ Bernoulli(expit(-1 + 0.3 D1 + 0.3 F1 - 0.4 W))
    F2 | W, F1, D1, L2 ~ Bernoulli(expit(-0.5 + 0.8 F1 + 0.4 L2 - 0.1 D1))
    D2 | W, F1, D1, L2, F2 ~ N(1 + 0.4 D1 + 0.5 F2 + 0.4 L2 - 0.2 W, 1) truncated to [0, 6]
    Y | ... ~ Bernoulli(expit(-2.2 + 0.3 F1 + 0.2 D1 + 0.4 F2 + 0.45 D2 - 0.03 D2^2
                                + 0.5 L2 + 0.3 W))

**The ordered law** (``O``): a flag and two doses at one node, the second dose's law reading the
first.  It is read as a point treatment by ``TMLE`` and as one mixed node by ``LTMLE``, and both
readings estimate the same policy mean.

.. code-block:: text

    W1, W2, F as in the point law
    D1 | F, W ~ N(2.4 + 0.6 F + 0.4 W1 + 0.3 W2, 1) truncated to [0, 6]
    D2 | D1, F, W ~ N(1 + 0.4 D1 + 0.5 F - 0.3 W1, 1) truncated to [0, 6]
    Y | ... ~ Bernoulli(expit(-2 + 0.4 F + 0.3 D1 - 0.03 D1^2 + 0.35 D2 - 0.03 D2^2
                                + 0.4 W1 + 0.3 W2))

**The categorical law** (``C``): two binary components, for the all-categorical vector.

.. code-block:: text

    W1, W2 as in the point law
    A1 | W ~ Bernoulli(expit(-0.2 + 0.5 W1 + 0.3 W2))
    A2 | A1, W ~ Bernoulli(expit(-0.3 + 0.8 A1 - 0.4 W1 + 0.2 W2))
    Y | A, W ~ Bernoulli(expit(-1 + 0.6 A1 + 0.5 A2 - 0.7 A1 A2 + 0.4 W1 + 0.3 W2))

**Plans.**  Each plan is a :class:`~cleverly.interventions.JointPolicy` and, beside it, the same
map written in numpy, which the truths and the closed-form ratios read.

==================  ====================================================================
plan                map
==================  ====================================================================
``natural``         every component keeps its natural value
``dose up``         ``Shift(0.5, cap=5.5)`` on the dose (``P`` and both nodes of ``L``)
``flag scaled``     the flag goes to 1, and the dose is multiplied by 1.25 (cap 5.5) where
                    the **natural** flag is 1 (``P`` and both nodes of ``L``): a flag move,
                    a many-to-one categorical preimage and a non-unit Jacobian
``ordered``         ``Scale(1.25, cap=5.5)`` on ``D1``, and ``D2`` up by 0.5 (cap 5.5) where
                    the natural ``D1`` is at most 3 (``O``): an ordered read
``full``            the flag goes to 1 where the natural ``D1`` is at least 3, ``D1`` is
                    multiplied by 1.25 (cap 5.5) where the natural flag is 1, and ``D2`` as in
                    ``ordered`` (``O``): every term of the induced density at once
``a2 off``          ``A2`` goes to 0 where ``A1`` is 1 (``C``)
``a1 on``           ``A1`` goes to 1 (``C``), many-to-one
==================  ====================================================================

**Truths** are the g-formula of Díaz et al. (2023), Theorem 1, with each node's policy reading
the intervened earlier values, by composite Gauss-Legendre quadrature split at every point
where a policy jumps, and over ``W1`` by Gauss-Legendre on ``[-1, 1]``.  Each truth is checked
by doubling every order (agreement below ``1e-10``).

**Efficiency bounds** are the standard deviation of the closed-form efficient influence
function over :data:`BOUND_DRAWS` draws.  Its nuisances are the law's own: the outcome
regression, the first-node regression by quadrature, and the induced density ratio of each
node, a sum over the categorical preimages and the dose pieces of the preimage density times
the inverse derivatives (Díaz et al. 2023, Equation (3), per conditional factor).  The fast
witnesses of ``tests/unit/test_vector_density_route.py`` check the same formula, as the
package evaluates it, against complex-step Gateaux derivatives on coarse laws.

**Oracle learners.**  :class:`OracleFlag` returns the exact flag probability, the categorical
factor :math:`g_C`.  :class:`OracleDoseHazard` returns the exact pooled-hazard bin
probabilities of each dose given its conditioning design (``bin_design = "index"``).
:class:`OracleLogOdds` is the correctly specified classifier of the stacked ratio route, a
logistic regression on the true log ratio.  Each dispatches on its design's width, which the
module docstring of each class lists per law.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from functools import cache, partial
from typing import Any, ClassVar

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.interventions import JointPolicy, ModifiedPolicy, Piece, Scale, Shift
from cleverly.longitudinal import DynamicRegimen
from tests.studies.fractional_glm import QuasiBinomialGLM

LOWER, UPPER = 0.0, 6.0
CAP = 5.5
STEP = 0.5
FACTOR = 1.25
#: ``D2`` moves, and the flag goes to 1 under ``full``, on either side of this natural ``D1``.
KNEE = 3.0
#: A shifted dose is held at its natural value above this.
SHIFT_HELD = CAP - STEP
#: A scaled dose is held at its natural value above this.
SCALE_HELD = CAP / FACTOR

POINT_LABELS = ("natural", "dose up", "flag scaled")
LONG_LABELS = ("natural", "dose up", "flag scaled")
ORDERED_LABELS = ("natural", "ordered", "full")
CATEGORICAL_LABELS = ("natural", "a2 off", "a1 on")

#: Gauss-Legendre order per segment of a dose integral, and over ``W1``.
ORDER = 96
W_ORDER = 24
#: The ordered law's double integral converges at far lower orders (agreement to ``1e-15`` at
#: 32 per segment and 16 over ``W1``), so it uses these, checked at twice each.
ORDERED_ORDER = 48
ORDERED_W_ORDER = 16
#: The order per segment of the first-node regression inside the influence function.  The
#: integrand is smooth on each segment, and 16 agrees with 96 to ``1e-15`` on 5,000 rows.
EIF_ORDER = 16
BOUND_DRAWS = 400_000
BOUND_SEED = 20263109


# ---------------------------------------------------------------------------- shared pieces


def truncated_pdf(a: Any, mean: Any) -> Any:
    """The unit-scale normal density truncated to ``[LOWER, UPPER]``."""
    a = np.asarray(a, dtype=float)
    mean = np.asarray(mean, dtype=float)
    mass = norm.cdf(UPPER - mean) - norm.cdf(LOWER - mean)
    inside = (a >= LOWER) & (a <= UPPER)
    return np.where(inside, norm.pdf(a - mean) / mass, 0.0)


def truncated_draw(rng: np.random.Generator, mean: Any) -> np.ndarray:
    """One truncated-normal draw per mean, by the inverse distribution function."""
    mean = np.asarray(mean, dtype=float)
    u = rng.uniform(norm.cdf(LOWER - mean), norm.cdf(UPPER - mean))
    return np.clip(mean + norm.ppf(u), LOWER, UPPER)


def _bernoulli(p: Any, value: Any) -> Any:
    p = np.asarray(p, dtype=float)
    return np.where(np.asarray(value, dtype=float) == 1.0, p, 1.0 - p)


def shift_up(a: Any) -> np.ndarray:
    """``Shift(0.5, cap=5.5)``."""
    a = np.asarray(a, dtype=float)
    return np.where(a + STEP > CAP, a, a + STEP)


def scale_up(a: Any) -> np.ndarray:
    """``Scale(1.25, cap=5.5)``."""
    a = np.asarray(a, dtype=float)
    return np.where(a * FACTOR > CAP, a, a * FACTOR)


def _segments(breaks: tuple[float, ...], order: int) -> tuple[np.ndarray, np.ndarray]:
    """Composite Gauss-Legendre nodes and weights on ``[LOWER, UPPER]``, split at ``breaks``."""
    points = sorted({LOWER, UPPER, *(b for b in breaks if LOWER < b < UPPER)})
    base, weights = np.polynomial.legendre.leggauss(order)
    nodes, scaled = [], []
    for low, high in itertools.pairwise(points):
        half = 0.5 * (high - low)
        nodes.append(low + half * (base + 1.0))
        scaled.append(half * weights)
    return np.concatenate(nodes), np.concatenate(scaled)


def _w1_nodes(order: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre on ``[-1, 1]`` with the uniform density ``1/2`` folded in."""
    nodes, weights = np.polynomial.legendre.leggauss(order)
    return nodes, 0.5 * weights


#: Every dose at which a policy's map jumps.
DOSE_BREAKS = (SCALE_HELD, SHIFT_HELD, KNEE)


# ---------------------------------------------------------------------------- the point law


def point_flag(w1: Any, w2: Any) -> Any:
    """``P(F = 1 | W)``, shared by the point and ordered laws."""
    return expit(0.1 + 0.6 * np.asarray(w1, dtype=float) - 0.4 * np.asarray(w2, dtype=float))


def point_dose_mean(f: Any, w1: Any, w2: Any) -> Any:
    """The mean of ``D`` (point law) and of ``D1`` (ordered law)."""
    return (
        2.4
        + 0.6 * np.asarray(f, dtype=float)
        + 0.4 * np.asarray(w1, dtype=float)
        + 0.3 * np.asarray(w2, dtype=float)
    )


def point_outcome(f: Any, d: Any, w1: Any, w2: Any) -> Any:
    d = np.asarray(d, dtype=float)
    return expit(
        -1.8
        + 0.5 * np.asarray(f, dtype=float)
        + 0.5 * d
        - 0.05 * d**2
        + 0.4 * np.asarray(w1, dtype=float)
        + 0.3 * np.asarray(w2, dtype=float)
    )


def sample_point(n: int, seed: int) -> pd.DataFrame:
    """``n`` rows of the point law: ``W1, W2, F, D, Y``."""
    rng = np.random.default_rng(seed)
    w1 = rng.uniform(-1.0, 1.0, n)
    w2 = rng.integers(0, 2, n).astype(float)
    f = rng.binomial(1, point_flag(w1, w2)).astype(float)
    d = truncated_draw(rng, point_dose_mean(f, w1, w2))
    y = rng.binomial(1, point_outcome(f, d, w1, w2)).astype(float)
    return pd.DataFrame({"W1": w1, "W2": w2, "F": f, "D": d, "Y": y})


def flag_dose_map(label: str, f: Any, d: Any) -> tuple[np.ndarray, np.ndarray]:
    """One flag-and-dose node's plan in numpy: ``(F', D')`` from the natural ``(F, D)``."""
    f = np.asarray(f, dtype=float)
    d = np.asarray(d, dtype=float)
    if label == "natural":
        return f, d
    if label == "dose up":
        return f, shift_up(d)
    if label == "flag scaled":
        return np.ones_like(f), np.where(f == 1.0, scale_up(d), d)
    raise KeyError(label)


def flag_dose_ratio(
    label: str,
    f: Any,
    b: Any,
    flag: Any,
    mean_of: Callable[[float], Any],
    *,
    cut: bool = False,
) -> np.ndarray:
    """The induced density ratio of a flag-and-dose node at the observed ``(f, b)``.

    ``flag`` is ``P(F = 1 | h)`` and ``mean_of(c)`` the dose mean under flag ``c``.  Under
    ``flag scaled`` the value ``(1, b)`` has three preimages: ``(0, b)`` (the flag moves and the
    dose stays), ``(1, b / 1.25)`` with Jacobian ``1 / 1.25`` while ``b <= 5.5``, and ``(1, b)``
    held by the cap where ``b > 4.4``.  No value with flag 0 is reached.  ``cut=True`` drops
    every preimage whose natural flag differs from the observed one, the control of
    ``preimage_necessity``.
    """
    f = np.asarray(f, dtype=float)
    b = np.asarray(b, dtype=float)

    def g(c: float, x: Any) -> Any:
        return _bernoulli(flag, c) * truncated_pdf(x, mean_of(c))

    observed = np.where(f == 1.0, g(1.0, b), g(0.0, b))
    safe = np.where(observed > 0.0, observed, 1.0)
    if label == "natural":
        return np.ones_like(b)
    if label == "dose up":
        moved = np.where(
            f == 1.0,
            g(1.0, b - STEP) * (b <= CAP) + g(1.0, b) * (b > SHIFT_HELD),
            g(0.0, b - STEP) * (b <= CAP) + g(0.0, b) * (b > SHIFT_HELD),
        )
        return np.where(observed > 0.0, moved / safe, 0.0)
    if label == "flag scaled":
        reached = (
            (0.0 if cut else g(0.0, b))
            + g(1.0, b / FACTOR) / FACTOR * (b <= CAP)
            + g(1.0, b) * (b > SCALE_HELD)
        )
        return np.where((f == 1.0) & (observed > 0.0), reached / safe, 0.0)
    raise KeyError(label)


def _point_mean(label: str, order: int, w_order: int) -> float:
    nodes, weights = _segments(DOSE_BREAKS, order)
    w1s, w1_weights = _w1_nodes(w_order)
    total = 0.0
    for w2 in (0.0, 1.0):
        for f in (0.0, 1.0):
            flag = _bernoulli(point_flag(w1s, w2), f)[:, None]
            density = truncated_pdf(nodes[None, :], point_dose_mean(f, w1s[:, None], w2))
            f_d, d_d = flag_dose_map(label, np.full(nodes.shape, f), nodes)
            q = point_outcome(f_d[None, :], d_d[None, :], w1s[:, None], w2)
            inner = np.sum(weights[None, :] * flag * density * q, axis=1)
            total += 0.5 * float(np.sum(w1_weights * inner))
    return total


@cache
def point_truth(label: str) -> float:
    """``E[Y]`` under a point plan, checked against twice every order."""
    value = _point_mean(label, ORDER, W_ORDER)
    check = _point_mean(label, 2 * ORDER, 2 * W_ORDER)
    if abs(value - check) > 1e-10:
        raise AssertionError(f"the quadrature of {label!r} moved by {abs(value - check):.3g}")
    return float(value)


def point_eif(label: str, frame: pd.DataFrame) -> np.ndarray:
    """The efficient influence function of ``E[Y]`` under a point plan, at every row."""
    w1, w2, f, d, y = (frame[c].to_numpy(dtype=float) for c in ("W1", "W2", "F", "D", "Y"))
    ratio = flag_dose_ratio(label, f, d, point_flag(w1, w2), lambda c: point_dose_mean(c, w1, w2))
    f_d, d_d = flag_dose_map(label, f, d)
    q = point_outcome(f, d, w1, w2)
    return ratio * (y - q) + point_outcome(f_d, d_d, w1, w2) - point_truth(label)


# ---------------------------------------------------------------------------- the ordered law


def ordered_d2_mean(d1: Any, f: Any, w1: Any) -> Any:
    return (
        1.0
        + 0.4 * np.asarray(d1, dtype=float)
        + 0.5 * np.asarray(f, dtype=float)
        - 0.3 * np.asarray(w1, dtype=float)
    )


def ordered_outcome(f: Any, d1: Any, d2: Any, w1: Any, w2: Any) -> Any:
    d1 = np.asarray(d1, dtype=float)
    d2 = np.asarray(d2, dtype=float)
    return expit(
        -2.0
        + 0.4 * np.asarray(f, dtype=float)
        + 0.3 * d1
        - 0.03 * d1**2
        + 0.35 * d2
        - 0.03 * d2**2
        + 0.4 * np.asarray(w1, dtype=float)
        + 0.3 * np.asarray(w2, dtype=float)
    )


def sample_ordered(n: int, seed: int) -> pd.DataFrame:
    """``n`` rows of the ordered law: ``W1, W2, F, D1, D2, Y``."""
    rng = np.random.default_rng(seed)
    w1 = rng.uniform(-1.0, 1.0, n)
    w2 = rng.integers(0, 2, n).astype(float)
    f = rng.binomial(1, point_flag(w1, w2)).astype(float)
    d1 = truncated_draw(rng, point_dose_mean(f, w1, w2))
    d2 = truncated_draw(rng, ordered_d2_mean(d1, f, w1))
    y = rng.binomial(1, ordered_outcome(f, d1, d2, w1, w2)).astype(float)
    return pd.DataFrame({"W1": w1, "W2": w2, "F": f, "D1": d1, "D2": d2, "Y": y})


def _d2_up(d2: Any, d1: Any) -> np.ndarray:
    """``D2`` up by 0.5 (cap 5.5) where the natural ``D1`` is at most :data:`KNEE`."""
    d2 = np.asarray(d2, dtype=float)
    return np.where(np.asarray(d1, dtype=float) <= KNEE, shift_up(d2), d2)


def ordered_map(label: str, f: Any, d1: Any, d2: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """An ordered-law plan in numpy: ``(F', D1', D2')`` from the natural values."""
    f = np.asarray(f, dtype=float)
    d1 = np.asarray(d1, dtype=float)
    d2 = np.asarray(d2, dtype=float)
    if label == "natural":
        return f, d1, d2
    if label == "ordered":
        return f, scale_up(d1), _d2_up(d2, d1)
    if label == "full":
        flag = np.where(d1 >= KNEE, 1.0, f)
        return flag, np.where(f == 1.0, scale_up(d1), d1), _d2_up(d2, d1)
    raise KeyError(label)


def ordered_ratio(label: str, f: Any, b1: Any, b2: Any, w1: Any, w2: Any) -> np.ndarray:
    """The induced density ratio of an ordered-law plan at the observed ``(f, b1, b2)``.

    A sum over the candidate natural flag ``c``, the pieces of the ``D1`` map in cell ``c``,
    and the pieces of the ``D2`` map at the preimage ``beta1`` of ``D1``, of the preimage
    density times the inverse derivatives, kept where the flag map sends ``(c, beta1)`` to the
    observed flag.
    """
    f = np.asarray(f, dtype=float)
    b1 = np.asarray(b1, dtype=float)
    b2 = np.asarray(b2, dtype=float)
    flag = point_flag(w1, w2)

    def g(c: float, x1: Any, x2: Any) -> Any:
        return (
            _bernoulli(flag, c)
            * truncated_pdf(x1, point_dose_mean(c, w1, w2))
            * truncated_pdf(x2, ordered_d2_mean(x1, c, w1))
        )

    observed = np.where(f == 1.0, g(1.0, b1, b2), g(0.0, b1, b2))
    if label == "natural":
        return np.ones_like(b1)
    reached = np.zeros_like(b1)
    for c in (0.0, 1.0):
        scaled = label == "ordered" or (label == "full" and c == 1.0)
        pieces1 = (
            ((b1 / FACTOR, 1.0 / FACTOR, b1 <= CAP), (b1, 1.0, b1 > SCALE_HELD))
            if scaled
            else ((b1, 1.0, np.ones_like(b1, dtype=bool)),)
        )
        for beta1, jacobian, valid1 in pieces1:
            image = np.where(beta1 >= KNEE, 1.0, c) if label == "full" else np.full_like(b1, c)
            hit = valid1 & (image == f)
            small = beta1 <= KNEE
            pieces2 = (
                (b2 - STEP, small & (b2 <= CAP)),
                (b2, small & (b2 > SHIFT_HELD)),
                (b2, ~small),
            )
            for beta2, valid2 in pieces2:
                reached = reached + np.where(hit & valid2, g(c, beta1, beta2) * jacobian, 0.0)
    safe = np.where(observed > 0.0, observed, 1.0)
    return np.where(observed > 0.0, reached / safe, 0.0)


def _ordered_mean(label: str, order: int, w_order: int) -> float:
    nodes, weights = _segments(DOSE_BREAKS, order)
    w1s, w1_weights = _w1_nodes(w_order)
    d1 = nodes[:, None]
    d2 = nodes[None, :]
    area = weights[:, None] * weights[None, :]
    total = 0.0
    for w2 in (0.0, 1.0):
        for f in (0.0, 1.0):
            f_d, d1_d, d2_d = ordered_map(label, np.full(area.shape, f), d1 + 0 * d2, d2 + 0 * d1)
            for w1, w1_weight in zip(w1s, w1_weights, strict=True):
                density = (
                    _bernoulli(point_flag(w1, w2), f)
                    * truncated_pdf(d1, point_dose_mean(f, w1, w2))
                    * truncated_pdf(d2, ordered_d2_mean(d1, f, w1))
                )
                q = ordered_outcome(f_d, d1_d, d2_d, w1, w2)
                total += 0.5 * w1_weight * float(np.sum(area * density * q))
    return total


@cache
def ordered_truth(label: str) -> float:
    """``E[Y]`` under an ordered-law plan, checked against twice every order."""
    value = _ordered_mean(label, ORDERED_ORDER, ORDERED_W_ORDER)
    check = _ordered_mean(label, 2 * ORDERED_ORDER, 2 * ORDERED_W_ORDER)
    if abs(value - check) > 1e-10:
        raise AssertionError(f"the quadrature of {label!r} moved by {abs(value - check):.3g}")
    return float(value)


def ordered_eif(label: str, frame: pd.DataFrame) -> np.ndarray:
    """The efficient influence function of ``E[Y]`` under an ordered-law plan, at every row."""
    w1, w2, f, d1, d2, y = (
        frame[c].to_numpy(dtype=float) for c in ("W1", "W2", "F", "D1", "D2", "Y")
    )
    ratio = ordered_ratio(label, f, d1, d2, w1, w2)
    f_d, d1_d, d2_d = ordered_map(label, f, d1, d2)
    q = ordered_outcome(f, d1, d2, w1, w2)
    return ratio * (y - q) + ordered_outcome(f_d, d1_d, d2_d, w1, w2) - ordered_truth(label)


# ---------------------------------------------------------------------------- the two-node law


def long_flag1(w: Any) -> Any:
    return expit(-0.1 + 0.4 * np.asarray(w, dtype=float))


def long_dose1_mean(f1: Any, w: Any) -> Any:
    return 2.4 + 0.6 * np.asarray(f1, dtype=float) + 0.4 * np.asarray(w, dtype=float)


def long_l2(w: Any, f1: Any, d1: Any) -> Any:
    return expit(
        -1.0
        + 0.3 * np.asarray(d1, dtype=float)
        + 0.3 * np.asarray(f1, dtype=float)
        - 0.4 * np.asarray(w, dtype=float)
    )


def long_flag2(w: Any, f1: Any, d1: Any, l2: Any) -> Any:
    del w
    return expit(
        -0.5
        + 0.8 * np.asarray(f1, dtype=float)
        + 0.4 * np.asarray(l2, dtype=float)
        - 0.1 * np.asarray(d1, dtype=float)
    )


def long_dose2_mean(w: Any, f1: Any, d1: Any, l2: Any, f2: Any) -> Any:
    del f1
    return (
        1.0
        + 0.4 * np.asarray(d1, dtype=float)
        + 0.5 * np.asarray(f2, dtype=float)
        + 0.4 * np.asarray(l2, dtype=float)
        - 0.2 * np.asarray(w, dtype=float)
    )


def long_outcome(w: Any, f1: Any, d1: Any, l2: Any, f2: Any, d2: Any) -> Any:
    d2 = np.asarray(d2, dtype=float)
    return expit(
        -2.2
        + 0.3 * np.asarray(f1, dtype=float)
        + 0.2 * np.asarray(d1, dtype=float)
        + 0.4 * np.asarray(f2, dtype=float)
        + 0.45 * d2
        - 0.03 * d2**2
        + 0.5 * np.asarray(l2, dtype=float)
        + 0.3 * np.asarray(w, dtype=float)
    )


def sample_long(n: int, seed: int) -> pd.DataFrame:
    """``n`` rows of the two-node law: ``W, F1, D1, L2, F2, D2, Y``."""
    rng = np.random.default_rng(seed)
    w = rng.integers(0, 2, n).astype(float)
    f1 = rng.binomial(1, long_flag1(w)).astype(float)
    d1 = truncated_draw(rng, long_dose1_mean(f1, w))
    l2 = rng.binomial(1, long_l2(w, f1, d1)).astype(float)
    f2 = rng.binomial(1, long_flag2(w, f1, d1, l2)).astype(float)
    d2 = truncated_draw(rng, long_dose2_mean(w, f1, d1, l2, f2))
    y = rng.binomial(1, long_outcome(w, f1, d1, l2, f2, d2)).astype(float)
    return pd.DataFrame({"W": w, "F1": f1, "D1": d1, "L2": l2, "F2": f2, "D2": d2, "Y": y})


def long_first_regression(label: str, w: Any, f1: Any, d1: Any, order: int = ORDER) -> np.ndarray:
    """``m_1(w, a_1) = E[m_2(H_2, d_2(A_2, H_2)) | A_1 = a_1, W = w]`` by quadrature, per row."""
    w = np.atleast_1d(np.asarray(w, dtype=float))
    f1 = np.broadcast_to(np.asarray(f1, dtype=float), w.shape)
    d1 = np.broadcast_to(np.asarray(d1, dtype=float), w.shape)
    nodes, weights = _segments(DOSE_BREAKS, order)
    out = np.zeros(w.shape)
    for l2 in (0.0, 1.0):
        p_l = _bernoulli(long_l2(w, f1, d1), l2)
        for f2 in (0.0, 1.0):
            p_f = _bernoulli(long_flag2(w, f1, d1, l2), f2)
            density = truncated_pdf(
                nodes[None, :], long_dose2_mean(w[:, None], f1[:, None], d1[:, None], l2, f2)
            )
            f2_d, d2_d = flag_dose_map(label, np.full(nodes.shape, f2), nodes)
            q = long_outcome(w[:, None], f1[:, None], d1[:, None], l2, f2_d[None, :], d2_d[None, :])
            out = out + p_l * p_f * np.sum(weights[None, :] * density * q, axis=1)
    return out


def _long_mean(label: str, order: int) -> float:
    nodes, weights = _segments(DOSE_BREAKS, order)
    total = 0.0
    for w in (0.0, 1.0):
        for f1 in (0.0, 1.0):
            p_f = float(_bernoulli(long_flag1(w), f1))
            density = truncated_pdf(nodes, long_dose1_mean(f1, w))
            f1_d, d1_d = flag_dose_map(label, np.full(nodes.shape, f1), nodes)
            m1 = long_first_regression(label, np.full(nodes.shape, w), f1_d, d1_d, order)
            total += 0.5 * p_f * float(np.sum(weights * density * m1))
    return total


@cache
def long_truth(label: str) -> float:
    """``E[Y]`` under a two-node plan, checked against twice the order."""
    value = _long_mean(label, ORDER)
    check = _long_mean(label, 2 * ORDER)
    if abs(value - check) > 1e-10:
        raise AssertionError(f"the quadrature of {label!r} moved by {abs(value - check):.3g}")
    return float(value)


def long_ratios(
    label: str, frame: pd.DataFrame, *, cut: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """The per-node induced density ratios of a two-node plan at the observed values."""
    w, f1, d1, l2, f2, d2 = (
        frame[c].to_numpy(dtype=float) for c in ("W", "F1", "D1", "L2", "F2", "D2")
    )
    r1 = flag_dose_ratio(label, f1, d1, long_flag1(w), lambda c: long_dose1_mean(c, w), cut=cut)
    r2 = flag_dose_ratio(
        label,
        f2,
        d2,
        long_flag2(w, f1, d1, l2),
        lambda c: long_dose2_mean(w, f1, d1, l2, c),
        cut=cut,
    )
    return r1, r2


def long_eif(label: str, frame: pd.DataFrame) -> np.ndarray:
    """The efficient influence function of ``E[Y]`` under a two-node plan, at every row."""
    w, f1, d1, l2, f2, d2, y = (
        frame[c].to_numpy(dtype=float) for c in ("W", "F1", "D1", "L2", "F2", "D2", "Y")
    )
    r1, r2 = long_ratios(label, frame)
    f1_d, d1_d = flag_dose_map(label, f1, d1)
    f2_d, d2_d = flag_dose_map(label, f2, d2)
    m2_observed = long_outcome(w, f1, d1, l2, f2, d2)
    m2_policy = long_outcome(w, f1, d1, l2, f2_d, d2_d)
    m1_observed = long_first_regression(label, w, f1, d1, EIF_ORDER)
    m1_policy = long_first_regression(label, w, f1_d, d1_d, EIF_ORDER)
    return (
        m1_policy - long_truth(label) + r1 * (m2_policy - m1_observed) + r1 * r2 * (y - m2_observed)
    )


# ---------------------------------------------------------------------------- the categorical law


def cat_a1(w1: Any, w2: Any) -> Any:
    return expit(-0.2 + 0.5 * np.asarray(w1, dtype=float) + 0.3 * np.asarray(w2, dtype=float))


def cat_a2(a1: Any, w1: Any, w2: Any) -> Any:
    return expit(
        -0.3
        + 0.8 * np.asarray(a1, dtype=float)
        - 0.4 * np.asarray(w1, dtype=float)
        + 0.2 * np.asarray(w2, dtype=float)
    )


def cat_outcome(a1: Any, a2: Any, w1: Any, w2: Any) -> Any:
    a1 = np.asarray(a1, dtype=float)
    a2 = np.asarray(a2, dtype=float)
    return expit(
        -1.0
        + 0.6 * a1
        + 0.5 * a2
        - 0.7 * a1 * a2
        + 0.4 * np.asarray(w1, dtype=float)
        + 0.3 * np.asarray(w2, dtype=float)
    )


def sample_categorical(n: int, seed: int) -> pd.DataFrame:
    """``n`` rows of the categorical law: ``W1, W2, A1, A2, Y``."""
    rng = np.random.default_rng(seed)
    w1 = rng.uniform(-1.0, 1.0, n)
    w2 = rng.integers(0, 2, n).astype(float)
    a1 = rng.binomial(1, cat_a1(w1, w2)).astype(float)
    a2 = rng.binomial(1, cat_a2(a1, w1, w2)).astype(float)
    y = rng.binomial(1, cat_outcome(a1, a2, w1, w2)).astype(float)
    return pd.DataFrame({"W1": w1, "W2": w2, "A1": a1, "A2": a2, "Y": y})


def categorical_map(label: str, a1: Any, a2: Any) -> tuple[np.ndarray, np.ndarray]:
    a1 = np.asarray(a1, dtype=float)
    a2 = np.asarray(a2, dtype=float)
    if label == "natural":
        return a1, a2
    if label == "a2 off":
        return a1, np.where(a1 == 1.0, 0.0, a2)
    if label == "a1 on":
        return np.ones_like(a1), a2
    raise KeyError(label)


def _cat_joint(a1: float, a2: float, w1: Any, w2: Any) -> Any:
    return _bernoulli(cat_a1(w1, w2), a1) * _bernoulli(cat_a2(a1, w1, w2), a2)


def categorical_ratio(label: str, a1: Any, a2: Any, w1: Any, w2: Any) -> np.ndarray:
    """The discrete formula: the mass the plan sends to the observed tuple over its own mass."""
    a1 = np.asarray(a1, dtype=float)
    a2 = np.asarray(a2, dtype=float)
    reached = np.zeros_like(a1)
    for c1, c2 in itertools.product((0.0, 1.0), repeat=2):
        t1, t2 = categorical_map(label, np.array([c1]), np.array([c2]))
        hit = (a1 == t1[0]) & (a2 == t2[0])
        reached = reached + np.where(hit, _cat_joint(c1, c2, w1, w2), 0.0)
    observed = np.zeros_like(a1)
    for c1, c2 in itertools.product((0.0, 1.0), repeat=2):
        observed = observed + np.where((a1 == c1) & (a2 == c2), _cat_joint(c1, c2, w1, w2), 0.0)
    return np.asarray(reached / observed, dtype=float)


def _categorical_mean(label: str, w_order: int) -> float:
    w1s, w1_weights = _w1_nodes(w_order)
    total = 0.0
    for w2 in (0.0, 1.0):
        for c1, c2 in itertools.product((0.0, 1.0), repeat=2):
            t1, t2 = categorical_map(label, np.array([c1]), np.array([c2]))
            mass = _cat_joint(c1, c2, w1s, w2)
            total += 0.5 * float(np.sum(w1_weights * mass * cat_outcome(t1[0], t2[0], w1s, w2)))
    return total


@cache
def categorical_truth(label: str) -> float:
    value = _categorical_mean(label, W_ORDER)
    check = _categorical_mean(label, 2 * W_ORDER)
    if abs(value - check) > 1e-10:
        raise AssertionError(f"the quadrature of {label!r} moved by {abs(value - check):.3g}")
    return float(value)


def categorical_eif(label: str, frame: pd.DataFrame) -> np.ndarray:
    w1, w2, a1, a2, y = (frame[c].to_numpy(dtype=float) for c in ("W1", "W2", "A1", "A2", "Y"))
    ratio = categorical_ratio(label, a1, a2, w1, w2)
    t1, t2 = categorical_map(label, a1, a2)
    q = cat_outcome(a1, a2, w1, w2)
    return ratio * (y - q) + cat_outcome(t1, t2, w1, w2) - categorical_truth(label)


# ---------------------------------------------------------------------------- bounds

SAMPLERS: dict[str, Callable[[int, int], pd.DataFrame]] = {
    "point": sample_point,
    "long": sample_long,
    "ordered": sample_ordered,
    "categorical": sample_categorical,
}
EIFS: dict[str, Callable[[str, pd.DataFrame], np.ndarray]] = {
    "point": point_eif,
    "long": long_eif,
    "ordered": ordered_eif,
    "categorical": categorical_eif,
}
TRUTHS: dict[str, Callable[[str], float]] = {
    "point": point_truth,
    "long": long_truth,
    "ordered": ordered_truth,
    "categorical": categorical_truth,
}


def contrast_sd(law: str, label: str, *, draws: int = BOUND_DRAWS, chunk: int = 25_000) -> float:
    """The efficiency bound of ``E[Y^label] - E[Y^natural]`` on ``law``, as a standard deviation.

    The draws are taken in chunks of ``chunk`` rows with one seed, so the memory stays below
    a few hundred megabytes whatever ``draws`` is.
    """
    frame = SAMPLERS[law](draws, BOUND_SEED)
    curve = np.empty(draws)
    for start in range(0, draws, chunk):
        part = frame.iloc[start : start + chunk]
        curve[start : start + len(part)] = EIFS[law](label, part) - EIFS[law]("natural", part)
    return float(np.std(curve))


# ---------------------------------------------------------------------------- declarations
#
# Module-level callables, so a declared plan pickles.


def _set_one(a: Any, h: Any) -> np.ndarray:
    return np.ones(len(a))


def _flag_if_knee(a: Any, h: Any, *, dose: str) -> np.ndarray:
    return np.where(np.asarray(h[dose], dtype=float) >= KNEE, 1, np.asarray(a))


def _a2_off(a: Any, h: Any) -> np.ndarray:
    return np.where(np.asarray(h["A1"], dtype=float) == 1.0, 0, np.asarray(a))


def _scale_map(a: Any, h: Any) -> np.ndarray:
    return FACTOR * np.asarray(a, dtype=float)


def _scale_inverse(b: Any, h: Any) -> np.ndarray:
    return np.asarray(b, dtype=float) / FACTOR


def _scale_slope(b: Any, h: Any) -> np.ndarray:
    return np.full(len(b), 1.0 / FACTOR)


def _shift_map(a: Any, h: Any) -> np.ndarray:
    return np.asarray(a, dtype=float) + STEP


def _shift_inverse(b: Any, h: Any) -> np.ndarray:
    return np.asarray(b, dtype=float) - STEP


def _unit_slope(b: Any, h: Any) -> np.ndarray:
    return np.ones(len(b))


def _flag_gate(h: Any, *, flag: str) -> np.ndarray:
    """The scaled piece ends at 4.4 where the natural flag is one, and is empty elsewhere."""
    return np.where(np.asarray(h[flag], dtype=float) == 1.0, SCALE_HELD, -np.inf)


def _knee_gate(h: Any, *, dose: str) -> np.ndarray:
    """The shifted piece ends at 5.0 where the natural earlier dose is at most 3."""
    return np.where(np.asarray(h[dose], dtype=float) <= KNEE, SHIFT_HELD, -np.inf)


def _gated_scale(flag: str) -> ModifiedPolicy:
    gate = partial(_flag_gate, flag=flag)
    return ModifiedPolicy(
        f"scale where {flag} is 1",
        pieces=(
            Piece(-np.inf, gate, _scale_map, _scale_inverse, _scale_slope, closed="right"),
            Piece(gate, np.inf, closed="right"),
        ),
        policy_kind="known",
    )


def _gated_shift(dose: str) -> ModifiedPolicy:
    gate = partial(_knee_gate, dose=dose)
    return ModifiedPolicy(
        f"up where {dose} is small",
        pieces=(
            Piece(-np.inf, gate, _shift_map, _shift_inverse, _unit_slope, closed="right"),
            Piece(gate, np.inf, closed="right"),
        ),
        policy_kind="known",
    )


def flag_dose_policy(label: str, flag: str, dose: str) -> JointPolicy:
    """A flag-and-dose node's plan as a joint policy over the columns ``flag`` and ``dose``."""
    if label == "natural":
        return JointPolicy(label, components={}, policy_kind="known")
    if label == "dose up":
        return JointPolicy(label, components={dose: Shift(STEP, cap=CAP)}, policy_kind="known")
    if label == "flag scaled":
        return JointPolicy(
            label,
            components={
                flag: ModifiedPolicy("flag on", apply=_set_one, policy_kind="known"),
                dose: _gated_scale(flag),
            },
            policy_kind="known",
        )
    raise KeyError(label)


def point_policies(labels: tuple[str, ...] = POINT_LABELS) -> tuple[JointPolicy, ...]:
    return tuple(flag_dose_policy(label, "F", "D") for label in labels)


def long_regimens(labels: tuple[str, ...] = LONG_LABELS) -> dict[str, DynamicRegimen]:
    return {
        label: DynamicRegimen(
            label, (flag_dose_policy(label, "F1", "D1"), flag_dose_policy(label, "F2", "D2"))
        )
        for label in labels
    }


def ordered_policy(label: str) -> JointPolicy:
    """An ordered-law plan over ``F, D1, D2``, with ``D2`` reading the natural ``D1``."""
    if label == "natural":
        return JointPolicy(label, components={}, policy_kind="known")
    if label == "ordered":
        return JointPolicy(
            label,
            components={"D1": Scale(FACTOR, cap=CAP), "D2": _gated_shift("D1")},
            order=("D1", "D2"),
            policy_kind="known",
        )
    if label == "full":
        return JointPolicy(
            label,
            components={
                "F": ModifiedPolicy(
                    "flag on where D1 is large",
                    apply=partial(_flag_if_knee, dose="D1"),
                    policy_kind="known",
                ),
                "D1": _gated_scale("F"),
                "D2": _gated_shift("D1"),
            },
            order=("D1", "D2"),
            policy_kind="known",
        )
    raise KeyError(label)


def ordered_policies(labels: tuple[str, ...] = ORDERED_LABELS) -> tuple[JointPolicy, ...]:
    return tuple(ordered_policy(label) for label in labels)


def ordered_regimens(labels: tuple[str, ...] = ORDERED_LABELS) -> dict[str, DynamicRegimen]:
    return {label: DynamicRegimen(label, (ordered_policy(label),)) for label in labels}


def categorical_policy(label: str) -> JointPolicy:
    if label == "natural":
        return JointPolicy(label, components={}, policy_kind="known")
    if label == "a2 off":
        return JointPolicy(
            label,
            components={"A2": ModifiedPolicy("a2 off", apply=_a2_off, policy_kind="known")},
            policy_kind="known",
        )
    if label == "a1 on":
        return JointPolicy(
            label,
            components={"A1": ModifiedPolicy("a1 on", apply=_set_one, policy_kind="known")},
            policy_kind="known",
        )
    raise KeyError(label)


def categorical_policies(labels: tuple[str, ...] = CATEGORICAL_LABELS) -> tuple[JointPolicy, ...]:
    return tuple(categorical_policy(label) for label in labels)


# ---------------------------------------------------------------------------- learners


def _hazard(index: np.ndarray, edges: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """The pooled-hazard probabilities of bin ``index`` under the truncated normal at ``mean``."""
    last = len(edges) - 2
    lower = np.where(index == 0, LOWER, edges[index])
    upper = np.where(index == last, UPPER, edges[np.minimum(index + 1, last + 1)])
    mass = norm.cdf(UPPER - mean) - norm.cdf(LOWER - mean)
    below = (norm.cdf(lower - mean) - norm.cdf(LOWER - mean)) / mass
    reached = (norm.cdf(upper - mean) - norm.cdf(LOWER - mean)) / mass
    survived = 1.0 - below
    hazard = np.divide(reached - below, survived, out=np.ones_like(below), where=survived > 1e-15)
    hazard = np.clip(hazard, 1e-12, 1.0 - 1e-12)
    return np.column_stack([1.0 - hazard, hazard])


#: The marginal dose mean the misspecified mechanism uses, and its flag probability.
MARGINAL_MEAN = 2.8
MARGINAL_FLAG = 0.5


class OracleDoseHazard(BaseEstimator):
    """The exact pooled-hazard probabilities of each dose of a law, given its design.

    The density layer hands this the dose's conditioning design followed by the bin index
    (``bin_design = "index"`` drops the indicator block): the history, the categorical
    components' indicators and the earlier doses.  The width tells the dose.

    ==========  =====  ======================================  =====================
    law         width  design                                  dose
    ==========  =====  ======================================  =====================
    point       4      ``[W1, W2, F, bin]``                    ``D``
    ordered     4      ``[W1, W2, F, bin]``                    ``D1``
    ordered     5      ``[W1, W2, F, D1, bin]``                ``D2``
    long        3      ``[W, F1, bin]``                        ``D1`` (node 1)
    long        6      ``[W, L2, F1, D1, F2, bin]``            ``D2`` (node 2)
    ==========  =====  ======================================  =====================

    ``marginal=True`` is the misspecified mechanism: the same family at
    :data:`MARGINAL_MEAN`, which ignores the design.

    Parameters
    ----------
    law : str
        ``"point"``, ``"ordered"`` or ``"long"``.
    edges : tuple of tuple of float
        The bin edges of each dose, in column order.
    marginal : bool
        Whether to ignore the design.
    """

    bin_design = "index"

    def __init__(
        self, law: str = "point", edges: tuple[tuple[float, ...], ...] = (), marginal: bool = False
    ) -> None:
        self.law = law
        self.edges = edges
        self.marginal = marginal

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> OracleDoseHazard:
        del X, y, sample_weight
        self.classes_ = np.array([0.0, 1.0])
        return self

    def _dose(self, design: np.ndarray) -> tuple[int, np.ndarray]:
        width = design.shape[1]
        x = design
        if self.law == "point" and width == 4:
            return 0, point_dose_mean(x[:, 2], x[:, 0], x[:, 1])
        if self.law == "ordered" and width == 4:
            return 0, point_dose_mean(x[:, 2], x[:, 0], x[:, 1])
        if self.law == "ordered" and width == 5:
            return 1, ordered_d2_mean(x[:, 3], x[:, 2], x[:, 0])
        if self.law == "long" and width == 3:
            return 0, long_dose1_mean(x[:, 1], x[:, 0])
        if self.law == "long" and width == 6:
            return 1, long_dose2_mean(x[:, 0], x[:, 2], x[:, 3], x[:, 1], x[:, 4])
        raise ValueError(f"unexpected {self.law} hazard design width {width}")

    def predict_proba(self, X: Any) -> np.ndarray:
        design = np.asarray(X, dtype=float)
        dose, mean = self._dose(design)
        if self.marginal:
            mean = np.full(len(design), MARGINAL_MEAN)
        index = np.rint(design[:, -1]).astype(int)
        return _hazard(index, np.asarray(self.edges[dose], dtype=float), np.asarray(mean))


class OracleFlag(BaseEstimator):
    """The exact flag probability, the categorical factor of a mixed node.

    ==========  =====  ==========================
    law         width  design
    ==========  =====  ==========================
    point       2      ``[W1, W2]``
    ordered     2      ``[W1, W2]``
    long        1      ``[W]`` (node 1)
    long        4      ``[W, L2, F1, D1]`` (node 2)
    ==========  =====  ==========================

    The classes are the cell codes, 0 for flag 0 and 1 for flag 1.  ``marginal=True`` returns
    :data:`MARGINAL_FLAG` everywhere.

    Parameters
    ----------
    law : str
        ``"point"``, ``"ordered"`` or ``"long"``.
    marginal : bool
        Whether to ignore the design.
    """

    def __init__(self, law: str = "point", marginal: bool = False) -> None:
        self.law = law
        self.marginal = marginal

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> OracleFlag:
        del X, y, sample_weight
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        x = np.asarray(X, dtype=float)
        width = x.shape[1]
        if self.law in {"point", "ordered"} and width == 2:
            p = point_flag(x[:, 0], x[:, 1])
        elif self.law == "long" and width == 1:
            p = long_flag1(x[:, 0])
        elif self.law == "long" and width == 4:
            p = long_flag2(x[:, 0], x[:, 2], x[:, 3], x[:, 1])
        else:
            raise ValueError(f"unexpected {self.law} flag design width {width}")
        if self.marginal:
            p = np.full(len(x), MARGINAL_FLAG)
        return np.column_stack([1.0 - p, p])


class Oracle(BaseEstimator):
    """The one treatment learner a density-route fit takes: the flag oracle and the hazards.

    The package fits the categorical factor and every dose's hazards with the same learner
    (:func:`~cleverly.interventions.joint_density.evaluate_joint_density`).  A binned design
    ends in the bin index and is one column wider than any flag design of the same law, so the
    width routes each call: a design that one of :class:`OracleFlag`'s widths names is a flag
    design, and every other one is a hazard design.

    Parameters
    ----------
    law : str
        ``"point"``, ``"ordered"`` or ``"long"``.
    edges : tuple of tuple of float
        The bin edges of each dose, in column order.
    marginal : bool
        The misspecified mechanism of both factors.
    """

    bin_design = "index"
    FLAG_WIDTHS: ClassVar[dict[str, tuple[int, ...]]] = {
        "point": (2,),
        "ordered": (2,),
        "long": (1, 4),
    }

    def __init__(
        self, law: str = "point", edges: tuple[tuple[float, ...], ...] = (), marginal: bool = False
    ) -> None:
        self.law = law
        self.edges = edges
        self.marginal = marginal

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> Oracle:
        del X, y, sample_weight
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        x = np.asarray(X, dtype=float)
        if x.shape[1] in self.FLAG_WIDTHS[self.law]:
            return OracleFlag(self.law, self.marginal).fit(x, None).predict_proba(x)
        return OracleDoseHazard(self.law, self.edges, self.marginal).predict_proba(x)


class OracleLogOdds(BaseEstimator):
    r"""The correctly specified classifier of the stacked ratio route, on one oracle feature.

    The stacked classification of Section 5.4 of Díaz et al. (2023) has the odds
    :math:`g^d(a \mid h) / g(a \mid h)`.  This learner fits an unpenalized logistic regression
    on the one feature :math:`\log r(a, h)`, the true log ratio of ``label``, so the true odds
    are the model at ``(0, 1)``.  The design is the component block followed by the history.

    ==========  =====  ====================================
    law         width  design
    ==========  =====  ====================================
    point       4      ``[F, D, W1, W2]``
    ordered     5      ``[F, D1, D2, W1, W2]``
    long        3      ``[F1, D1, W]`` (node 1)
    long        6      ``[F2, D2, W, L2, F1, D1]`` (node 2)
    ==========  =====  ====================================

    Parameters
    ----------
    law : str
        ``"point"``, ``"ordered"`` or ``"long"``.
    label : str
        The plan whose node ratio is the feature.
    """

    def __init__(self, law: str = "point", label: str = "flag scaled") -> None:
        self.law = law
        self.label = label

    def _ratio(self, x: np.ndarray) -> np.ndarray:
        width = x.shape[1]
        if self.law == "point" and width == 4:
            f, d, w1, w2 = x.T
            return flag_dose_ratio(
                self.label, f, d, point_flag(w1, w2), lambda c: point_dose_mean(c, w1, w2)
            )
        if self.law == "ordered" and width == 5:
            f, d1, d2, w1, w2 = x.T
            return ordered_ratio(self.label, f, d1, d2, w1, w2)
        if self.law == "long" and width == 3:
            f1, d1, w = x.T
            return flag_dose_ratio(
                self.label, f1, d1, long_flag1(w), lambda c: long_dose1_mean(c, w)
            )
        if self.law == "long" and width == 6:
            f2, d2, w, l2, f1, d1 = x.T
            return flag_dose_ratio(
                self.label,
                f2,
                d2,
                long_flag2(w, f1, d1, l2),
                lambda c: long_dose2_mean(w, f1, d1, l2, c),
            )
        raise ValueError(f"unexpected {self.law} classifier design width {width}")

    def _feature(self, X: Any) -> np.ndarray:
        ratio = self._ratio(np.asarray(X, dtype=float))
        return np.log(np.clip(ratio, 1e-13, None)).reshape(-1, 1)

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> OracleLogOdds:
        self.model_ = LogisticRegression(penalty=None, max_iter=1000).fit(
            self._feature(X), np.asarray(y), sample_weight=sample_weight
        )
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        return np.asarray(self.model_.predict_proba(self._feature(X)), dtype=float)


class SquaredLogistic(BaseEstimator):
    """An unpenalized quasibinomial GLM on the design plus the squares of ``squared`` columns.

    The point and ordered outcome regressions are logistic in the treatment, the squares of the
    doses and the covariates, so this is the correctly specified model wherever ``squared``
    names the dose columns of the design.  With ``squared=()`` it is the main-terms GLM.

    Parameters
    ----------
    squared : tuple of int
        Design columns whose squares are appended.
    product : tuple of int
        Two design columns whose product is appended, or ``()``.
    """

    def __init__(self, squared: tuple[int, ...] = (), product: tuple[int, ...] = ()) -> None:
        self.squared = squared
        self.product = product

    def _design(self, X: Any) -> np.ndarray:
        matrix = np.asarray(X, dtype=float)
        extra = [matrix[:, index] ** 2 for index in self.squared]
        if self.product:
            left, right = self.product
            extra.append(matrix[:, left] * matrix[:, right])
        return np.column_stack([matrix, *extra]) if extra else matrix

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> SquaredLogistic:
        self.model_ = QuasiBinomialGLM().fit(self._design(X), y, sample_weight=sample_weight)
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict(self, X: Any) -> np.ndarray:
        return np.asarray(self.model_.predict(self._design(X)), dtype=float)

    def predict_proba(self, X: Any) -> np.ndarray:
        p = np.clip(self.predict(X), 0.0, 1.0)
        return np.column_stack([1.0 - p, p])


class RichPseudo(BaseEstimator):
    """Least squares on a rich basis of ``[W, F1, D1]``, for the two-node law's first regression.

    The first node's regression is an integral of the final one and has no closed form.  A
    degree-five polynomial in ``D1``, its interactions with ``W`` and ``F1``, and steps at the
    caps of the plans, fitted by least squares, approximates it to well below the sampling error
    at the study's sizes (``tests/unit/test_vector_mtp_design.py`` measures the gap).
    """

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> RichPseudo:
        self.model_ = LinearRegression().fit(self._design(X), y, sample_weight=sample_weight)
        return self

    @staticmethod
    def _design(X: Any) -> np.ndarray:
        matrix = np.asarray(X, dtype=float)
        w, f, d = matrix[:, 0], matrix[:, 1], matrix[:, 2]
        powers = [d**k for k in range(1, 6)]
        steps = [(d > SCALE_HELD).astype(float), (d > SHIFT_HELD).astype(float)]
        return np.column_stack(
            [
                w,
                f,
                w * f,
                *powers,
                *(w * p for p in powers[:3]),
                *(f * p for p in powers[:3]),
                *steps,
                *(f * s for s in steps),
            ]
        )

    def predict(self, X: Any) -> np.ndarray:
        return np.asarray(self.model_.predict(self._design(X)), dtype=float)


class StackedGLM(BaseEstimator):
    """The unpenalized main-terms logistic GLM of the unsupplied classifier pair.

    ``lmtp``'s ``SL.glm`` fits ``glm(..., family = binomial())`` on the stacked rows by IRLS to
    R's relative deviance tolerance.  :class:`~tests.studies.fractional_glm.QuasiBinomialGLM`
    runs the same score equations to the same tolerance, so the two sides fit one model.
    """

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> StackedGLM:
        self.model_ = QuasiBinomialGLM().fit(
            np.asarray(X, dtype=float), np.asarray(y, dtype=float), sample_weight=sample_weight
        )
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        p = np.clip(np.asarray(self.model_.predict(np.asarray(X, dtype=float))), 0.0, 1.0)
        return np.column_stack([1.0 - p, p])
