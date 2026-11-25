r"""The kernel-smoothed dose-response curve: its declaration, weights, quadrature and targeting.

For a continuous treatment :math:`A`, the dose-response curve
:math:`\theta(a) = E\{\bar Q(a, W)\}` is not pathwise differentiable at a fixed dose, so no
root-n interval exists for it in a nonparametric model.  Its kernel-smoothed version at a
declared bandwidth :math:`h` and dose :math:`a_0`,

.. math::

    \theta_h(a_0) = \int_D w_{h,a_0}(a)\,\theta(a)\,da,

is a known linear functional of the curve and is pathwise differentiable.  The weight
:math:`w_{h,a_0}` is the local-linear equivalent kernel under Lebesgue measure on the declared
dose interval :math:`D`: it integrates to one and has zero first moment about :math:`a_0`, so
:math:`\theta_h(a_0) = \theta(a_0)` wherever the curve is linear on the kernel's support, at
interior and boundary doses.  In the interior it equals the kernel
:math:`K_h(\cdot - a_0)`; at a boundary it is signed.

The efficient influence function is

.. math::

    D(O) = \frac{w(A)}{g(A \mid W)}\{Y - \bar Q(A, W)\} + \int_D w(a)\bar Q(a, W)\,da
    - \theta_h(a_0),

and the exact remainder is a product of the density error and the outcome-regression error
(``docs/technical-reference/dose-response.md``).

This module holds the parts that do not depend on how the nuisances were fitted:
:class:`DoseResponse` (the declaration), :func:`equivalent_kernel`, :class:`QuadratureRule`
(a composite Gauss-Legendre rule on a partition of :math:`D` shared by every pair), and
:func:`target_pair` (one pooled one-dimensional fluctuation and the plug-in).  The estimator
that fits the nuisances is :class:`cleverly.estimators.DoseResponseTMLE`.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
from scipy.special import expit, logit

from ._typing import FloatArray
from .exceptions import DataError

__all__ = [
    "DoseResponse",
    "KernelName",
    "PairTarget",
    "QuadratureRule",
    "equivalent_kernel",
    "influence_curve",
    "kernel",
    "pair_name",
    "plug_in",
    "target_pair",
]

#: The kernels a declaration may name.  Each is a symmetric density on ``[-1, 1]``.
KernelName = Literal["epanechnikov", "triweight", "uniform"]

#: The most (bandwidth, dose) pairs one declaration may hold.  Each pair is its own fluctuation,
#: and the node table grows with the pair count, so the cap bounds memory and run time.
MAX_PAIRS: Final[int] = 200

#: Gauss-Legendre nodes used to compute the exact local-linear moment matrix.  Every kernel in
#: :data:`KernelName` is a polynomial of degree at most 6 on its support, so the integrand of the
#: moment matrix has degree at most 8 and 16 nodes integrate it exactly.
_MOMENT_NODES: Final[int] = 16

#: The default Kish effective-size floor below which a pair's interval carries a label.
DEFAULT_SUPPORT_FLOOR: Final[float] = 50.0


def kernel(name: KernelName, u: FloatArray) -> FloatArray:
    """Evaluate a declared kernel at ``u``; zero outside ``[-1, 1]``.

    Parameters
    ----------
    name : {"epanechnikov", "triweight", "uniform"}
        The kernel.
    u : ndarray
        Points at which to evaluate it.

    Returns
    -------
    ndarray
        The kernel at ``u``, with the shape of ``u``.
    """
    x = np.asarray(u, dtype=float)
    inside = np.abs(x) <= 1.0
    if name == "epanechnikov":
        value = 0.75 * (1.0 - x * x)
    elif name == "triweight":
        value = (35.0 / 32.0) * (1.0 - x * x) ** 3
    elif name == "uniform":
        value = np.full_like(x, 0.5)
    else:
        raise DataError(f"unknown kernel {name!r}; choose epanechnikov, triweight or uniform")
    return np.asarray(np.where(inside, value, 0.0), dtype=float)


def _support_piece(a0: float, h: float, lower: float, upper: float) -> tuple[float, float]:
    return max(a0 - h, lower), min(a0 + h, upper)


def _moment_inverse_row(
    name: KernelName, a0: float, h: float, lower: float, upper: float
) -> FloatArray:
    """The first row of :math:`M^{-1}`, the local-linear moment matrix on the kernel's support."""
    left, right = _support_piece(a0, h, lower, upper)
    x, wq = np.polynomial.legendre.leggauss(_MOMENT_NODES)
    a = 0.5 * (right - left) * x + 0.5 * (right + left)
    mass = 0.5 * (right - left) * wq * kernel(name, (a - a0) / h) / h
    z = (a - a0) / h
    moment = np.array(
        [
            [mass.sum(), (mass * z).sum()],
            [(mass * z).sum(), (mass * z * z).sum()],
        ]
    )
    return np.asarray(np.linalg.solve(moment, np.array([1.0, 0.0])), dtype=float)


def equivalent_kernel(
    a: FloatArray,
    a0: float,
    h: float,
    support: tuple[float, float],
    name: KernelName = "epanechnikov",
) -> FloatArray:
    r"""The local-linear equivalent kernel :math:`w_{h,a_0}(a)`, Lebesgue measure on ``support``.

    :math:`w(a) = e_1^\top M^{-1}(1, (a - a_0)/h)^\top K_h(a - a_0)\,1\{a \in D\}`, with
    :math:`M = \int_D (1, z)^\top (1, z) K_h(a - a_0)\,da` and :math:`z = (a - a_0)/h`.  It
    depends on the declaration only, never on the data.

    Parameters
    ----------
    a : ndarray
        Doses at which to evaluate the weight.
    a0 : float
        The evaluation dose, inside ``support``.
    h : float
        The bandwidth, positive.
    support : tuple of float
        The declared dose interval ``(lower, upper)``.
    name : {"epanechnikov", "triweight", "uniform"}
        The kernel.

    Returns
    -------
    ndarray
        The weight at ``a``, with the shape of ``a``.
    """
    lower, upper = float(support[0]), float(support[1])
    row = _moment_inverse_row(name, float(a0), float(h), lower, upper)
    x = np.asarray(a, dtype=float)
    z = (x - a0) / h
    inside = (x >= lower) & (x <= upper)
    value = (row[0] + row[1] * z) * kernel(name, z) / h
    return np.asarray(np.where(inside, value, 0.0), dtype=float)


def pair_name(h: float, a0: float) -> str:
    """The parameter name of one (bandwidth, dose) pair, with a fixed float format.

    Parameters
    ----------
    h : float
        The bandwidth.
    a0 : float
        The dose.

    Returns
    -------
    str
        ``"dose_response[h=<h>,a=<a0>]"``, each number formatted with ``%.10g``.
    """
    return f"dose_response[h={h:.10g},a={a0:.10g}]"


@dataclass(frozen=True)
class DoseResponse:
    r"""The declaration of a kernel-smoothed dose-response curve.

    Every field is part of the estimand and must be chosen without reading the data, as
    :attr:`cleverly.msm.MSM.doses` is.  A dose interval taken from sample quantiles of the
    treatment moves the target at boundary pairs by an amount of order :math:`n^{-1/2}` that the
    reported influence curve omits.

    The target at each pair :math:`(h, a_0)` is the smoothed curve
    :math:`\theta_h(a_0) = \int_D w_{h,a_0}(a)\theta(a)\,da` (:func:`equivalent_kernel`), not the
    curve :math:`\theta(a_0)` itself.  The two differ by :math:`O(h^2)` where the curve is twice
    differentiable.

    Parameters
    ----------
    doses : tuple of float
        The evaluation doses, strictly increasing and inside ``support``.
    bandwidths : tuple of float
        The bandwidths, positive and distinct.
    support : tuple of float
        The dose interval :math:`D = (a_\mathrm{lo}, a_\mathrm{hi})`.  It must lie inside the
        observed dose range, with a margin, because the fitted density is zero outside it.
    kernel : {"epanechnikov", "triweight", "uniform"}
        The kernel.
    nodes_per_piece : int
        Gauss-Legendre nodes on each piece of the quadrature partition.
    support_floor : float
        The Kish effective size below which a pair's interval carries the
        ``"low_dose_support_plugin"`` label.

    Examples
    --------
    >>> from cleverly.dose_response import DoseResponse
    >>> curve = DoseResponse(doses=(1.0, 2.0), bandwidths=(0.5,), support=(0.0, 3.0))
    >>> curve.pairs
    ((0.5, 1.0), (0.5, 2.0))
    """

    doses: tuple[float, ...]
    bandwidths: tuple[float, ...]
    support: tuple[float, float]
    kernel: KernelName = "epanechnikov"
    nodes_per_piece: int = 8
    support_floor: float = DEFAULT_SUPPORT_FLOOR

    def __post_init__(self) -> None:
        doses = tuple(float(d) for d in np.asarray(self.doses, dtype=float).reshape(-1))
        bandwidths = tuple(float(b) for b in np.asarray(self.bandwidths, dtype=float).reshape(-1))
        if len(self.support) != 2:
            raise DataError("support must be a pair (lower, upper)")
        lower, upper = float(self.support[0]), float(self.support[1])
        object.__setattr__(self, "doses", doses)
        object.__setattr__(self, "bandwidths", bandwidths)
        object.__setattr__(self, "support", (lower, upper))
        if not (math.isfinite(lower) and math.isfinite(upper) and lower < upper):
            raise DataError("support must be finite with lower < upper")
        if not doses:
            raise DataError("a dose-response curve needs at least one dose")
        if not bandwidths:
            raise DataError("a dose-response curve needs at least one bandwidth")
        if any(not math.isfinite(d) for d in doses) or np.any(np.diff(doses) <= 0.0):
            raise DataError("doses must be finite and strictly increasing")
        if doses[0] < lower or doses[-1] > upper:
            raise DataError(f"every dose must lie inside the support [{lower:g}, {upper:g}]")
        if any(not (math.isfinite(b) and b > 0.0) for b in bandwidths):
            raise DataError("bandwidths must be finite and positive")
        if len(set(bandwidths)) != len(bandwidths):
            raise DataError("bandwidths must be distinct")
        if self.kernel not in ("epanechnikov", "triweight", "uniform"):
            raise DataError(
                f"unknown kernel {self.kernel!r}; choose epanechnikov, triweight or uniform"
            )
        if len(doses) * len(bandwidths) > MAX_PAIRS:
            raise DataError(
                f"a dose-response declaration holds at most {MAX_PAIRS} (bandwidth, dose) pairs; "
                f"got {len(doses) * len(bandwidths)}"
            )
        if int(self.nodes_per_piece) != self.nodes_per_piece or self.nodes_per_piece < 2:
            raise DataError("nodes_per_piece must be an integer of at least 2")
        if not (math.isfinite(self.support_floor) and self.support_floor >= 0.0):
            raise DataError("support_floor must be finite and non-negative")
        names = [pair_name(h, a0) for h, a0 in self.pairs]
        if len(set(names)) != len(names):
            raise DataError(
                "two declared values coincide after formatting to 10 significant digits"
            )

    @classmethod
    def spanning(
        cls,
        support: tuple[float, float],
        n_doses: int = 9,
        fractions: Sequence[float] = (0.1, 0.15, 0.2, 0.3),
        **kwargs: object,
    ) -> DoseResponse:
        """Build doses and bandwidths from the declared interval alone.

        The doses are ``n_doses`` equally spaced points from the lower to the upper end of
        ``support``, and the bandwidths are ``fractions`` of its length.

        Parameters
        ----------
        support : tuple of float
            The dose interval.
        n_doses : int
            The number of evaluation doses.
        fractions : sequence of float
            Bandwidths as fractions of the interval's length.
        **kwargs : object
            The remaining fields of :class:`DoseResponse`.

        Returns
        -------
        DoseResponse
            The declaration.
        """
        lower, upper = float(support[0]), float(support[1])
        if int(n_doses) != n_doses or n_doses < 1:
            raise DataError("n_doses must be a positive integer")
        doses = tuple(np.linspace(lower, upper, int(n_doses)).tolist())
        span = upper - lower
        bandwidths = tuple(float(f) * span for f in fractions)
        return cls(doses=doses, bandwidths=bandwidths, support=(lower, upper), **kwargs)  # type: ignore[arg-type]

    @property
    def pairs(self) -> tuple[tuple[float, float], ...]:
        """Every declared ``(bandwidth, dose)`` pair, bandwidth-major."""
        return tuple((h, a0) for h in self.bandwidths for a0 in self.doses)

    @property
    def names(self) -> tuple[str, ...]:
        """The parameter name of each pair, in :attr:`pairs` order."""
        return tuple(pair_name(h, a0) for h, a0 in self.pairs)

    def weight(self, h: float, a0: float, a: FloatArray) -> FloatArray:
        """The equivalent kernel of one pair at doses ``a``.

        Parameters
        ----------
        h : float
            The bandwidth.
        a0 : float
            The dose.
        a : ndarray
            Doses at which to evaluate the weight.

        Returns
        -------
        ndarray
            The weight, with the shape of ``a``.
        """
        return equivalent_kernel(a, a0, h, self.support, self.kernel)

    def breaks(self) -> FloatArray:
        """The break points the declaration puts into the quadrature partition.

        Returns
        -------
        ndarray
            The interval's endpoints and every :math:`a_0 \\pm h` inside it, sorted.
        """
        lower, upper = self.support
        points = [lower, upper]
        for h, a0 in self.pairs:
            for point in (a0 - h, a0 + h):
                if lower < point < upper:
                    points.append(point)
        return np.unique(np.asarray(points, dtype=float))

    def quadrature(
        self, extra_breaks: Sequence[float] | FloatArray = (), *, refine: int = 1
    ) -> QuadratureRule:
        """The composite Gauss-Legendre rule shared by every pair.

        Parameters
        ----------
        extra_breaks : sequence of float
            More break points, such as a binned density's edges and bin centres, so that the
            density is log-linear on each piece.
        refine : int
            Multiplies :attr:`nodes_per_piece`; the doubling diagnostic passes 2.

        Returns
        -------
        QuadratureRule
            Nodes and per-pair weights.
        """
        lower, upper = self.support
        extra = np.asarray(list(extra_breaks), dtype=float)
        extra = extra[(extra > lower) & (extra < upper)]
        points = np.unique(np.concatenate([self.breaks(), extra]))
        x, wq = np.polynomial.legendre.leggauss(int(self.nodes_per_piece) * int(refine))
        nodes: list[FloatArray] = []
        masses: list[FloatArray] = []
        for left, right in itertools.pairwise(points):
            nodes.append(0.5 * (right - left) * x + 0.5 * (right + left))
            masses.append(0.5 * (right - left) * wq)
        node = np.concatenate(nodes)
        mass = np.concatenate(masses)
        weights = np.vstack([mass * self.weight(h, a0, node) for h, a0 in self.pairs])
        return QuadratureRule(node, weights)


@dataclass(frozen=True)
class QuadratureRule:
    r"""A composite quadrature rule over the declared dose interval.

    Parameters
    ----------
    nodes : ndarray
        ``(K,)`` quadrature doses.
    weights : ndarray
        ``(P, K)``: for each pair, the quadrature mass times the equivalent kernel at each node,
        so :math:`\int_D w_p(a) f(a)\,da \approx \sum_k \mathrm{weights}_{pk} f(a_k)`.
    """

    nodes: FloatArray
    weights: FloatArray

    @property
    def n_nodes(self) -> int:
        """Return the number of quadrature doses."""
        return int(self.nodes.size)

    def used(self) -> FloatArray:
        """``(K,)`` booleans: the nodes where some pair's weight is nonzero."""
        return np.asarray(np.any(self.weights != 0.0, axis=0))


@dataclass(frozen=True)
class PairTarget:
    """One pair's targeted estimate on the scaled outcome, before the scale is undone.

    Parameters
    ----------
    psi : float
        The plug-in of the targeted regression.
    influence_curve : ndarray
        ``(n,)`` per-row influence curve on the scaled outcome.
    epsilon : float
        The fluctuation coefficient.
    score : float
        The weighted mean of the clever covariate times the residual at the solution.
    converged : bool
        Whether the score reached the tolerance.
    """

    psi: float
    influence_curve: FloatArray
    epsilon: float
    score: float
    converged: bool


@dataclass(frozen=True)
class _PairSupport:
    rows: int
    kish: float
    min_density: float
    max_covariate: float


def plug_in(q_nodes: FloatArray, pair_weights: FloatArray, row_weights: FloatArray) -> FloatArray:
    r"""Per-row :math:`\int_D w(a)\bar Q(a, W_i)\,da` by the quadrature rule.

    Parameters
    ----------
    q_nodes : ndarray
        ``(n, K)`` outcome regression at the quadrature doses.
    pair_weights : ndarray
        ``(K,)`` one row of :attr:`QuadratureRule.weights`.
    row_weights : ndarray
        ``(n,)`` observation weights; unused here, accepted so a caller's call reads alike.

    Returns
    -------
    ndarray
        ``(n,)`` per-row integrals.
    """
    del row_weights
    return np.asarray(q_nodes @ pair_weights, dtype=float)


def influence_curve(
    y: FloatArray,
    q_obs: FloatArray,
    h_obs: FloatArray,
    integral: FloatArray,
    psi: float,
    row_weights: FloatArray | None = None,
) -> FloatArray:
    r"""The influence curve :math:`H(A, W)\{Y - \bar Q(A, W)\} + \int_D w\bar Q(\cdot, W) - \psi`.

    Parameters
    ----------
    y : ndarray
        ``(n,)`` outcome.
    q_obs : ndarray
        ``(n,)`` outcome regression at the observed dose.
    h_obs : ndarray
        ``(n,)`` clever covariate :math:`w(A)/g(A \mid W)` at the observed dose.
    integral : ndarray
        ``(n,)`` per-row :math:`\int_D w(a)\bar Q(a, W)\,da`.
    psi : float
        The parameter value.
    row_weights : ndarray or None
        ``(n,)`` observation weights normalised to mean one, or ``None`` for equal weights.

    Returns
    -------
    ndarray
        ``(n,)`` influence curve.
    """
    curve = h_obs * (y - q_obs) + integral - psi
    if row_weights is not None:
        curve = row_weights * curve
    return np.asarray(curve, dtype=float)


def _fluctuate(
    q: FloatArray, h: FloatArray, epsilon: float, kind: Literal["logistic", "linear"], bound: float
) -> FloatArray:
    if kind == "linear":
        return np.asarray(q + epsilon * h, dtype=float)
    clipped = np.clip(q, bound, 1.0 - bound)
    return np.asarray(expit(logit(clipped) + epsilon * h), dtype=float)


def target_pair(
    y: FloatArray,
    q_obs: FloatArray,
    q_nodes: FloatArray,
    h_obs: FloatArray,
    h_nodes: FloatArray,
    pair_weights: FloatArray,
    row_weights: FloatArray,
    *,
    fluctuation: Literal["logistic", "linear"] = "logistic",
    bound: float = 1e-5,
    tol: float = 1e-10,
    max_iter: int = 100,
) -> PairTarget:
    r"""One pooled one-dimensional fluctuation for one pair, and its plug-in and influence curve.

    The submodel is :math:`\mathrm{logit}\,\bar Q_\varepsilon(a, W) = \mathrm{logit}\,\bar Q(a, W)
    + \varepsilon H(a, W)` with :math:`H(a, W) = w(a)/\hat g(a \mid W)`, one function of
    :math:`(a, W)`.  :math:`\varepsilon` solves the weighted score
    :math:`\sum_i \omega_i H(A_i, W_i)\{Y_i - \bar Q_\varepsilon(A_i, W_i)\} = 0` over all rows,
    and the plug-in reads the same function at the quadrature doses.  ``"linear"`` uses
    :math:`\bar Q + \varepsilon H`.

    Parameters
    ----------
    y : ndarray
        ``(n,)`` outcome on the scaled range.
    q_obs : ndarray
        ``(n,)`` initial regression at the observed dose, scaled.
    q_nodes : ndarray
        ``(n, K)`` initial regression at the quadrature doses, scaled.
    h_obs : ndarray
        ``(n,)`` clever covariate at the observed dose.
    h_nodes : ndarray
        ``(n, K)`` clever covariate at the quadrature doses.
    pair_weights : ndarray
        ``(K,)`` the pair's row of :attr:`QuadratureRule.weights`.
    row_weights : ndarray
        ``(n,)`` observation weights.
    fluctuation : {"logistic", "linear"}
        The submodel.
    bound : float
        The clip applied to a scaled regression before the logit.
    tol : float
        Tolerance on the mean weighted score.
    max_iter : int
        Newton steps allowed.

    Returns
    -------
    PairTarget
        The targeted plug-in and influence curve on the scaled outcome.
    """
    omega = np.asarray(row_weights, dtype=float)
    omega = omega / omega.mean()
    epsilon = 0.0
    converged = False
    initial = h_obs * (y - _fluctuate(q_obs, h_obs, 0.0, fluctuation, bound))
    score = float(np.mean(omega * initial))
    # The score need only be small next to its own standard error; an absolute tolerance
    # alone can sit below float precision when the clever covariate is large (a small h).
    tol = max(tol, 1e-6 * float(np.sqrt(np.mean(omega * initial * initial) / y.size)))
    if fluctuation == "linear":
        denominator = float(np.mean(omega * h_obs * h_obs))
        epsilon = score / denominator if denominator > 0.0 else 0.0
        fitted = _fluctuate(q_obs, h_obs, epsilon, fluctuation, bound)
        score = float(np.mean(omega * h_obs * (y - fitted)))
        converged = denominator > 0.0
    else:
        for _ in range(max_iter):
            fitted = _fluctuate(q_obs, h_obs, epsilon, fluctuation, bound)
            score = float(np.mean(omega * h_obs * (y - fitted)))
            if abs(score) <= tol:
                converged = True
                break
            information = float(np.mean(omega * h_obs * h_obs * fitted * (1.0 - fitted)))
            if information <= 0.0:
                break
            step = score / information
            # Halve the step until the absolute score falls, so a far start cannot overshoot.
            for _half in range(40):
                trial = _fluctuate(q_obs, h_obs, epsilon + step, fluctuation, bound)
                trial_score = float(np.mean(omega * h_obs * (y - trial)))
                if abs(trial_score) < abs(score):
                    break
                step *= 0.5
            epsilon += step
        else:
            fitted = _fluctuate(q_obs, h_obs, epsilon, fluctuation, bound)
            score = float(np.mean(omega * h_obs * (y - fitted)))
            converged = abs(score) <= tol
    targeted_obs = _fluctuate(q_obs, h_obs, epsilon, fluctuation, bound)
    targeted_nodes = _fluctuate(q_nodes, h_nodes, epsilon, fluctuation, bound)
    integral = plug_in(targeted_nodes, pair_weights, omega)
    psi = float(np.mean(omega * integral))
    curve = influence_curve(y, targeted_obs, h_obs, integral, psi, omega)
    return PairTarget(psi, curve, float(epsilon), score, converged)


def pair_support(
    h_obs: FloatArray,
    g_obs: FloatArray,
    g_nodes: FloatArray,
    weight_obs: FloatArray,
    pair_weights: FloatArray,
) -> _PairSupport:
    """One pair's support: loaded rows, Kish size, smallest density, largest covariate."""
    loaded = weight_obs != 0.0
    rows = int(loaded.sum())
    absolute = np.abs(h_obs)
    total = float(absolute.sum())
    squares = float((h_obs * h_obs).sum())
    kish = total * total / squares if squares > 0.0 else 0.0
    densities = [g_obs[loaded]] if rows else []
    used = pair_weights != 0.0
    if np.any(used):
        densities.append(g_nodes[:, used].reshape(-1))
    smallest = float(min(d.min() for d in densities if d.size)) if densities else float("nan")
    return _PairSupport(rows, kish, smallest, float(absolute.max()) if absolute.size else 0.0)
