r"""Simultaneous confidence bands via the multiplier bootstrap.

Reporting seven estimands with seven 95% intervals does not give 95% confidence
that *all seven* contain their truth -- with correlated estimates the joint
coverage is somewhere below that, and the usual fixes (Bonferroni) are needlessly
wide precisely because the estimates are correlated.

The multiplier bootstrap uses the influence curves, which are already available,
to get the joint distribution right.  Draw i.i.d. mean-zero, unit-variance
multipliers :math:`\xi_i` and form

.. math::

    R^*_{b,j} = \frac{1}{n}\sum_i \xi_{b,i}
                \bigl(\mathrm{IC}_{i,j} - \overline{\mathrm{IC}}_j\bigr),
    \qquad
    t^*_b = \max_j \frac{|R^*_{b,j}|}{\widehat{se}_j}.

The :math:`1 - \alpha` quantile :math:`\hat c` of :math:`t^*` gives bands
:math:`\hat\psi_j \pm \hat c \cdot \widehat{se}_j` with asymptotic *joint* coverage
:math:`1 - \alpha`.  Because :math:`\hat c` adapts to the observed correlation, it
sits between the pointwise critical value (1.96) and the Bonferroni one, and
collapses to 1.96 when only one estimand is requested.

No resampling of the data is involved, so this costs one matrix product per chunk
of replicates -- cheap enough to run by default.

Gaussian multipliers need no resampling at all
----------------------------------------------

For ``kind="normal"`` the draws are a linear map of a Gaussian, so

.. math::

    R^*_b = \frac{1}{n} \xi_b^\top C
    \;\sim\; N\!\left(0, \; \frac{C^\top C}{n^2}\right),
    \qquad C = \mathrm{IC} - \overline{\mathrm{IC}},

*exactly*.  Sampling that :math:`m`-dimensional normal directly costs
:math:`O(n m^2 + B m^2)` instead of :math:`O(B n m)` and never allocates a
``(B, n)`` array -- at n=100,000 the critical value takes about 5 ms instead of
1.5 s.

Why that speed is not free, and why the default is not ``"normal"``
------------------------------------------------------------------

The closed form exists *because* the Gaussian max-t law depends on the influence
curves only through their covariance.  Everything above second moments is discarded,
so ``kind="normal"`` cannot distinguish a heavy-tailed influence curve from a
Gaussian one with the same covariance -- it is a plug-in normal approximation, not a
resampling scheme, which is precisely why it is orders of magnitude cheaper.

The two-point multipliers keep that information: conditional on the data the draws
have the same covariance but their higher conditional moments depend on the actual
:math:`C_i`, so influential observations still register.  That matters here, because
a TMLE influence curve carries :math:`1/g(W)` and a positivity problem gives it
exactly the leverage the Gaussian approximation smooths away.  Simulated against a
brute-force max-t distribution, with a clever covariate under weak overlap:

===========  ======  ===============  ================  ==============
regime       n       truth            ``"normal"``      ``"rademacher"``
===========  ======  ===============  ================  ==============
gaussian     2,000   2.3193           +0.013 (95.2%)    +0.014 (95.2%)
overlap        200   2.1637           +0.139 (96.7%)    -0.019 (94.0%)
overlap      2,000   2.2539           +0.070 (95.9%)    -0.017 (94.7%)
===========  ======  ===============  ================  ==============

Under a well-behaved influence curve the three kinds are interchangeable.  Under
leverage ``"normal"`` is biased conservative -- bands several percent wider than the
data warrant -- and the bias does not wash out by n = 2,000.

So ``"rademacher"`` is the default: it tracks the truth most closely, and its minimal
fourth moment (:math:`E\xi^4 = 1`, against 3 for a Gaussian) minimises the
variability of the bootstrap distribution.  ``"mammen"`` additionally matches
:math:`E\xi^3 = 1`, the term an Edgeworth expansion wants, and sits between the two.
Reach for ``"normal"`` when ``n`` is large, the influence curves are well behaved, and
the resampling cost actually shows up in a profile -- it is an opt-in speed trade,
not a free one.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .._typing import BoolArray, FloatArray, IntArray
from .cluster import cluster_sums
from .influence import ParameterEstimate

__all__ = ["SimultaneousBands", "multiplier_critical_value", "simultaneous_bands"]

MultiplierKind = Literal["rademacher", "mammen", "normal"]

# Replicates are generated in chunks so the multiplier matrix stays small even for
# large n and many replicates.
_CHUNK = 256


@dataclass(frozen=True)
class SimultaneousBands:
    """Joint confidence bands across several estimands."""

    critical_value: float
    bands: dict[str, tuple[float, float]]
    alpha: float
    n_replicates: int
    kind: str

    @property
    def pointwise_critical_value(self) -> float:
        """The critical value a single-estimand interval would have used."""
        from scipy import stats

        return float(stats.norm.ppf(1.0 - self.alpha / 2.0))


#: Sign lookup for the two-point multipliers, indexed by a 0/1 draw.
_SIGNS = np.array([-1.0, 1.0])


def _multipliers(
    rng: np.random.Generator, shape: tuple[int, int], kind: MultiplierKind
) -> FloatArray:
    """Mean-zero, unit-variance multipliers."""
    if kind == "rademacher":
        # A Rademacher draw carries one bit, so generate bits rather than a float64
        # uniform per element: packed bytes plus np.unpackbits measures about 2.4x
        # faster than comparing rng.random(shape) against 0.5.
        rows, columns = shape
        packed = rng.integers(0, 256, size=(rows, (columns + 7) // 8), dtype=np.uint8)
        return _SIGNS[np.unpackbits(packed, axis=1, count=columns)]
    if kind == "normal":
        return rng.standard_normal(shape)
    if kind == "mammen":
        # Mammen's two-point distribution: mean 0, variance 1, third moment 1, which
        # matches one more moment of the sampling distribution than Rademacher does.
        root5 = np.sqrt(5.0)
        prob = (root5 + 1.0) / (2.0 * root5)
        low = -(root5 - 1.0) / 2.0
        high = (root5 + 1.0) / 2.0
        return np.where(rng.random(shape) < prob, low, high)
    raise ValueError(f"kind must be 'rademacher', 'mammen' or 'normal'; got {kind!r}")


def multiplier_critical_value(
    influence_curves: FloatArray,
    std_errors: FloatArray,
    *,
    n: int,
    cluster: IntArray | None = None,
    alpha: float = 0.05,
    n_replicates: int = 1000,
    kind: MultiplierKind = "rademacher",
    random_state: int | None = None,
) -> float:
    """The ``1 - alpha`` quantile of the max-t statistic."""
    ic = np.asarray(influence_curves, dtype=float)
    if ic.ndim != 2:
        raise ValueError(f"expected an (n, m) influence-curve matrix; got shape {ic.shape}")
    se = np.asarray(std_errors, dtype=float).reshape(-1)
    if se.shape[0] != ic.shape[1]:
        raise ValueError(
            f"got {ic.shape[1]} influence curve(s) but {se.shape[0]} standard error(s)"
        )
    if ic.shape[1] == 1:
        from scipy import stats

        return float(stats.norm.ppf(1.0 - alpha / 2.0))

    scores = cluster_sums(ic, cluster) if cluster is not None else ic
    centred = scores - scores.mean(axis=0, keepdims=True)
    usable = np.isfinite(se) & (se > 0)
    if not usable.any():
        return float("nan")

    rng = np.random.default_rng(random_state)
    if kind == "normal":
        statistics = _normal_statistics(
            centred, se, usable, n=n, n_replicates=n_replicates, rng=rng
        )
    else:
        statistics = np.empty(n_replicates, dtype=float)
        done = 0
        while done < n_replicates:
            size = min(_CHUNK, n_replicates - done)
            xi = _multipliers(rng, (size, centred.shape[0]), kind)
            draws = (xi @ centred) / n
            standardised = np.abs(draws[:, usable]) / se[usable]
            statistics[done : done + size] = standardised.max(axis=1)
            done += size
    return float(np.quantile(statistics, 1.0 - alpha))


def _normal_statistics(
    centred: FloatArray,
    se: FloatArray,
    usable: BoolArray,
    *,
    n: int,
    n_replicates: int,
    rng: np.random.Generator,
) -> FloatArray:
    """Max-t draws for Gaussian multipliers, sampled from their exact distribution.

    ``xi @ centred`` is a linear map of a standard normal, so the replicate vector is
    exactly ``N(0, centred.T @ centred / n^2)``.  Drawing from that ``m``-dimensional
    normal is the same distribution as resampling, without the ``(n_replicates, n)``
    multiplier matrix.

    The covariance is formed from ``centred`` directly rather than via
    :func:`~cleverly.inference.cluster.influence_covariance`, which normalises by the
    number of clusters where this needs the raw cross-product.

    It is factorised with a symmetric eigendecomposition rather than a Cholesky, because
    the covariance is routinely *singular*: the estimands are functionally related, and
    ``IC_ate == IC_ey1 - IC_ey0`` holds exactly, so the default estimand set already
    produces a rank-deficient matrix.  That is a property of the parameters, not a
    numerical problem -- the max-t distribution is still perfectly well defined on the
    lower-dimensional support, and the resampling path handles it without complaint
    because it never factorises anything.
    """
    covariance = (centred.T @ centred) / (float(n) ** 2)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    # Clip the small negative eigenvalues that rounding puts on a singular PSD matrix.
    factor = eigenvectors * np.sqrt(np.clip(eigenvalues, 0.0, None))
    draws = rng.standard_normal((n_replicates, covariance.shape[0])) @ factor.T
    standardised = np.abs(draws[:, usable]) / se[usable]
    return np.asarray(standardised.max(axis=1), dtype=float)


def simultaneous_bands(
    estimates: Mapping[str, ParameterEstimate] | Sequence[ParameterEstimate],
    *,
    alpha: float = 0.05,
    n_replicates: int = 1000,
    kind: MultiplierKind = "rademacher",
    random_state: int | None = None,
    cluster: IntArray | None = None,
) -> SimultaneousBands:
    """Joint confidence bands over a set of estimates.

    Bands are reported on each estimand's own scale, so a ratio's band is
    exponentiated from the log scale exactly as its pointwise interval is.
    """
    items = (
        list(estimates.items())
        if isinstance(estimates, Mapping)
        else [(estimate.name, estimate) for estimate in estimates]
    )
    if not items:
        raise ValueError("no estimates supplied")
    lengths = {estimate.influence_curve.shape[0] for _, estimate in items}
    if len(lengths) != 1:
        raise ValueError(f"influence curves have inconsistent lengths: {lengths}")

    n = items[0][1].n
    ic = np.column_stack([estimate.influence_curve for _, estimate in items])
    se = np.array([estimate.std_error for _, estimate in items])
    critical = multiplier_critical_value(
        ic,
        se,
        n=n,
        cluster=cluster,
        alpha=alpha,
        n_replicates=n_replicates,
        kind=kind,
        random_state=random_state,
    )

    bands: dict[str, tuple[float, float]] = {}
    for name, estimate in items:
        half_width = critical * estimate.std_error
        if estimate.scale == "ratio":
            assert estimate.log_psi is not None
            bands[name] = (
                float(np.exp(estimate.log_psi - half_width)),
                float(np.exp(estimate.log_psi + half_width)),
            )
        else:
            bands[name] = (estimate.psi - half_width, estimate.psi + half_width)
    return SimultaneousBands(
        critical_value=critical,
        bands=bands,
        alpha=alpha,
        n_replicates=n_replicates,
        kind=kind,
    )
