r"""Simultaneous confidence bands via the multiplier bootstrap.

Reporting several estimands with a 95% interval each does not give 95% confidence
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

No resampling of the data is involved, so this costs one matrix product per block
of replicates -- cheap enough to run by default.

Where the time goes, and why the block is sized in bytes
--------------------------------------------------------

A profile of this path reads "92-95% multiplier generation", which points at the
random draw and is misleading.  Split at ``n = 100,000`` with a 256-replicate block:

============================================  =========  ======
step                                          ms         share
============================================  =========  ======
``rng.integers`` -- *the seeded draw itself*      3.5      2%
``np.unpackbits`` -- bits to ``uint8``            1.7      1%
expanding those bits to a float64 matrix        159.0     89%
``xi @ centred`` -- the ``dgemm``                12.6      7%
============================================  =========  ======

Drawing the bits is 2% of it.  Turning one bit per element into a float64 -- and
writing a 205 MB array to hold the result -- is the other 89%, and both halves of
that are avoidable without touching the draw.

So the multipliers are expanded **in place, into a preallocated buffer**, with the
block sized by *bytes* rather than by replicates: the optimum tracks the buffer's
footprint against the cache and the allocator, not the replicate count, and a fixed
block that is comfortable at ``n = 10,000`` allocates two gigabytes at
``n = 1,000,000``.  Measured at ``n = 100,000``, ``B = 512``: 292 ms at a 256-replicate
block (205 MB), 135 ms at 64 (51 MB), 258 ms at 4 (3 MB).  At ``n = 1,000,000`` the
same curve peaks at an 8-replicate block, which is the same footprint again.

**The block size does not change the draw.**  ``rng.integers(0, 256, ..., uint8)``
fills from buffered 32-bit words, so a block that is a multiple of four consumes a
whole number of words and leaves the byte stream where the next block picks it up:
every block hands replicate ``b`` the same multipliers.  That is a property of numpy's
filling and not a promise it makes, so
:func:`tests.unit.test_inference.TestSimultaneousBands.test_the_block_size_does_not_change_the_multipliers`
pins it rather than this docstring asserting it.

The *critical value* then follows to rounding rather than exactly, because ``xi @
centred`` is a ``dgemm`` whose blocking depends on its operand shape and the sum over
``n`` can be accumulated in a different order at a different block.  Measured at a
relative 1e-15 -- far below what a Monte Carlo quantile means, and worth stating
rather than papering over.

The remaining float64 expansion is the one cost that is still there to take: doing it
at ``float32`` measured 7.2x against the shipped path rather than 1.9x, at a
disagreement of 1e-6 on a critical value whose own resampling error is 1e-2.  That is
a change to the arithmetic of a reported quantity and wants its own argument, so it is
not made here; ``docs/benchmarks/bootstrap_numpy.md`` records the measurement.

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

#: Target footprint of the multiplier buffer, in bytes.  The block is derived from this
#: and ``n`` rather than fixed, because what the timing tracks is the footprint: see the
#: module docstring for the measured curve at two sample sizes.
_BLOCK_BYTES = 32 << 20

#: Bounds on the derived block.  The lower one keeps the ``dgemm`` wide enough to be worth
#: calling at very large ``n``; the upper one keeps the buffer from growing past the target
#: at small ``n``, where the whole thing fits in cache anyway.  Both are multiples of four,
#: which is what keeps the drawn byte stream independent of the block -- see the docstring.
_MIN_BLOCK = 4
_MAX_BLOCK = 256


def _block_size(n_rows: int, n_replicates: int) -> int:
    """Replicates per pass: a byte budget, clamped, rounded down to a multiple of four."""
    wanted = _BLOCK_BYTES // max(1, n_rows * 8)
    block = min(_MAX_BLOCK, max(_MIN_BLOCK, int(wanted)))
    block -= block % 4
    return max(_MIN_BLOCK, min(block, max(1, n_replicates)))


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


#: Mammen's two-point distribution: mean 0, variance 1, third moment 1, which matches one
#: more moment of the sampling distribution than Rademacher does.
_ROOT5 = np.sqrt(5.0)
_MAMMEN_PROB = (_ROOT5 + 1.0) / (2.0 * _ROOT5)
_MAMMEN_LOW = -(_ROOT5 - 1.0) / 2.0
_MAMMEN_HIGH = (_ROOT5 + 1.0) / 2.0


def _fill_multipliers(rng: np.random.Generator, out: FloatArray, kind: MultiplierKind) -> None:
    """Fill ``out`` with mean-zero, unit-variance multipliers, allocating nothing large.

    The values are exactly those :func:`_multipliers` returns for the same generator
    state -- ``+-1`` and Mammen's two points are representable, so "in place" costs no
    accuracy -- and the point is what is *not* allocated: the caller's buffer is written
    over, and the intermediates are one packed byte per eight elements and one bool per
    element rather than a second and third float64 array the size of the first.
    """
    if kind == "rademacher":
        # A Rademacher draw carries one bit, so generate bits rather than a float64
        # uniform per element: packed bytes plus np.unpackbits measures about 2.4x
        # faster than comparing rng.random(shape) against 0.5.  Then widen in place --
        # a sign lookup indexed by the bit is a 25-million-element gather that writes a
        # whole second array, and is 89% of this path's cost where the widen-and-rescale
        # in place is a third of that.
        rows, columns = out.shape
        packed = rng.integers(0, 256, size=(rows, (columns + 7) // 8), dtype=np.uint8)
        np.copyto(out, np.unpackbits(packed, axis=1, count=columns))
        out *= 2.0
        out -= 1.0
        return
    if kind == "normal":
        rng.standard_normal(out=out)
        return
    if kind == "mammen":
        rng.random(out=out)
        flags = out < _MAMMEN_PROB
        np.copyto(out, _MAMMEN_HIGH)
        np.copyto(out, _MAMMEN_LOW, where=flags)
        return
    raise ValueError(f"kind must be 'rademacher', 'mammen' or 'normal'; got {kind!r}")


def _multipliers(
    rng: np.random.Generator, shape: tuple[int, int], kind: MultiplierKind
) -> FloatArray:
    """Mean-zero, unit-variance multipliers, as an array.

    Not what the critical value uses -- that fills a reused buffer through
    :func:`_fill_multipliers` and never holds a ``(B, n)`` array at all.  This is the
    same draw with an allocation around it, kept because a caller inspecting the
    multipliers themselves wants them materialised.
    """
    out = np.empty(shape, dtype=float)
    _fill_multipliers(rng, out, kind)
    return out


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
        statistics = _two_point_statistics(
            centred, se, usable, n=n, n_replicates=n_replicates, kind=kind, rng=rng
        )
    return float(np.quantile(statistics, 1.0 - alpha))


def _two_point_statistics(
    centred: FloatArray,
    se: FloatArray,
    usable: BoolArray,
    *,
    n: int,
    n_replicates: int,
    kind: MultiplierKind,
    rng: np.random.Generator,
) -> FloatArray:
    """Max-t draws for two-point multipliers, one reused buffer at a time.

    The buffer is allocated once and written over per block, so the largest thing this
    holds is ``block x n`` doubles -- a bounded footprint by construction, where the
    replicate-indexed version grew with ``n`` until it was the first thing to break at
    scale.  See the module docstring for how the block is chosen and why it does not
    move the answer.
    """
    rows = centred.shape[0]
    block = _block_size(rows, n_replicates)
    buffer = np.empty((block, rows), dtype=float)
    statistics = np.empty(n_replicates, dtype=float)
    scale = se[usable]
    done = 0
    while done < n_replicates:
        size = min(block, n_replicates - done)
        xi = buffer[:size]
        _fill_multipliers(rng, xi, kind)
        # The matmul spans every estimand rather than the usable ones: each output column
        # is its own dot product, so restricting first would be the same arithmetic, but
        # it would hand BLAS a different shape and the last bits are a regression pin.
        draws = (xi @ centred) / n
        standardised = np.abs(draws[:, usable]) / scale
        statistics[done : done + size] = standardised.max(axis=1)
        done += size
    return statistics


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
