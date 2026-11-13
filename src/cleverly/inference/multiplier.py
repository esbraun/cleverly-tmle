r"""Simultaneous confidence bands via the multiplier bootstrap.

Reporting several estimands with a 95% interval each does not give 95% confidence
that *all* of them contain their truth: with correlated estimates the joint coverage is
somewhere below that, and the usual fix (Bonferroni) is needlessly wide precisely because
the estimates are correlated.

The multiplier bootstrap uses the influence curves, which are already available,
to get the joint distribution right.  Draw i.i.d. mean-zero, unit-variance
multipliers :math:`\xi_i` and form

.. math::

    R^*_{b,j} = \frac{1}{n}\sum_i \xi_{b,i}
                \bigl(\mathrm{IC}_{i,j} - \overline{\mathrm{IC}}_j\bigr),
    \qquad
    t^*_b = \max_j \frac{|R^*_{b,j}|}{\widehat{se}_j}.

The :math:`1 - \alpha` quantile :math:`\hat c` of :math:`t^*` is the critical value.
Because :math:`\hat c` adapts to the observed correlation, it sits between the pointwise
critical value (1.96) and the Bonferroni one, and collapses to 1.96 when only one estimand
is requested.  No resampling of the data is involved, so this costs one matrix product per
block of replicates.

The default band
----------------

``simultaneous_bands()``, ``median_bands()`` and every estimator with a band default to
``band_scale="logit"``, ``kind="mammen"`` and ``n_replicates=10_000`` (``n_multiplier`` on
an estimator).  The band-finite-sample review of the six-regimen ``longitudinal-sdr`` cell
chose them; ``docs/technical-reference/inference.md`` gives the evidence and which rows
measure them.

**Why the transform.**  A band :math:`\hat\psi_j \pm \hat c\,\widehat{se}_j` is a Wald
interval at a larger critical value, and for a probability its standard error moves with
its estimate: an estimate above 0.5 that comes out high has a small standard error, so the
interval misses on one side more often than the nominal rate.  At the band's critical value
that tail is where joint coverage is lost.  Under ``band_scale="logit"`` an estimate with
known :attr:`~cleverly.inference.ParameterEstimate.bounds` :math:`(a, b)` is banded on
:math:`f(\psi) = \operatorname{logit}((\psi - a) / (b - a))`, which is the logit of a
probability and the ``atanh`` of a difference of two probabilities, and mapped back.  The
band stays inside the parameter's range, and its limits are no longer symmetric around the
estimate.  Every other estimand keeps its own inference scale (the log scale for a ratio).
A bounded estimate on or outside its range with a positive standard error has no logit
band: the functions here raise, and a fit skips its band and says why in ``summary()``.
Pointwise intervals stay on the identity scale.

**Why the critical value is the same on either scale.**  The delta method carries column
:math:`j` to the transformed scale by multiplying its influence curve and its standard
error by the same factor :math:`f'(\hat\psi_j)`.  The max-t statistic divides one by the
other, so its law, and :math:`\hat c`, do not change.  Only the band limits move.

**Why Mammen.**  First-order theory ties the multiplier kinds: Chernozhukov, Chetverikov
and Kato (2013, Theorems 3.1 and 3.2; 2017, Corollary 4.2) and Chernozhukov, Chetverikov,
Kato and Koike (2022, Theorem 2.2) give one bound for every multiplier that meets their
moment condition.  Mammen's two-point law also matches :math:`E\xi^3 = 1`.  Deng and Zhang
(2020) prove a faster rate for a wild bootstrap with that third moment, and their
simulations on skewed data rank Mammen first.  The influence curve of a binomial outcome is
skewed.  No theorem ranks the kinds for this studentized statistic, so Mammen is a default
chosen between first-order equivalent kinds, and ``"rademacher"`` stays available.

**Why 10,000 draws.**  At 1,000 or 2,000 draws the Monte Carlo error of a 95% quantile of
:math:`t^*` is a visible part of a band's coverage error.  The faster Mammen fill below
keeps 10,000 draws affordable.

**Why no bootstrap-t.**  The standard error is held fixed in every draw.  A bootstrap-t
that re-estimates it per draw on the identity scale calibrates a statistic the band does
not invert, and none of the results cited above analyses a maximum with the standard
error re-estimated in each resample.  The bootstrap-t of the transformed
statistic was computed on the registered draws of the ``longitudinal-sdr`` cell and
covered less than the fixed-SE band.  A resample can also omit every unit that carries a
sparse column, which makes its critical value unbounded; a multiplier keeps every unit in
every draw.  Hall (1986) notes that a fixed-variance bootstrap leaves a one-sided term of
order :math:`n^{-1/2}`, so this module claims first-order validity only.

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

The remaining float64 expansion could be reduced with lower-precision arithmetic, but that would
change a reported quantity and requires independent numerical and statistical acceptance evidence;
it is deliberately not done here.

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
-------------------------------------------------------------------

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

So the default is a two-point kind.  ``"rademacher"`` has the minimal fourth moment
(:math:`E\xi^4 = 1`, against 3 for a Gaussian), which minimises the variability of the
bootstrap distribution.  ``"mammen"`` additionally matches :math:`E\xi^3 = 1`, the term an
Edgeworth expansion wants, which is why it is the default (above).
Reach for ``"normal"`` when ``n`` is large, the influence curves are well behaved, and
the resampling cost actually shows up in a profile -- it is an opt-in speed trade,
not a free one.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .._summary import Detail, Summary

from .._inference_status import (
    CONSTANT_ARM_CELL_STATUS,
    FEW_CLUSTER_THRESHOLD,
    NO_CONSTANT_ARM_BANDS,
    OUT_OF_RANGE_STATUS,
)
from .._support import UNCLASSIFIED, Support, combine_support
from .._typing import BoolArray, FloatArray, IntArray
from ..exceptions import CapabilityError, Refusal, RefusalReason, refuse_inference
from .cluster import cluster_sums
from .delta import Transform
from .influence import ParameterEstimate, with_precedent_status

#: Why no band covers an estimate with a Student t reference.
T_REFERENCE_BAND_REFUSAL = Refusal(
    RefusalReason.INFERENCE_GAP,
    "simultaneous_bands() has no joint result for an estimate with a Student t reference. "
    f"The fit reads fewer than {FEW_CLUSTER_THRESHOLD} clusters with positive weight mass, "
    "and no source read here gives a simultaneous band at that count.",
    alternative="Report the pointwise intervals, which use the t reference.",
    roadmap="F28",
)

#: Why no band covers the known-regime values or contrasts of a learned-rule fit (X11 (c2)).
KNOWN_REGIME_BAND_REFUSAL = Refusal(
    RefusalReason.INFERENCE_GAP,
    "simultaneous_bands() has no joint result over a learned-rule value, the known-regime "
    "values beside it (ey_known_regime) and their contrasts (ate_learned_rule). Joint "
    "normality of the stack gives each contrast's own interval, and a band over a "
    "data-adaptive value and fixed regimes has no witness (X11 (c2) in docs/roadmap.md).",
    alternative=(
        "Report the pointwise intervals, or pass simultaneous_bands() the estimates without "
        "the ey_known_regime and ate_learned_rule ones."
    ),
    roadmap="X11",
)


def constant_arm_band_refusal(names: Sequence[str]) -> Refusal:
    """Why no band covers a family that holds an estimate with a constant arm cell.

    Parameters
    ----------
    names : sequence of str
        The members whose status is :data:`CONSTANT_ARM_CELL_STATUS`.

    Returns
    -------
    Refusal
        The ``inference_gap`` refusal :func:`simultaneous_bands` and :func:`median_bands`
        raise, whose condition is the line a fit's summary prints.
    """
    return Refusal(
        RefusalReason.INFERENCE_GAP,
        NO_CONSTANT_ARM_BANDS.format(names=", ".join(names)),
        alternative=(
            "Report the pointwise estimates; a band needs two observed outcome values in "
            "every treatment arm of every stratum its estimates read."
        ),
    )


__all__ = [
    "DEFAULT_BAND_SCALE",
    "DEFAULT_KIND",
    "DEFAULT_N_REPLICATES",
    "BandScale",
    "SimultaneousBands",
    "band_member",
    "constant_arm_band_refusal",
    "constant_arm_members",
    "median_bands",
    "multiplier_critical_value",
    "out_of_range",
    "simultaneous_bands",
]

MultiplierKind = Literal["rademacher", "mammen", "normal"]

#: The scale a band is built on.  ``"logit"`` bands a parameter with known
#: :attr:`~cleverly.inference.ParameterEstimate.bounds` on the logit of its position in that
#: range; ``"identity"`` bands every parameter on its own inference scale.
BandScale = Literal["logit", "identity"]

_BAND_SCALES: tuple[str, ...] = ("logit", "identity")

#: The default band: the scale, the multiplier kind and the draws every band function and
#: estimator uses unless told otherwise (the module docstring says why).
DEFAULT_BAND_SCALE: BandScale = "logit"
DEFAULT_KIND: MultiplierKind = "mammen"
DEFAULT_N_REPLICATES: int = 10_000

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
    """Joint confidence bands across several estimands.

    Parameters
    ----------
    critical_value : float
        Multiplier the pointwise standard errors are scaled by.
    bands : dict of str to tuple of float
        One joint interval per estimand alias.
    alpha : float
        Joint significance level the bands hold at.
    n_replicates : int
        Multiplier draws behind the critical value.
    kind : str
        Distribution the multiplier weights were drawn from.
    draw_critical_values : tuple of float
        On a repeated longitudinal fit, the critical value of each draw of the
        cross-fitting split, whose median is :attr:`critical_value`. Empty otherwise.
    support : Support, default=UNCLASSIFIED
        The combination of the support of the estimates the band covers. A fit narrows it
        further when its row licenses no ``band`` kind for the fit.
    band_scale : {"logit", "identity"}, default="identity"
        The scale the bands were requested on.
    scales : dict of str to str, default={}
        The scale each estimand's band was built on: ``"logit"`` for a parameter in
        ``(0, 1)``, ``"range_logit"`` for one in another known range, ``"log"`` for a ratio,
        the name of an explicit transform, or ``"identity"``.
    """

    critical_value: float
    bands: dict[str, tuple[float, float]]
    alpha: float
    n_replicates: int
    kind: str
    draw_critical_values: tuple[float, ...] = ()
    support: Support = UNCLASSIFIED
    band_scale: str = "identity"
    scales: dict[str, str] = field(default_factory=dict)

    @property
    def pointwise_critical_value(self) -> float:
        """The critical value a single-estimand interval would have used."""
        from scipy import stats

        return float(stats.norm.ppf(1.0 - self.alpha / 2.0))

    def to_frame(self, backend: str | None = None) -> Any:
        """One row per estimand: the lower and upper limit of its band.

        Parameters
        ----------
        backend : str or None, default=None
            The dataframe library to return, such as ``"pandas"`` or ``"polars"``; ``None``
            uses the default one.

        Returns
        -------
        dataframe
            The columns ``estimand``, ``lower``, ``upper`` and ``scale`` (the scale each band
            was built on), in ``backend``.
        """
        from ..utils.frames import frame_from_dict

        return frame_from_dict(
            {
                "estimand": list(self.bands),
                "lower": [float(low) for low, _ in self.bands.values()],
                "upper": [float(high) for _, high in self.bands.values()],
                "scale": [self.scales.get(name, "identity") for name in self.bands],
            },
            backend=backend,
        )

    def summary(self, *, detail: Detail = "short") -> Summary:
        """A printable report of the bands, laid out like a statsmodels summary.

        Parameters
        ----------
        detail : {"short", "full"}, default "short"
            ``"full"`` adds the critical value a single interval would use.

        Returns
        -------
        Summary
            A ``str`` whose value is the printed text, with the parts as data.
        """
        from .._summary import check_detail
        from ._summaries import bands_summary

        return bands_summary(self, check_detail(detail))

    def __repr__(self) -> str:
        from ._summaries import bands_repr

        return bands_repr(self)

    def _repr_html_(self) -> str:
        return self.summary().as_html()


#: Mammen's two-point distribution: mean 0, variance 1, third moment 1, which matches one
#: more moment of the sampling distribution than Rademacher does.
_ROOT5 = np.sqrt(5.0)
_MAMMEN_PROB = (_ROOT5 + 1.0) / (2.0 * _ROOT5)
_MAMMEN_LOW = -(_ROOT5 - 1.0) / 2.0
_MAMMEN_HIGH = (_ROOT5 + 1.0) / 2.0
#: The integer threshold a uint32 draw falls below with probability :data:`_MAMMEN_PROB`, to
#: within ``2**-32``.
_MAMMEN_THRESHOLD = np.uint32(round(_MAMMEN_PROB * 2**32))


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
        # One uint32 per element against an integer threshold, rather than a float64 uniform:
        # half the bytes drawn, and the comparison writes a bool array the size of the
        # buffer instead of a second float64 one.  The threshold moves the low point's
        # probability by less than 2**-32 (the exact moments are pinned in the tests).
        flags = rng.integers(0, 2**32, size=out.shape, dtype=np.uint32) < _MAMMEN_THRESHOLD
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
    n_replicates: int,
    kind: MultiplierKind,
    random_state: int | None = None,
) -> float:
    """The ``1 - alpha`` quantile of the max-t statistic.

    Parameters
    ----------
    influence_curves : ndarray
        ``(n, k)`` influence curves, one column per estimand.
    std_errors : ndarray
        ``(k,)`` pointwise standard errors.
    n : int
        Number of observations.
    cluster : ndarray or None
        ``(n,)`` cluster codes. Weights are drawn per cluster when given.
    alpha : float
        Joint significance level.
    n_replicates : int
        Multiplier draws. Required, so a change of the band defaults never reaches a
        caller that does not name it.
    kind : {"rademacher", "mammen", "normal"}
        Distribution of the multiplier weights. Required for the same reason.
    random_state : int or None
        Seed for those draws.

    Returns
    -------
    float
        The ``1 - alpha`` quantile of the max-t statistic across estimands. Rescaling one
        column of ``influence_curves`` and its standard error by the same nonzero factor
        leaves it unchanged, so a band built on a delta-method transform of an estimand
        reads the same critical value.
    """
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
    n_replicates: int = DEFAULT_N_REPLICATES,
    kind: MultiplierKind = DEFAULT_KIND,
    random_state: int | None = None,
    cluster: IntArray | None = None,
    band_scale: BandScale = DEFAULT_BAND_SCALE,
) -> SimultaneousBands:
    """Joint confidence bands over a set of estimates.

    Bands are reported on each estimand's own scale, so a ratio's band is
    exponentiated from the log scale exactly as its pointwise interval is.

    With ``band_scale="logit"``, an estimate with known
    :attr:`~cleverly.inference.ParameterEstimate.bounds` :math:`(a, b)` is banded on
    :math:`f(\\psi) = \\operatorname{logit}((\\psi - a) / (b - a))`: the band is
    :math:`f^{-1}(f(\\hat\\psi) \\pm \\hat c\\, \\widehat{se}\\, f'(\\hat\\psi))`. A
    probability gets the logit band and a difference of two probabilities the ``atanh``
    band. The critical value :math:`\\hat c` does not depend on the scale, because the
    max-t statistic is invariant to rescaling one column of the curves together with its
    standard error.

    Parameters
    ----------
    estimates : mapping or sequence of ParameterEstimate
        The estimates to band jointly.
    alpha : float
        Joint significance level.
    n_replicates : int, default=10_000
        Multiplier draws.
    kind : {"rademacher", "mammen", "normal"}, default="mammen"
        Distribution of the multiplier weights.
    random_state : int or None
        Seed for those draws.
    cluster : ndarray or None
        ``(n,)`` cluster codes. Weights are drawn per cluster when given.
    band_scale : {"logit", "identity"}, default="logit"
        The scale of the band of a bounded estimate. Every other estimate keeps its own
        inference scale.

    Returns
    -------
    SimultaneousBands
        One interval per estimand, holding jointly at ``1 - alpha``.

    Raises
    ------
    ValueError
        If ``estimates`` is empty, or if the influence curves have different lengths, or
        if ``band_scale`` is not ``"logit"`` or ``"identity"``.
    ValueError
        If ``cluster`` has a different length from the influence curves, or if ``kind``
        is not a supported multiplier, when two or more estimates are banded.
    CapabilityError
        If any estimate's ``supplies_inference`` is ``False``. A simultaneous band is
        a confidence statement, and the package supplies no inference for such an
        estimate. Also if any estimate carries a Student t reference
        (:attr:`~cleverly.inference.ParameterEstimate.reference_df`), because no source
        gives a t-calibrated band. Also, with code ``inference_gap``, if any estimate
        declares ``covariance_rule="second_moment"``. The multiplier draws center each
        influence curve, which matches the raw second moment only on a mean-zero curve,
        and this function does not check the mean. Also, with code ``inference_gap``, if
        any estimate is a known-regime value (``ey_known_regime``) or a contrast of the
        learned rule with a known regime (``ate_learned_rule``): no joint result covers a
        band over a data-adaptive value and fixed regimes (X11 (c2)). Also, with code
        ``inference_gap``, under ``band_scale="logit"``, if a bounded estimate with a
        positive standard error lies on or outside its range, where the logit band is not
        defined.
    """
    _check_band_scale(band_scale)
    items = _checked_items(estimates)
    _refuse_out_of_range(items, band_scale)
    n = items[0][1].n
    ic = np.column_stack([estimate.influence_curve for _, estimate in items])
    se = np.array([_band_std_error(estimate) for _, estimate in items])
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
    bands, scales = _band_endpoints(items, critical, band_scale)
    return SimultaneousBands(
        critical_value=critical,
        bands=bands,
        alpha=alpha,
        n_replicates=n_replicates,
        kind=kind,
        band_scale=band_scale,
        scales=scales,
    )


def constant_arm_members(
    estimates: Mapping[str, ParameterEstimate] | Sequence[tuple[str, ParameterEstimate]],
) -> tuple[str, ...]:
    """The members of a band's family whose status is ``"constant_arm_cell_plugin"``.

    Such an estimate reads a treatment arm whose observed outcomes in its stratum take a single
    value, so its standard error leaves that arm's variance out and no band covers it.  A band
    over a family that holds one is refused rather than built over the others, because that
    band would be a statement about a family the data chose.

    Parameters
    ----------
    estimates : mapping of str to ParameterEstimate, or sequence of pairs
        The band's family, before any member is filtered out.

    Returns
    -------
    tuple of str
        The names, in order.
    """
    pairs = estimates.items() if isinstance(estimates, Mapping) else estimates
    return tuple(
        name for name, estimate in pairs if str(estimate.inference) == CONSTANT_ARM_CELL_STATUS
    )


def band_member(estimate: ParameterEstimate) -> bool:
    """Whether an estimate belongs to a fit's band family.

    An estimate that supplies inference does, and so does a bounded estimate on or outside its
    range (status ``"out_of_range_plugin"``), which keeps the band behaviour it had before the
    status: the logit band refuses it and the identity band covers it.  Every other withheld
    estimate leaves the band.

    Parameters
    ----------
    estimate : ParameterEstimate
        A reported estimate.

    Returns
    -------
    bool
        ``True`` for a member.
    """
    return bool(estimate.supplies_inference) or str(estimate.inference) == OUT_OF_RANGE_STATUS


def _band_std_error(estimate: ParameterEstimate) -> float:
    """The standard error a band reads: the body :attr:`ParameterEstimate.std_error` returns.

    The same arithmetic on the same curve, so the band of an estimate that supplies inference
    is unchanged, and a band member with status ``"out_of_range_plugin"`` reads its plug-in
    standard error without the accessor's refusal.
    """
    return estimate._plugin_std_error()


def _checked_items(
    estimates: Mapping[str, ParameterEstimate] | Sequence[ParameterEstimate],
) -> list[tuple[str, ParameterEstimate]]:
    """The ``(name, estimate)`` pairs of a band, after the refusals of ``simultaneous_bands``."""
    items = (
        list(estimates.items())
        if isinstance(estimates, Mapping)
        else [(estimate.name, estimate) for estimate in estimates]
    )
    if not items:
        raise ValueError("no estimates supplied")
    # A constant arm cell refuses the band itself, ahead of the estimate's own refusal, so the
    # error names the band's status rather than the estimate's.
    constant = constant_arm_members(items)
    if constant:
        raise CapabilityError.from_refusal(constant_arm_band_refusal(constant))
    # An *inferential* use, unlike the tolerance scales elsewhere in the package: a band
    # is a joint coverage claim.  Refused here, naming the operation, rather than left to
    # raise out of ``estimate.std_error`` below under an accessor the caller never wrote.
    # A bounded estimate on or outside its range stays a member (:func:`band_member`).
    for _, estimate in items:
        if str(estimate.inference) == OUT_OF_RANGE_STATUS:
            continue
        refuse_inference(estimate.inference, operation="simultaneous_bands()")
    if any(estimate.reference_df is not None for _, estimate in items):
        raise CapabilityError.from_refusal(T_REFERENCE_BAND_REFUSAL)
    if known_regime_estimates(name for name, _ in items):
        raise CapabilityError.from_refusal(KNOWN_REGIME_BAND_REFUSAL)
    second_moment = [
        name for name, estimate in items if estimate.covariance_rule == "second_moment"
    ]
    if second_moment:
        raise CapabilityError(
            f"simultaneous_bands() cannot band {second_moment}, which declare "
            "covariance_rule='second_moment'. The multiplier draws center each influence "
            "curve, which matches the raw second moment only on a mean-zero curve, and "
            "simultaneous_bands() does not check the mean.",
            reason=RefusalReason.INFERENCE_GAP,
            alternative=(
                "Report the pointwise interval of each named estimate, or pass the other "
                "estimates to simultaneous_bands() without them."
            ),
        )
    lengths = {estimate.influence_curve.shape[0] for _, estimate in items}
    if len(lengths) != 1:
        raise ValueError(f"influence curves have inconsistent lengths: {lengths}")
    return items


def known_regime_estimates(names: Iterable[str]) -> list[str]:
    """The names among ``names`` that read a known regime beside a learned rule.

    Each is an ``ey_known_regime`` or ``ate_learned_rule`` parameter, a stratum alias
    included, and no band covers it (:data:`KNOWN_REGIME_BAND_REFUSAL`).

    Parameters
    ----------
    names : iterable of str
        Reported parameter names.

    Returns
    -------
    list of str
        The names whose stem is in
        :data:`~cleverly.interventions.learned.CONTRAST_TARGETS`, in order.
    """
    # Imported here: the interventions package imports this one.
    from ..interventions.learned import CONTRAST_TARGETS
    from ..targets.base import parameter_stem

    return [name for name in names if parameter_stem(name) in CONTRAST_TARGETS]


def _check_band_scale(band_scale: str) -> None:
    """Refuse a band scale this module does not build."""
    if band_scale not in _BAND_SCALES:
        raise ValueError(f"band_scale must be 'logit' or 'identity'; got {band_scale!r}")


def _range_transform(estimate: ParameterEstimate, band_scale: str) -> Transform | None:
    """The range-logit transform of a bounded estimate under ``band_scale="logit"``."""
    if band_scale != "logit" or estimate.bounds is None:
        return None
    if estimate.transform is not None or estimate.scale == "ratio":
        return None
    low, high = estimate.bounds
    return Transform.range_logit(low, high)


def out_of_range(estimates: Mapping[str, ParameterEstimate], band_scale: str) -> tuple[str, ...]:
    """The bounded estimates a ``band_scale`` band cannot be built around.

    Under ``band_scale="logit"`` an estimate with known bounds and a positive standard error
    must lie strictly inside its range: the logit of a value on the boundary is infinite and
    outside it undefined.  A zero standard error is never out of range, because its band is
    the point itself on any scale.  A fit's default band reads this to skip the band and say
    why; :func:`simultaneous_bands` and :func:`median_bands` refuse.

    Parameters
    ----------
    estimates : mapping of str to ParameterEstimate
        The estimates a band would cover.
    band_scale : {"logit", "identity"}
        The band's scale.

    Returns
    -------
    tuple of str
        The names that sit on or outside their range, in order; empty for an identity band.
    """
    if band_scale != "logit":
        return ()
    out = []
    for name, estimate in estimates.items():
        if _range_transform(estimate, band_scale) is None:
            continue
        low, high = estimate.bounds  # type: ignore[misc]
        if float(_band_std_error(estimate)) > 0.0 and not (low < float(estimate.psi) < high):
            out.append(name)
    return tuple(out)


def _refuse_out_of_range(items: Sequence[tuple[str, ParameterEstimate]], band_scale: str) -> None:
    """Refuse a logit band over a bounded estimate that sits on or outside its range."""
    named = dict(items)
    for name in out_of_range(named, band_scale):
        estimate = named[name]
        low, high = estimate.bounds  # type: ignore[misc]
        raise CapabilityError.from_refusal(
            Refusal(
                RefusalReason.INFERENCE_GAP,
                f"simultaneous_bands() has no logit-scale band for {name!r}: its "
                f"estimate {float(estimate.psi):.6g} lies on or outside its range "
                f"({low:g}, {high:g}), where the logit is not defined.",
                alternative=(
                    "Pass band_scale='identity' for a band on the estimate's own scale, "
                    "or band the other estimates without this one."
                ),
            )
        )


def _band_endpoints(
    items: Sequence[tuple[str, ParameterEstimate]], critical: float, band_scale: str
) -> tuple[dict[str, tuple[float, float]], dict[str, str]]:
    """Each estimate's band, ``critical`` standard errors either side on its band scale.

    Returns the bands and the scale each was built on.  A bounded estimate under
    ``band_scale="logit"`` reads its standard error on the range-logit scale by the delta
    method; a zero standard error gives the point band.
    """
    bands: dict[str, tuple[float, float]] = {}
    scales: dict[str, str] = {}
    for name, estimate in items:
        std_error = _band_std_error(estimate)
        half_width = critical * std_error
        transform = estimate.transform
        ranged = _range_transform(estimate, band_scale)
        if ranged is not None:
            psi = float(estimate.psi)
            scales[name] = ranged.name
            if std_error == 0.0:
                bands[name] = (psi, psi)
                continue
            center = float(ranged.forward(psi))
            spread = half_width * ranged.slope(psi)
            low, high = sorted(
                (float(ranged.inverse(center - spread)), float(ranged.inverse(center + spread)))
            )
            bands[name] = (low, high)
        elif transform is not None:
            # The band is built on the transform's scale and mapped back through the
            # inverse, sorted so a decreasing transform still gives an ordered band.
            scales[name] = transform.name
            center = estimate.inference_value
            low, high = sorted(
                (
                    float(transform.inverse(center - half_width)),
                    float(transform.inverse(center + half_width)),
                )
            )
            bands[name] = (low, high)
        elif estimate.scale == "ratio":
            scales[name] = "log"
            center = estimate.inference_value
            bands[name] = (
                float(np.exp(center - half_width)),
                float(np.exp(center + half_width)),
            )
        else:
            scales[name] = "identity"
            bands[name] = (estimate.psi - half_width, estimate.psi + half_width)
    return bands, scales


def median_bands(
    per_draw: Sequence[Mapping[str, ParameterEstimate]],
    combined: Mapping[str, ParameterEstimate],
    *,
    alpha: float = 0.05,
    n_replicates: int = DEFAULT_N_REPLICATES,
    kind: MultiplierKind = DEFAULT_KIND,
    random_state: int | None = None,
    cluster: IntArray | None = None,
    band_scale: BandScale = DEFAULT_BAND_SCALE,
) -> SimultaneousBands:
    r"""Joint bands of a repeated fit: the median critical value around the median estimates.

    Each draw :math:`r` of the cross-fitting split gives its own multiplier critical value
    :math:`c_r`, from that draw's influence curves and standard errors and the one
    ``random_state``.  The band of estimand :math:`j` is
    :math:`\tilde\psi_j \pm \tilde c\,\tilde\sigma_j` on its inference scale, with
    :math:`\tilde c = \operatorname{median}_r c_r` and :math:`\tilde\psi_j`,
    :math:`\tilde\sigma_j` the split-adjusted estimate and standard error of ``combined``.
    It maps back to the reported scale as :func:`simultaneous_bands` maps a band.  Each
    draw is first stamped with the precedent of the draws' statuses, so an estimand that
    withholds inference in any draw is refused here as it is there.

    Parameters
    ----------
    per_draw : sequence of mappings
        Each draw's estimates.  Every name of ``combined`` must be present in every draw.
    combined : mapping of str to ParameterEstimate
        The median-combined estimates to band, in band order.
    alpha : float
        Joint significance level.
    n_replicates : int, default=10_000
        Multiplier draws behind each critical value.
    kind : {"rademacher", "mammen", "normal"}, default="mammen"
        Distribution of the multiplier weights.
    random_state : int or None
        Seed for those draws, the same for every draw of the split.
    cluster : ndarray or None
        ``(n,)`` cluster codes. Weights are drawn per cluster when given.
    band_scale : {"logit", "identity"}, default="logit"
        The scale of the band of a bounded estimate, as in :func:`simultaneous_bands`.

    Returns
    -------
    SimultaneousBands
        The bands, with :attr:`SimultaneousBands.draw_critical_values` holding each
        :math:`c_r`.

    Raises
    ------
    ValueError
        As :func:`simultaneous_bands` raises, for the combined or for a draw's estimates.
    CapabilityError
        As :func:`simultaneous_bands` raises.
    """
    _check_band_scale(band_scale)
    items = _checked_items(combined)
    _refuse_out_of_range(items, band_scale)
    names = [name for name, _ in items]
    stamped = with_precedent_status([{name: draw[name] for name in names} for draw in per_draw])
    critical_values: list[float] = []
    for draw in stamped:
        draw_items = _checked_items(draw)
        critical_values.append(
            multiplier_critical_value(
                np.column_stack([estimate.influence_curve for _, estimate in draw_items]),
                np.array([_band_std_error(estimate) for _, estimate in draw_items]),
                n=draw_items[0][1].n,
                cluster=cluster,
                alpha=alpha,
                n_replicates=n_replicates,
                kind=kind,
                random_state=random_state,
            )
        )
    critical = float(np.median(critical_values))
    bands, scales = _band_endpoints(items, critical, band_scale)
    return SimultaneousBands(
        critical_value=critical,
        bands=bands,
        alpha=alpha,
        n_replicates=n_replicates,
        kind=kind,
        draw_critical_values=tuple(critical_values),
        support=combine_support([estimate.support for _, estimate in items]),
        band_scale=band_scale,
        scales=scales,
    )
