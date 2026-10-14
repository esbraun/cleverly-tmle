r"""The conditional density of a continuous treatment, by pooled hazard.

An arm-coded treatment has a mechanism that is a *distribution over arms*: an ``(n, K)``
matrix, estimated by a classifier.  A continuous one has a conditional *density*
:math:`g(a \mid W)` on a continuum, and nothing in the learner layer predicts one.

Rather than widening the learner contract, this module factorises the density into the
conditional probabilities the layer already estimates.  Cut the treatment's range into
``B`` bins with edges :math:`e_0 < \dots < e_B`, write :math:`b(a)` for the bin holding
:math:`a`, and define the **discrete hazard**

.. math::

    \lambda_b(W) = P\bigl(b(A) = b \mid b(A) \ge b,\ W\bigr).

Then

.. math::

    P\bigl(b(A) = b \mid W\bigr)
      = \lambda_b(W) \prod_{j < b} \bigl(1 - \lambda_j(W)\bigr),
    \qquad
    \hat g(a \mid W) = \frac{P(b(A) = b(a) \mid W)}{e_{b(a)+1} - e_{b(a)}} .

Every :math:`\lambda_b` is a conditional probability of a binary event, so **one** binary
classifier fit on a long ``(unit, bin)`` expansion estimates all of them at once.  That is
why :data:`~cleverly._typing.Learner` needs no ``predict_density``: any scikit-learn
classifier, any preset library, and :class:`~cleverly.learners.SuperLearner` itself serve
here unchanged, and the Super Learner's negative log-likelihood on the long data *is* the
discretised conditional log-likelihood up to the constant :math:`\log(e_{b+1} - e_b)`, so
its model selection is genuinely selecting a density.

The construction follows Diaz & van der Laan (2011).

**What the bins cost, stated rather than hidden.**  A histogram density is constant within
a bin.  A shift :math:`\delta` much smaller than a bin width therefore leaves
:math:`\hat g(a - \delta \mid W) / \hat g(a \mid W)` at exactly one for most rows, and the
intervention becomes invisible to the estimator -- not noisy, *invisible*.
:attr:`DensityDiagnostics.crossing_fraction` reports the share of rows a shift actually
moves across a bin edge, and a fit warns when it is small.  The bias is
:math:`O(h)` in the smoothness of :math:`\log g`; the plain statement is that **the bin
edges are the resolution at which a policy is legible.**

**Why the edges come from the whole sample.**  They discretise the *support*, and marginal
quantiles of :math:`A` involve neither :math:`W` nor :math:`Y`, so the dependence they
introduce is far weaker than the conditional fit the cross-fitting exists to protect.  The
alternative -- edges per fold -- would give each fold its own grid, and the evaluated
density could then no longer be one ``(n, B)`` matrix that
:meth:`ConditionalDensity.density_at` reads by lookup.  Pass ``edges=`` to remove the
question entirely.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Protocol, overload

import numpy as np

from .._typing import BoolArray, FloatArray, IntArray, Learner
from ..exceptions import DataError, MethodConfigurationError
from ..utils.memory import available_memory
from ..utils.parallel import map_parallel, resolve_n_jobs
from ._fitting import fit_learner, predict_mean
from .crossfit import Folds
from .super_learner import SuperLearnerDiagnostics

__all__ = [
    "ConditionalDensity",
    "DensityDiagnostics",
    "DesignRows",
    "PredictionMemory",
    "bin_edges",
    "fit_conditional_density",
    "preflight_conditional_densities",
    "warn_if_unresolved",
]

#: Hazards are floored away from 0 and 1 before the log.  ``pyproject.toml`` turns a
#: ``RuntimeWarning`` into an error, so an unfloored ``log(0)`` in a fold where some
#: classifier saturates is a build failure rather than a ``-inf`` that propagates.
_HAZARD_EPS = 1e-12

#: Below this share of rows moved across a bin edge, a shift is mostly invisible to a
#: binned density and the fit says so.  Not a threshold with a derivation behind it -- it
#: is the point at which most rows contribute a clever covariate of exactly one.
_CROSSING_WARNING = 0.5


#: The smallest default bin count, the count every fit used before the default grew with n.
MIN_DEFAULT_BINS = 20


def default_density_bins(n: int) -> int:
    """The default bin count of a binned density fitted on ``n`` rows: ``max(20, ceil(2 n^(1/3)))``.

    A histogram density is consistent only if its bin count grows with ``n`` while each bin
    still holds a growing number of rows.  A fixed count is not: on the continuous laws of the
    ``longitudinal-mtp`` study, the error of a binned density ratio in the tail bins did not
    shrink from ``n = 2,000`` to ``n = 8,000`` at a fixed count, and fell as the count grew.
    Scott (1979, *Biometrika* 66(3):605-610) gives the rate: the bin width that minimises the
    integrated squared error shrinks as ``n^(-1/3)``.  The constant 2 is the Rice
    rule.  Below ``n = 1,000`` the floor of 20 bins applies.

    Parameters
    ----------
    n : int
        The number of rows the density is fitted on.

    Returns
    -------
    int
        The bin count.
    """
    return max(MIN_DEFAULT_BINS, math.ceil(2.0 * max(int(n), 1) ** (1.0 / 3.0)))


def hazard_design_bytes(n: int, n_bins: int, n_covariates: int, *, index_only: bool = False) -> int:
    """The bytes of the pooled hazard design at equal-mass bins: ``n (K + 1) / 2`` records.

    Equal-mass bins put about ``n / K`` rows in each bin, and a row in bin ``b`` contributes
    ``b + 1`` records, so the long table holds about ``n (K + 1) / 2`` records.  Each record
    holds the covariates, the bin index and ``K - 2`` drop-first indicators of the ``K - 1``
    hazards, at 8 bytes each.  A learner with ``bin_design = "index"`` omits the indicators.
    At the default count the total grows as ``n^(5/3)``.

    Parameters
    ----------
    n : int
        The rows the density is fitted on.
    n_bins : int
        The bin count ``K``.
    n_covariates : int
        The covariate columns.
    index_only : bool
        Whether the design omits the indicator block.

    Returns
    -------
    int
        The design's bytes, before any copy a learner makes.
    """
    records = n * (n_bins + 1) // 2
    columns = n_covariates + (1 if index_only else max(n_bins - 1, 1))
    return int(records * columns * 8)


def _refuse_unaffordable(
    records: int,
    columns: int,
    n: int,
    n_bins: int,
    n_covariates: int,
    index_only: bool,
    *,
    concurrent: int = 1,
    retained: int = 0,
    designs: int = 0,
) -> None:
    """Refuse a hazard design larger than the available memory, before it is allocated.

    ``concurrent`` is the number of fold fits that hold a long table at the same time, which
    is the number of parallel workers of a cross-fitted fit.  ``retained`` is the bytes that
    stay allocated beside the table, such as the evaluated densities of the earlier
    components of a vector node.  A refusal names the design's size, the memory available,
    and the largest bin count whose design fits in half of it.  Where the platform reports no
    available memory, nothing is checked.

    ``designs`` is the bytes the counterfactual predictions of a vector node hold beside the
    table (:class:`PredictionMemory`), which no bin count changes.  When they alone exceed
    half the memory, the refusal names the ways to shrink them.
    """
    free = available_memory()
    table = records * columns * 8
    needed = table * concurrent + retained + designs
    if free is None or needed <= free:
        return
    fits = [
        k
        for k in range(2, n_bins)
        if hazard_design_bytes(n, k, n_covariates, index_only=index_only) * concurrent
        + retained
        + designs
        <= free // 2
    ]
    advice = (
        f"Pass density_bins={max(fits)} or fewer, whose design takes at most half of it."
        if fits
        else "No bin count fits; fit on fewer rows."
    )
    gib = 1024.0**3
    if designs:
        advice += (
            f" The counterfactual designs and values of the node hold {designs / gib:.1f} GiB, "
            "which no bin count changes. To shrink them, declare fewer policies or pieces at "
            "the node, collapse a categorical component, pass n_jobs=1, or pass "
            "ratio='classifier', which evaluates no counterfactual design."
        )
    held = (
        ""
        if concurrent == 1 and not retained and not designs
        else (
            f" ({concurrent} fold fit(s) at once, beside {retained / gib:.1f} GiB already held"
            + (f" and {designs / gib:.1f} GiB of counterfactual designs" if designs else "")
            + ")"
        )
    )
    raise MethodConfigurationError(
        f"the binned density's pooled hazard design needs {needed / gib:.1f} GiB "
        f"({records:,} records x {columns:,} columns x 8 bytes{held}) for {n:,} rows and "
        f"{n_bins} bins, and {free / gib:.1f} GiB of memory is available. {advice} The default "
        "bin count grows as 2 n^(1/3), so the design grows as n^(5/3)."
    )


def bin_edges(values: FloatArray, n_bins: int, *, name: str | None = None) -> FloatArray:
    """Equal-mass bin edges for ``values``: ``n_bins + 1`` of them, strictly increasing.

    Empirical quantiles, so each bin holds about the same number of observations rather
    than the same width.  That is the right default for a density whose tails are the
    part positivity depends on: equal-width bins put almost no data in the tails, and the
    hazard there is then estimated from a handful of rows.

    Ties can make two quantiles coincide; duplicates are dropped and the bin count falls,
    which is reported rather than worked around.  ``name`` is the column the values belong
    to, which a refusal names; a vector node fits one density per continuous component.
    """
    a = np.asarray(values, dtype=float).reshape(-1)
    if n_bins < 2:
        raise DataError(f"a binned density needs at least 2 bins; got {n_bins}")
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(a, quantiles))
    if edges.size < 3:
        what = "the treatment" if name is None else f"the dose {name!r}"
        raise DataError(
            f"{what} has too few distinct values to form {n_bins} bins; "
            f"only {edges.size - 1} non-empty bin(s) survive. Reduce n_bins, or model "
            "the treatment as arms."
        )
    # Widen the outer edges by a hair so the smallest and largest observations land
    # strictly inside a bin rather than on its boundary, where ``digitize`` would push
    # the maximum into a bin that does not exist.
    span = float(edges[-1] - edges[0])
    pad = max(span, 1.0) * 1e-9
    edges[0] -= pad
    edges[-1] += pad
    return np.asarray(edges, dtype=float)


@dataclass(frozen=True)
class DensityDiagnostics:
    """What the density fit looked like, for the report and for the positivity check.

    Parameters
    ----------
    n_bins : int
        Bins actually used, after duplicate quantiles were dropped.
    edges : ndarray
        The bin boundaries.
    min_density : float
        Smallest :math:`\\hat g(A_i \\mid W_i)` over the sample.  The denominator of every
        clever covariate, so the number a positivity problem shows up in first.
    integrated : tuple of float
        ``(min, max)`` of :math:`\\int \\hat g(a \\mid W_i)\\,da` over rows.  Both should be
        one to machine precision; anything else is a bug in the hazard product, not a
        modelling matter.
    learner : tuple of SuperLearnerDiagnostics
        Per-fold Super Learner diagnostics, empty when the learner is not an ensemble.
    """

    n_bins: int
    edges: FloatArray
    min_density: float
    integrated: tuple[float, float]
    learner: tuple[SuperLearnerDiagnostics, ...] = ()


@dataclass(frozen=True)
class ConditionalDensity:
    """An evaluated conditional density: one row of bin probabilities per observation.

    Like :class:`~cleverly.interventions.RegimeSet`, this holds *evaluated arrays and no
    callables*.  The density is estimated once, out of fold, and stored as the ``(n, B)``
    matrix :math:`P(b(A) = b \\mid W_i)`; every later evaluation -- at the observed
    treatment, at a shifted one, at a grid for a plot -- is a **lookup into row** ``i``,
    not a model call.

    That is not only an efficiency: it is what makes "``g(A | W)`` and ``g(A - delta | W)``
    come from the same out-of-fold model" a structural fact rather than an invariant
    somebody has to maintain.  It is also what lets a fit survive
    :func:`~cleverly.load`, a bootstrap resample and every
    :meth:`~cleverly.estimators.TMLE.retarget` without the learner being refit.

    Parameters
    ----------
    bin_probabilities : ndarray
        ``(n, B)``, rows summing to one.  Out-of-fold predictions.
    edges : ndarray
        ``(B + 1,)`` bin boundaries, strictly increasing.
    """

    bin_probabilities: FloatArray
    edges: FloatArray

    def __post_init__(self) -> None:
        p = np.asarray(self.bin_probabilities, dtype=float)
        e = np.asarray(self.edges, dtype=float)
        if p.ndim != 2:
            raise ValueError(f"bin probabilities must be (n, B); got shape {p.shape}")
        if e.ndim != 1 or e.size != p.shape[1] + 1:
            raise ValueError(
                f"a density over {p.shape[1]} bins needs {p.shape[1] + 1} edges; got {e.size}"
            )
        if np.any(np.diff(e) <= 0.0):
            raise ValueError("bin edges must be strictly increasing")

    # ------------------------------------------------------------------ access

    @property
    def n(self) -> int:
        """Return the number of observations."""
        return int(self.bin_probabilities.shape[0])

    @property
    def n_bins(self) -> int:
        """Return the number of bins the treatment support is divided into.

        Returns
        -------
        int
            Bin count, which is the second axis of :attr:`bin_probabilities`.
        """
        return int(self.bin_probabilities.shape[1])

    @property
    def widths(self) -> FloatArray:
        """``(B,)`` bin widths, the Jacobian turning a probability into a density."""
        return np.asarray(np.diff(self.edges), dtype=float)

    @property
    def support(self) -> tuple[float, float]:
        """``(lower, upper)`` -- the range the density is nonzero on."""
        return float(self.edges[0]), float(self.edges[-1])

    def bin_of(self, values: FloatArray) -> IntArray:
        """Which bin each value falls in; ``-1`` outside the support.

        ``-1`` rather than a clamp, so a caller that evaluates outside the observed range
        gets a zero density and can see that it did, instead of silently reading the
        nearest edge bin.

        Parameters
        ----------
        values : ndarray
            ``(n,)`` treatment values, one per row.

        Returns
        -------
        ndarray
            ``(n,)`` bin index per row, and ``-1`` where the value is outside the support.
        """
        a = np.asarray(values, dtype=float).reshape(-1)
        index = np.digitize(a, self.edges) - 1
        outside = (a < self.edges[0]) | (a > self.edges[-1])
        index = np.clip(index, 0, self.n_bins - 1)
        return np.asarray(np.where(outside, -1, index), dtype=np.int64)

    def density_at(self, values: FloatArray) -> FloatArray:
        """:math:`\\hat g(a_i \\mid W_i)` for one value per row; zero outside the support.

        A pure lookup into :attr:`bin_probabilities`, so this can be called as often as an
        estimand needs without refitting anything.

        Parameters
        ----------
        values : ndarray
            ``(n,)`` treatment values, one per row.

        Returns
        -------
        ndarray
            ``(n,)`` density at those values, and zero outside the support.
        """
        a = np.asarray(values, dtype=float).reshape(-1)
        if a.size != self.n:
            raise ValueError(f"density_at needs one value per row: got {a.size} for {self.n} rows")
        index = self.bin_of(a)
        inside = index >= 0
        safe = np.where(inside, index, 0)
        probability = self.bin_probabilities[np.arange(self.n), safe]
        density = probability / self.widths[safe]
        return np.asarray(np.where(inside, density, 0.0), dtype=float)

    def integrated(self) -> FloatArray:
        """``(n,)`` values of :math:`\\int \\hat g(a \\mid W_i)\\, da`, which must all be one.

        The hazard product is constructed to sum to one exactly, so this is a check on the
        arithmetic rather than on the model; it is reported because a silent failure here
        would rescale every clever covariate by an amount no other diagnostic would see.

        Returns
        -------
        ndarray
            ``(n,)`` integral of the fitted density over the support, per row.
        """
        return np.asarray(self.bin_probabilities.sum(axis=1), dtype=float)

    def crossing_fraction(self, shifted: FloatArray, observed: FloatArray) -> float:
        """Share of rows a shift actually moves into a different bin.

        The resolution guard described in the module docstring: a shift that moves nobody
        across an edge has a clever covariate of exactly one everywhere and estimates
        nothing, however clean the fit looks.

        Parameters
        ----------
        shifted : ndarray
            ``(n,)`` dose the policy assigns.
        observed : ndarray
            ``(n,)`` dose actually received.

        Returns
        -------
        float
            Share of rows whose shifted dose falls in a different bin. A shift that
            moves nobody across a bin boundary is invisible to a binned density.
        """
        return float(np.mean(self.bin_of(shifted) != self.bin_of(observed)))

    def subset(self, index: Any) -> ConditionalDensity:
        """The same density on a row subset -- a bootstrap resample, a validation fold.

        The rows are sliced and the edges kept, exactly as
        :meth:`~cleverly.interventions.RegimeSet.subset` slices its densities: re-deriving
        the edges from the resample would give the subset a different discretisation from
        the fit it is meant to be a replicate of.

        Parameters
        ----------
        index : array_like
            Row positions or a boolean mask.

        Returns
        -------
        ConditionalDensity
            The same density over the selected rows.
        """
        idx = np.asarray(index)
        if idx.dtype == bool:
            idx = np.flatnonzero(idx)
        return replace(self, bin_probabilities=self.bin_probabilities[idx])


# ------------------------------------------------------------------- fitting


#: A hazard learner that sets ``bin_design = "index"`` reads only the bin index, so the
#: pooled design omits the indicator block.  The block has ``K - 1`` columns over about
#: ``n K / 2`` records, which is most of the memory of a fit at a large bin count.
INDEX_ONLY = "index"


def _index_only(learner: Any) -> bool:
    return getattr(learner, "bin_design", None) == INDEX_ONLY


def _bin_block(index: IntArray, n_hazards: int, index_only: bool = False) -> FloatArray:
    """How the bin a record belongs to enters the pooled hazard design.

    Two representations of the same integer, side by side, because the two kinds of
    candidate estimators need different ones:

    - **drop-first indicators**, so a model linear in its design gets a *free* baseline
      hazard -- one coefficient per bin, unconstrained.  This is the discrete-time hazard
      model's nonparametric baseline, and without it a logistic regression would force the
      log-odds of the hazard to be linear in the bin index, which is a very particular
      density family rather than a mild approximation.  The drop-first convention is the
      one :meth:`~cleverly.data.CausalData.treatment_block` already uses, for the same
      full-rank reason.
    - **the index itself**, so a tree can split on it and so any learner can form an
      interaction between the bin and ``W``.  Without it, indicators alone would let a
      linear model shift the whole baseline by ``W`` but never change its *shape* with
      ``W`` -- a proportional-hazards restriction on the density that nothing here needs.
    """
    if index_only:
        return index.astype(float).reshape(-1, 1)
    indicators = (index.reshape(-1, 1) == np.arange(1, n_hazards).reshape(1, -1)).astype(float)
    return np.column_stack([index.astype(float), indicators])


def _long_expansion(
    covariates: FloatArray, bins: IntArray, n_hazards: int, index_only: bool = False
) -> tuple[FloatArray, FloatArray, IntArray]:
    """The ``(unit, bin)`` records a pooled hazard model is fit on.

    Unit ``i`` contributes one record per bin ``b <= min(b_i, n_hazards - 1)``: it was "at
    risk" in each of them and "stopped" only in its own.  The bin joins the design through
    :func:`_bin_block`, which is what lets a single pooled model carry a different hazard
    per bin.

    Returns the design, the 0/1 target, and the originating row of each record -- the last
    so weights and cluster codes can be lifted to the long table without being recomputed.
    """
    counts = np.minimum(bins, n_hazards - 1) + 1
    rows = np.repeat(np.arange(bins.size, dtype=np.int64), counts)
    offsets = np.repeat(np.cumsum(counts) - counts, counts)
    index = np.arange(rows.size, dtype=np.int64) - offsets
    design = np.column_stack([covariates[rows], _bin_block(index, n_hazards, index_only)])
    target = (index == bins[rows]).astype(float)
    return design, target, rows


def _hazard_matrix(
    model: Learner, covariates: FloatArray, n_hazards: int, index_only: bool = False
) -> FloatArray:
    """``(n, n_hazards)`` predicted hazards, one column per bin the model covers."""
    n = covariates.shape[0]
    columns = [
        predict_mean(
            model,
            np.column_stack(
                [covariates, _bin_block(np.full(n, b, dtype=np.int64), n_hazards, index_only)]
            ),
            "classification",
        )
        for b in range(n_hazards)
    ]
    return np.clip(np.column_stack(columns), _HAZARD_EPS, 1.0 - _HAZARD_EPS)


def _probabilities_from_hazards(hazards: FloatArray) -> FloatArray:
    """Turn ``(n, B-1)`` hazards into ``(n, B)`` bin probabilities, in log space.

    The last bin needs no hazard: a unit that survived every earlier bin is in it with
    probability one, which is what makes the rows sum to one *exactly* rather than to
    within a tolerance.

    The product runs through ``log1p`` rather than ``cumprod`` because a saturated
    classifier gives hazards at the clip boundary, and a cumulative product of a few
    hundred of those underflows to zero -- after which dividing by a bin width raises the
    ``RuntimeWarning`` that this project's pytest configuration turns into a failure.
    """
    log_survival = np.cumsum(np.log1p(-hazards), axis=1)
    leading = np.log(hazards)
    leading[:, 1:] += log_survival[:, :-1]
    return np.asarray(
        np.exp(np.column_stack([leading, log_survival[:, -1]])),
        dtype=float,
    )


class DesignRows(Protocol):
    """A conditioning design built on demand, a block of rows at a time.

    A counterfactual design differs from the fitted one in a few columns.  Building it for
    one fold's held-out rows only, from the columns it changes, keeps no ``(n, p)`` array
    per design (:mod:`~cleverly.interventions.joint_density`).
    """

    @property
    def nbytes(self) -> int:
        """Bytes the object holds besides the fitted design."""
        ...

    def __call__(self, index: IntArray, fitted: FloatArray) -> FloatArray:
        """The design at rows ``index``, given ``fitted``, the fitted design at those rows."""
        ...


#: Out-of-fold densities at further conditioning designs: by key, ``(design, values)``, with
#: ``design`` the ``(n, p)`` covariates the density conditions on, or a :class:`DesignRows`
#: that builds them, and ``values`` the ``(n,)`` or ``(n, m)`` doses it is evaluated at.
PredictDesigns = Mapping[Hashable, tuple["FloatArray | DesignRows", FloatArray]]


@dataclass(frozen=True)
class _Grid:
    """The bins of one density fit, read from the rows it is fitted on."""

    edges: FloatArray
    bins: IntArray
    n_total: int
    n_hazards: int


def _grid(
    treatment: FloatArray,
    mask: BoolArray | None,
    n_bins: int | None,
    edges: Sequence[float] | None,
    name: str | None,
) -> _Grid:
    """The edges and each row's bin, with the bin count's default and the edges checked."""
    grid_values = treatment if mask is None else treatment[mask]
    if n_bins is None:
        n_bins = default_density_bins(grid_values.size)
    grid = (
        bin_edges(grid_values, n_bins, name=name)
        if edges is None
        else np.asarray(edges, dtype=float)
    )
    if grid.ndim != 1 or grid.size < 3 or np.any(np.diff(grid) <= 0.0):
        raise DataError("edges= must be a strictly increasing sequence of at least 3 values")
    n_total = int(grid.size - 1)
    bins = np.clip(np.digitize(treatment, grid) - 1, 0, n_total - 1).astype(np.int64)
    return _Grid(grid, bins, n_total, n_total - 1)


def _table_size(
    learner: Learner, n_covariates: int, grid: _Grid, mask: BoolArray | None
) -> tuple[int, int, int]:
    """``(records, columns, rows)`` of the pooled hazard table over the fitted rows."""
    index_only = _index_only(learner)
    fitted_bins = grid.bins if mask is None else grid.bins[mask]
    records = int(np.sum(np.minimum(fitted_bins, grid.n_hazards - 1) + 1))
    columns = (
        n_covariates + _bin_block(np.zeros(1, dtype=np.int64), grid.n_hazards, index_only).shape[1]
    )
    return records, columns, int(fitted_bins.size)


def _concurrent_folds(folds: Folds, n_jobs: int) -> int:
    """How many fold fits hold a long table at once: one, or the parallel workers."""
    if folds.is_single:
        return 1
    return max(1, min(resolve_n_jobs(n_jobs), folds.n_folds))


@dataclass(frozen=True)
class PredictionMemory:
    """What one density's counterfactual predictions hold beside its long table.

    Parameters
    ----------
    held : int
        Bytes the calling process holds while the density is fitted: the stored columns
        of every design, the values, and the predictions, counted twice when the fold
        results arrive at once.
    shipped : int
        Bytes each parallel worker receives a copy of: the stored columns and the values.
    rows : int
        Bytes one fold builds at once to predict one design: its held-out rows of the
        design, of the hazards and of the bin probabilities.
    kept : int
        Bytes the predictions keep after the fit, which later components see as held.
    """

    held: int = 0
    shipped: int = 0
    rows: int = 0
    kept: int = 0

    def peak(self, concurrent: int) -> int:
        """The bytes beside the tables when ``concurrent`` fold fits run at once.

        Parameters
        ----------
        concurrent : int
            The fold fits that run at once.

        Returns
        -------
        int
            The peak.
        """
        copies = concurrent * self.shipped if concurrent > 1 else 0
        return self.held + copies + concurrent * self.rows


def preflight_conditional_densities(
    learner: Learner,
    components: Sequence[tuple[int, FloatArray]],
    folds: Folds,
    *,
    n_bins: int | None = None,
    fit_mask: Any = None,
    n_jobs: int = 1,
    names: Sequence[str | None] = (),
    predictions: Sequence[PredictionMemory] = (),
) -> None:
    """Refuse a set of densities that one node fits in turn, before the first is fitted.

    A vector node fits one density per continuous component, one after another.  Each fit's
    long table is freed before the next one is built, and each fit keeps its ``(n, B)`` bin
    probabilities.  The peak at component ``k`` is therefore the bin probabilities of the
    components before it plus the table of component ``k`` times the fold fits that run at
    once.  This checks that peak for every component before any learner runs, so a node never
    fits its first density and then refuses its second.

    Parameters
    ----------
    learner : Learner
        The hazard learner.
    components : sequence of tuple
        ``(n_covariates, dose)`` per component, in fit order.
    folds : Folds
        The outer split.
    n_bins : int or None
        The bin count, or ``None`` for the default.
    fit_mask : array_like or None
        The rows each density is fitted on.
    n_jobs : int
        Parallel workers across the folds.
    names : sequence of str or None
        The component names, for the refusals.
    predictions : sequence of PredictionMemory
        What each component's counterfactual predictions hold, in fit order; missing
        entries hold nothing.  The peak at component ``k`` adds its own
        :meth:`PredictionMemory.peak` and the predictions the earlier components kept.

    Raises
    ------
    MethodConfigurationError
        If a component's peak exceeds the available memory.
    """
    mask = None if fit_mask is None else np.asarray(fit_mask, dtype=bool).reshape(-1)
    concurrent = _concurrent_folds(folds, n_jobs)
    retained = 0
    labels = list(names) + [None] * (len(components) - len(names))
    costs = list(predictions) + [PredictionMemory()] * (len(components) - len(predictions))
    for (n_covariates, dose), name, cost in zip(components, labels, costs, strict=True):
        a = np.asarray(dose, dtype=float).reshape(-1)
        grid = _grid(a, mask, n_bins, None, name)
        records, columns, rows = _table_size(learner, n_covariates, grid, mask)
        _refuse_unaffordable(
            records,
            columns,
            rows,
            grid.n_total,
            n_covariates,
            _index_only(learner),
            concurrent=concurrent,
            retained=retained,
            designs=cost.peak(concurrent),
        )
        retained += a.size * grid.n_total * 8 + cost.kept


def _lookup(probabilities: FloatArray, edges: FloatArray, values: FloatArray) -> FloatArray:
    """The density of each row's bin probabilities at its ``(m,)`` values: ``(rows, m)``."""
    widths = np.diff(edges)
    n_total = widths.size
    v = np.asarray(values, dtype=float).reshape(probabilities.shape[0], -1)
    index = np.clip(np.digitize(v, edges) - 1, 0, n_total - 1)
    inside = (v >= edges[0]) & (v <= edges[-1]) & np.isfinite(v)
    rows = np.arange(probabilities.shape[0])[:, None]
    density = probabilities[rows, index] / widths[index]
    return np.asarray(np.where(inside, density, 0.0), dtype=float)


@overload
def fit_conditional_density(
    learner: Learner,
    covariates: FloatArray,
    treatment: FloatArray,
    weights: FloatArray,
    folds: Folds,
    *,
    n_bins: int | None = ...,
    edges: Sequence[float] | None = ...,
    groups: IntArray | None = ...,
    n_jobs: int = ...,
    fit_mask: Any = ...,
    name: str | None = ...,
    retained: int = ...,
    predict_designs: None = ...,
) -> tuple[ConditionalDensity, DensityDiagnostics]: ...


@overload
def fit_conditional_density(
    learner: Learner,
    covariates: FloatArray,
    treatment: FloatArray,
    weights: FloatArray,
    folds: Folds,
    *,
    n_bins: int | None = ...,
    edges: Sequence[float] | None = ...,
    groups: IntArray | None = ...,
    n_jobs: int = ...,
    fit_mask: Any = ...,
    name: str | None = ...,
    retained: int = ...,
    predict_designs: PredictDesigns,
) -> tuple[ConditionalDensity, DensityDiagnostics, dict[Hashable, FloatArray]]: ...


def fit_conditional_density(
    learner: Learner,
    covariates: FloatArray,
    treatment: FloatArray,
    weights: FloatArray,
    folds: Folds,
    *,
    n_bins: int | None = None,
    edges: Sequence[float] | None = None,
    groups: IntArray | None = None,
    n_jobs: int = 1,
    fit_mask: Any = None,
    name: str | None = None,
    retained: int = 0,
    predict_designs: PredictDesigns | None = None,
) -> (
    tuple[ConditionalDensity, DensityDiagnostics]
    | tuple[ConditionalDensity, DensityDiagnostics, dict[Hashable, FloatArray]]
):
    """Cross-fit :math:`g(a \\mid W)` and return it evaluated at every row.

    The fold loop is deliberately *not* :func:`cross_fit_predictions`.  That helper's
    contract is ``n`` rows in, ``n`` predictions out; a hazard model trains on a long
    expansion of :math:`\\sum_i (b_i + 1)` records and produces an ``(n, B)`` matrix, so
    bending the shared helper would put two orthogonal modes on one function.  What is
    shared is what matters: the same :class:`~cleverly.learners.Folds`, so a row's density
    and its outcome regression come from the same split, and
    :func:`~cleverly.learners.fit_learner`, so weights, cluster codes and screening
    pipelines route exactly as they do for every other nuisance.

    ``fit_mask`` restricts the rows a fold trains on, and the rows the bin edges are read
    from, to the rows that hold a dose: a longitudinal node is fitted on the units still in
    the study before it.  ``None`` reads every row, the point-treatment fit unchanged.

    ``predict_designs`` evaluates the density at further conditioning designs, by key: each
    entry is ``(design, values)``, and the result holds the density at ``values`` given
    ``design``, one value per entry of ``values``.  Row ``i`` of every design is predicted by
    the model of the fold that held row ``i`` out, as
    :func:`~cleverly.estimators._nuisance.cross_fit_predictions` predicts its designs.  Only
    these values are kept, not the bin probabilities at each design.  The density of a
    vector treatment reads a component's density at counterfactual values of the components
    it conditions on, which is what this is for.  With ``predict_designs`` the function
    returns the predictions as a third element.

    Before any fold allocates its long table, the fit counts the records of the table over
    every fitted row and raises :class:`~cleverly.exceptions.MethodConfigurationError` when
    the table, times the fold fits that run at once, plus ``retained`` bytes the caller
    already holds, exceeds the available memory.  The message gives the size and a bin count
    that fits.  A fold trains on a subset of those rows, so the count bounds every fold.
    """
    w = np.asarray(covariates, dtype=float)
    a = np.asarray(treatment, dtype=float).reshape(-1)
    sample_weight = np.asarray(weights, dtype=float).reshape(-1)
    mask = None if fit_mask is None else np.asarray(fit_mask, dtype=bool).reshape(-1)
    grid = _grid(a, mask, n_bins, edges, name)
    n_total, n_hazards, bins = grid.n_total, grid.n_hazards, grid.bins
    index_only = _index_only(learner)
    records, columns, rows_fitted = _table_size(learner, w.shape[1], grid, mask)
    _refuse_unaffordable(
        records,
        columns,
        rows_fitted,
        n_total,
        w.shape[1],
        index_only,
        concurrent=_concurrent_folds(folds, n_jobs),
        retained=retained,
    )
    designs = {} if predict_designs is None else dict(predict_designs)
    for key, (design, values) in designs.items():
        shape = (
            (a.size, design(np.zeros(1, dtype=np.int64), w[:1]).shape[1])
            if callable(design) and a.size
            else np.asarray(design).shape
        )
        if shape != (a.size, w.shape[1]):
            raise ValueError(
                f"predict design {key!r} has shape {shape}; the density "
                f"conditions on {(a.size, w.shape[1])}"
            )
        if np.asarray(values).shape[0] != a.size:
            raise ValueError(f"predict design {key!r} needs one row of values per unit")

    def fit_on(rows: IntArray) -> Learner:
        if mask is not None:
            rows = rows[mask[rows]]
        design, target, source = _long_expansion(w[rows], bins[rows], n_hazards, index_only)
        return fit_learner(
            learner,
            design,
            target,
            sample_weight[rows][source],
            groups=None if groups is None else np.asarray(groups)[rows][source],
            warn_unweighted=False,
        )

    def predicted_at(model: Learner, test: IntArray) -> dict[Hashable, FloatArray]:
        out = {}
        for key, (design, values) in designs.items():
            held_out = (
                np.asarray(design(test, w[test]), dtype=float)
                if callable(design)
                else np.asarray(design, dtype=float)[test]
            )
            probabilities = _probabilities_from_hazards(
                _hazard_matrix(model, held_out, n_hazards, index_only)
            )
            out[key] = _lookup(probabilities, grid.edges, np.asarray(values, dtype=float)[test])
        return out

    probabilities = np.empty((a.size, n_total), dtype=float)
    predictions = {
        key: np.zeros((a.size, int(np.prod(np.asarray(values).shape[1:], dtype=int))))
        for key, (_, values) in designs.items()
    }
    diagnostics: list[SuperLearnerDiagnostics] = []

    if folds.is_single:
        everyone = np.arange(a.size, dtype=np.int64)
        model = fit_on(everyone)
        probabilities[:] = _probabilities_from_hazards(
            _hazard_matrix(model, w, n_hazards, index_only)
        )
        for key, values in predicted_at(model, everyone).items():
            predictions[key][:] = values
        found = getattr(model, "diagnostics_", None)
        if found is not None:
            diagnostics.append(found)
    else:

        def run_fold(
            train: IntArray, test: IntArray
        ) -> tuple[IntArray, FloatArray, dict[Hashable, FloatArray], Any]:
            model = fit_on(train)
            hazards = _hazard_matrix(model, w[test], n_hazards, index_only)
            return (
                test,
                _probabilities_from_hazards(hazards),
                predicted_at(model, test),
                getattr(model, "diagnostics_", None),
            )

        for test, values, at_designs, found in map_parallel(
            run_fold, [(train, test) for train, test in folds], n_jobs=n_jobs
        ):
            probabilities[test] = values
            for key, predicted in at_designs.items():
                predictions[key][test] = predicted
            if found is not None:
                diagnostics.append(found)

    density = ConditionalDensity(probabilities, grid.edges)
    integrated = density.integrated()
    observed = density.density_at(a)
    if mask is not None:
        observed = observed[mask]
    report = DensityDiagnostics(
        n_bins=n_total,
        edges=grid.edges,
        min_density=float(observed.min()),
        integrated=(float(integrated.min()), float(integrated.max())),
        learner=tuple(diagnostics),
    )
    if predict_designs is None:
        return density, report
    shaped = {
        key: predictions[key].reshape(np.asarray(values).shape)
        for key, (_, values) in designs.items()
    }
    return density, report, shaped


def warn_if_unresolved(
    density: ConditionalDensity, shifted: FloatArray, observed: FloatArray
) -> float:
    """Warn when a policy moves too few rows across a bin edge, and return the share.

    Separated from the fit so the check happens where the *policy* is known: the density
    knows its own resolution, but only an intervention knows how far it means to move.
    """
    crossing = density.crossing_fraction(shifted, observed)
    if crossing < _CROSSING_WARNING:
        warnings.warn(
            f"the policy moves only {crossing:.1%} of rows across a bin edge of the "
            f"estimated density ({density.n_bins} bins over "
            f"[{density.support[0]:.3g}, {density.support[1]:.3g}]). A binned density is "
            "constant within a bin, so for the rest the clever covariate is exactly one "
            "and the intervention is invisible rather than merely noisy. Use more bins, "
            "or a larger change of dose, or read the estimate as the effect of a policy the bin "
            "width can resolve.",
            UserWarning,
            stacklevel=3,
        )
    return crossing
