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
      = \lambda_b(W) \prod_{j < b} \bigl(1 - \lambda_j(W)\bigr).

Every :math:`\lambda_b` is a conditional probability of a binary event, so **one** binary
classifier fit on a long ``(unit, bin)`` expansion estimates all of them at once.  That is
why :data:`~cleverly._typing.Learner` needs no ``predict_density``: any scikit-learn
classifier, any preset library, and :class:`~cleverly.learners.SuperLearner` itself serve
here unchanged.  The Super Learner's negative log-likelihood on the long data is the
discretised conditional log-likelihood of the bin masses, so its model selection selects
bin masses; the reading below derives the density from them.

The construction follows Diaz & van der Laan (2011).

**The grid.**  :func:`bin_edges` starts from ``K`` equal-mass bins, the *core*, and refines
the outer ones.  A core bin wider than the mean width ``h`` of the interior core bins is cut
into pieces no wider than ``h``.  An empty run of pieces longer than ``2 h`` and longer than
the span of the :data:`TAIL_ROWS` observations beside it becomes one empty *gap* bin, and the
other pieces are merged from the outside in until each holds at least
``min(TAIL_ROWS, m)`` of the ``m`` observations of its stretch.  Core edges are never removed,
so ``density_bins`` keeps meaning ``K``.  Equal-mass bins alone widen in the tails as ``n``
grows (Diaz & van der Laan 2011, Section 2.2, relay Denby and Mallows: equal-area bins
oversmooth the tails), and a density ratio read from them does not converge there.

**The reading.**  The density is not read as a histogram.  For each row, take its bin
probabilities :math:`p_b`, and, if the fit declares a support :math:`[l_i, u_i]`, each bin's
*effective interval*, its intersection with that support (the whole bin otherwise).  Mass on
bins wholly outside the support is dropped and the row rescaled to one.  A bin is *active*
when it holds an observation of the sample the edges were read from (:attr:`occupied`), its
effective interval has positive length, and :math:`p_b > 10^{-10}`.  A maximal run of active
bins is a *segment*.  Inside a segment of two or more bins, :math:`\log \hat g` is the
corrected log frequency polygon: linear between knots at the effective bin centres, with
values :math:`c_b = \log(p_b / w_b) - \varphi(s_b w_b)`, where
:math:`\varphi(x) = \log(\sinh(x/2) / (x/2))` turns a bin average into a point value under a
local slope :math:`s_b`, and the knots are the fixed point of the slopes they imply.  The end
half-bins continue the end secant (linear log tails, as in Kooperberg & Stone 1991,
Section 2).  Each segment is rescaled to its total mass, so the total mass, the fitted
hazards and the pooled likelihood are unchanged.  An inactive bin and a one-bin segment read
:math:`p_b / w_b` on the effective interval.  The density is zero outside the edges and
outside a declared support.  The reading is exact for a density that is exponential on a
segment, and its local error is second order in the bin width where :math:`\log g` has a
bounded second derivative.  How fast a ratio read this way converges is the package's own
argument and is measured, not taken from a published theorem; the technical reference states
it with its conditions.

**Why the edges come from the whole sample.**  They discretise the *support*, and marginal
quantiles of :math:`A` involve neither :math:`W` nor :math:`Y`, so the dependence they
introduce is far weaker than the conditional fit the cross-fitting exists to protect.  The
alternative -- edges per fold -- would give each fold its own grid, and the evaluated
density could then no longer be one ``(n, B)`` matrix that
:meth:`ConditionalDensity.density_at` reads row by row.  Pass ``edges=`` to remove the
question entirely.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Protocol, cast, overload

import numpy as np

from .._typing import BoolArray, FloatArray, IntArray, Learner
from .._warnings import warn as _warn
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
    "MixedDensity",
    "PredictionMemory",
    "bin_edges",
    "bin_occupancy",
    "fit_conditional_density",
    "preflight_conditional_densities",
    "warn_if_flat",
]

#: Hazards are floored away from 0 and 1 before the log.  ``pyproject.toml`` turns a
#: ``RuntimeWarning`` into an error, so an unfloored ``log(0)`` in a fold where some
#: classifier saturates is a build failure rather than a ``-inf`` that propagates.
_HAZARD_EPS = 1e-12

#: Above this share of moved rows whose observed and moved dose share one flat bin, the
#: ratio carries no information from the density for most rows and the fit says so.  Not a
#: threshold with a derivation behind it: it is the point at which most moved rows read a
#: ratio fixed by the bin alone (exactly one for a shift).
_FLAT_WARNING = 0.5

#: The fewest observations a refined outer bin holds, where its stretch has that many.  A bin
#: probability estimated from ``m`` rows has a relative standard error of about
#: ``1 / sqrt(m)``, 22% at 20 rows, and the outer bins are where a density ratio weights that
#: error most.
TAIL_ROWS = 20

#: A bin whose probability is at most this is inactive for its row and breaks the row's
#: segment.  Hazards are clipped to ``[1e-12, 1 - 1e-12]`` (:data:`_HAZARD_EPS`), so a bin
#: outside a row's support, or past the bin where its hazard reaches the upper clip, carries
#: at most about ``1e-12`` of the row's mass; the floor sits a factor of 100 above that.
_ACTIVE_FLOOR = 1e-10

#: The knots of a row are iterated until no knot moves by more than this, in log units.  The
#: iteration contracts slowly on steep rows, so the stopping step is far above the error
#: left: at ``1e-12`` an exponential law is read to about ``1e-13`` on ordinary slopes and
#: ``4e-12`` where ``|s w|`` reaches 21 (``tests/unit/test_density_reconstruction.py``).
_KNOT_TOLERANCE = 1e-12

#: The most passes of the knot iteration.  A row still moving at the cap keeps its capped
#: iterate; segment masses are exact either way, because each segment is rescaled after.
_KNOT_PASSES = 200

#: The reading works on blocks of rows of about this many ``(row, bin)`` entries, which
#: bounds its temporaries at a few tens of megabytes whatever ``n`` and ``B`` are.
_BLOCK_ENTRIES = 500_000


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
    table_at: Callable[[int], int] | None = None,
    core: int | None = None,
) -> None:
    """Refuse a hazard design larger than the available memory, before it is allocated.

    ``concurrent`` is the number of fold fits that hold a long table at the same time, which
    is the number of parallel workers of a cross-fitted fit.  ``retained`` is the bytes that
    stay allocated beside the table, such as the evaluated densities of the earlier
    components of a vector node.  A refusal names the design's size, the memory available,
    and the largest core count (``density_bins``) whose design fits in half of it.  Where the
    platform reports no available memory, nothing is checked.

    ``table_at`` gives the bytes of the table at a core count, on the refined grid that count
    gives the same rows (:func:`bin_edges`), and ``core`` is the core count of this grid.
    Without them the advice falls back to :func:`hazard_design_bytes` at ``n_bins``.

    ``designs`` is the bytes the counterfactual predictions of a vector node hold beside the
    table (:class:`PredictionMemory`), which no bin count changes.  When they alone exceed
    half the memory, the refusal names the ways to shrink them.
    """
    free = available_memory()
    table = records * columns * 8
    needed = table * concurrent + retained + designs
    if free is None or needed <= free:
        return

    def bytes_at(k: int) -> int:
        if table_at is None:
            return hazard_design_bytes(n, k, n_covariates, index_only=index_only)
        return table_at(k)

    def affordable(k: int) -> bool:
        return bytes_at(k) * concurrent + retained + designs <= free // 2

    # The table grows with the core count, so the largest affordable count is found by
    # bisection; each candidate's refined grid is computed before its memory is quoted.
    top = (n_bins if core is None else core) - 1
    fits: list[int] = []
    if top >= 2 and affordable(2):
        low, high = 2, top
        while low < high:
            middle = (low + high + 1) // 2
            if affordable(middle):
                low = middle
            else:
                high = middle - 1
        fits = [low]
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


def _equal_mass_edges(a: FloatArray, n_bins: int, name: str | None) -> FloatArray:
    """Equal-mass edges: empirical quantiles, duplicates dropped, outer edges padded."""
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


def _count(ordered: FloatArray, lower: float, upper: float) -> int:
    """How many sorted values lie in the closed interval ``[lower, upper]``."""
    return int(
        np.searchsorted(ordered, upper, side="right") - np.searchsorted(ordered, lower, side="left")
    )


def _merge_stretch(ordered: FloatArray, cuts: FloatArray, upper: bool) -> list[float]:
    """The inner cuts of one non-empty stretch that survive the :data:`TAIL_ROWS` floor.

    The stretch is merged from its outer end inward: a bin closes once it holds at least
    ``min(TAIL_ROWS, m)`` of the stretch's ``m`` observations, and a short remainder at the
    inner end joins the last closed bin.
    """
    lower, higher = float(cuts[0]), float(cuts[-1])
    need = min(TAIL_ROWS, _count(ordered, lower, higher))
    inner = [float(cut) for cut in cuts[1:-1]]
    sequence = inner[::-1] if upper else inner
    kept: list[float] = []
    start = higher if upper else lower
    for cut in sequence:
        held = _count(ordered, cut, start) if upper else _count(ordered, start, cut)
        if held >= need:
            kept.append(cut)
            start = cut
    if kept:
        end = lower if upper else higher
        rest = _count(ordered, end, kept[-1]) if upper else _count(ordered, kept[-1], end)
        if rest < need:
            kept.pop()
    return kept


def _width_cap(core: FloatArray) -> float:
    """``h``, the mean width of the interior core bins.

    The outer core edges are left out, so one extreme dose, which moves only ``e_0`` or
    ``e_K``, cannot coarsen the grid.
    """
    k = core.size - 1
    return float(core[k - 1] - core[1]) / (k - 2)


def _refine(a: FloatArray, core: FloatArray) -> FloatArray:
    """The outer refinement of an equal-mass core: width cap, gap bins and the row floor."""
    k = core.size - 1
    if k < 3:
        return core
    h = _width_cap(core)
    if not h > 0.0:
        return core
    ordered = np.sort(a)
    median = float(np.median(a))
    out = [float(core[0])]
    for lower, higher in itertools.pairwise(core):
        parts = math.ceil((higher - lower) / h - 1e-9)
        if parts <= 1:
            out.append(float(higher))
            continue
        cuts = np.linspace(lower, higher, parts + 1)
        # Each piece is half open on the right, as a bin is, except the last.
        counts = np.array(
            [
                _count(ordered, x, y) - (0 if j == parts - 1 else _count(ordered, y, y))
                for j, (x, y) in enumerate(itertools.pairwise(cuts))
            ]
        )
        upper = 0.5 * (lower + higher) > median
        stretches: list[tuple[str, int, int]] = []
        start = 0
        j = 0
        while j < parts:
            if counts[j] > 0:
                j += 1
                continue
            stop = j
            while stop < parts and counts[stop] == 0:
                stop += 1
            x0, x1 = float(cuts[j]), float(cuts[stop])
            # The span of the TAIL_ROWS observations next to the run on its inner side: an
            # empty run shorter than that is ordinary spacing in a thinning tail, not a gap.
            if upper:
                below = int(np.searchsorted(ordered, x0, side="right"))
                local = x0 - float(ordered[max(below - TAIL_ROWS, 0)])
            else:
                above = int(np.searchsorted(ordered, x1, side="left"))
                local = float(ordered[min(above + TAIL_ROWS - 1, ordered.size - 1)]) - x1
            if x1 - x0 > 2.0 * h and x1 - x0 > local:
                if j > start:
                    stretches.append(("data", start, j))
                stretches.append(("gap", j, stop))
                start = stop
            j = stop
        if parts > start:
            stretches.append(("data", start, parts))
        for kind, first, last in stretches:
            piece = cuts[first : last + 1]
            out.extend([float(piece[0]), float(piece[-1])])
            if kind == "data":
                out.extend(_merge_stretch(ordered, piece, upper))
        out.append(float(higher))
    return np.asarray(np.unique(np.asarray(out, dtype=float)), dtype=float)


def bin_edges(values: FloatArray, n_bins: int, *, name: str | None = None) -> FloatArray:
    """The bin edges of a binned density: an equal-mass core of ``n_bins`` bins, refined outside.

    The core is the empirical quantiles, so each core bin holds about the same number of
    observations.  Ties can make two quantiles coincide; duplicates are dropped and the core
    count falls, which is reported rather than worked around.  Equal-mass bins widen in the
    tails as the sample grows, so the outer core bins are refined:

    - a core bin wider than ``h``, the mean width of the interior core bins, is cut into equal
      pieces no wider than ``h``;
    - a run of pieces holding no observation becomes one empty *gap* bin when it is longer
      than ``2 h`` and longer than the span of the :data:`TAIL_ROWS` observations next to it
      on its inner side;
    - the other pieces are merged from the outer end inward until each bin holds at least
      ``min(TAIL_ROWS, m)`` of the ``m`` observations of its stretch.

    Core edges are never removed, so the grid has at least ``n_bins`` bins, and a core bin
    that is not split keeps its share of the rows.  ``name`` is the column the values belong
    to, which a refusal names; a vector node fits one density per continuous component.

    Parameters
    ----------
    values : ndarray
        The doses the grid is read from.
    n_bins : int
        The core count ``K``.
    name : str or None
        The dose's name, for the refusal.

    Returns
    -------
    ndarray
        Strictly increasing edges, at least ``n_bins + 1`` of them when no quantiles tie.

    Raises
    ------
    DataError
        If fewer than two core bins survive the ties, or ``n_bins`` is below two.
    """
    a = np.asarray(values, dtype=float).reshape(-1)
    return _refine(a, _equal_mass_edges(a, n_bins, name))


def bin_occupancy(values: FloatArray, edges: FloatArray) -> BoolArray:
    """The ``(B,)`` mask of bins that hold at least one of ``values``.

    A bin holds a value in ``[e_b, e_{b+1})``, and the last bin holds its upper edge too, as
    the density fit assigns rows to bins.  Values outside the edges are ignored.

    Parameters
    ----------
    values : ndarray
        The doses the edges were read from.
    edges : ndarray
        ``(B + 1,)`` bin edges.

    Returns
    -------
    ndarray
        ``(B,)`` booleans.
    """
    a = np.asarray(values, dtype=float).reshape(-1)
    e = np.asarray(edges, dtype=float)
    inside = a[(a >= e[0]) & (a <= e[-1])]
    index = np.clip(np.digitize(inside, e) - 1, 0, e.size - 2)
    return np.asarray(np.bincount(index, minlength=e.size - 1) > 0, dtype=bool)


# ------------------------------------------------------------------- the reading


def _phi(x: FloatArray) -> FloatArray:
    """``log(sinh(x/2) / (x/2))``: the log-average of ``exp(s (a - m))`` over a bin, ``x = s w``."""
    half = np.abs(np.asarray(x, dtype=float)) * 0.5
    small = half < 5e-3
    large = half > 300.0
    middle = np.where(small | large, 1.0, half)
    out = np.log(np.sinh(middle) / middle)
    if small.any():
        h2 = half[small] ** 2
        out[small] = h2 / 6.0 - h2 * h2 / 180.0 + h2 * h2 * h2 / 2835.0
    if large.any():
        out[large] = half[large] - np.log(2.0 * half[large])
    return np.asarray(out, dtype=float)


def _exp_integral(start: FloatArray, slope: FloatArray, length: FloatArray) -> FloatArray:
    """``int_0^length exp(start + slope x) dx``, stable as ``slope * length`` goes to zero."""
    z = slope * length
    small = np.abs(z) < 1e-8
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        relative = np.where(small, 1.0 + 0.5 * z, np.expm1(z) / np.where(small, 1.0, z))
        out = np.exp(start) * length * relative
    return np.asarray(np.where(length > 0.0, out, 0.0), dtype=float)


@dataclass(frozen=True)
class _Knots:
    """The reading of a block of rows: knot values, secants, links and the projection.

    ``value`` holds each bin's knot after the segment rescale (zero off multi-bin segments),
    ``secant`` the slope between two linked knots, and ``link`` whether bins ``b`` and
    ``b + 1`` lie in one segment of the row.  ``scale`` is the support projection's row
    factor: a bin's projected probability is its probability over ``scale`` inside the
    row's support.
    """

    probabilities: FloatArray
    edges: FloatArray
    lower: FloatArray | None
    upper: FloatArray | None
    value: FloatArray
    secant: FloatArray
    link: BoolArray
    scale: FloatArray
    passes: int
    capped: int

    @property
    def nbytes(self) -> int:
        return int(self.value.nbytes + self.secant.nbytes + self.link.nbytes + self.scale.nbytes)

    def take(self, rows: IntArray, index: IntArray) -> dict[str, FloatArray]:
        """Per-row arrays at bins ``index`` (one bin per entry of ``rows``)."""
        e = self.edges
        n_bins = e.size - 1
        left = e[index]
        right = e[index + 1]
        if self.lower is not None:
            left = np.maximum(left, self.lower[rows])
        if self.upper is not None:
            right = np.minimum(right, self.upper[rows])
        width = np.clip(right - left, 0.0, None)
        linked_left = np.where(index > 0, self.link[rows, np.maximum(index - 1, 0)], False)
        linked_right = np.where(
            index < n_bins - 1, self.link[rows, np.minimum(index, n_bins - 2)], False
        )
        secant_left = np.where(linked_left, self.secant[rows, np.maximum(index - 1, 0)], 0.0)
        secant_right = np.where(linked_right, self.secant[rows, np.minimum(index, n_bins - 2)], 0.0)
        probability = self.probabilities[rows, index] / self.scale[rows]
        probability = np.where(width > 0.0, probability, 0.0)
        return {
            "left": left,
            "right": right,
            "width": width,
            "centre": 0.5 * (left + right),
            "multi": linked_left | linked_right,
            "value": self.value[rows, index],
            "slope_left": np.where(linked_left, secant_left, secant_right),
            "slope_right": np.where(linked_right, secant_right, secant_left),
            "flat": probability / np.where(width > 0.0, width, 1.0),
        }


def _effective(
    edges: FloatArray, lower: FloatArray | None, upper: FloatArray | None, n: int
) -> tuple[FloatArray, FloatArray]:
    left = np.broadcast_to(edges[:-1], (n, edges.size - 1))
    right = np.broadcast_to(edges[1:], (n, edges.size - 1))
    if lower is not None:
        left = np.maximum(left, lower[:, None])
    if upper is not None:
        right = np.minimum(right, upper[:, None])
    return np.asarray(left, dtype=float), np.asarray(right, dtype=float)


def _knots(
    probabilities: FloatArray,
    edges: FloatArray,
    occupied: BoolArray,
    lower: FloatArray | None = None,
    upper: FloatArray | None = None,
    *,
    floor: float = _ACTIVE_FLOOR,
    passes: int = _KNOT_PASSES,
) -> _Knots:
    """The corrected log polygon of each row's bin probabilities (the module docstring)."""
    p = np.asarray(probabilities, dtype=float)
    n, n_bins = p.shape
    e = np.asarray(edges, dtype=float)
    left, right = _effective(e, lower, upper, n)
    width = np.clip(right - left, 0.0, None)
    supported = width > 0.0
    scale = np.ones(n)
    if lower is not None or upper is not None:
        total = np.where(supported, p, 0.0).sum(axis=1)
        scale = np.where(total > 0.0, total, 1.0)
    projected = np.where(supported, p, 0.0) / scale[:, None]
    active = np.asarray(occupied, dtype=bool)[None, :] & supported & (projected > floor)
    safe_width = np.where(supported, width, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        level = np.where(active, np.log(np.where(active, projected, 1.0) / safe_width), 0.0)
    centre = 0.5 * (left + right)
    link = active[:, :-1] & active[:, 1:]
    gap = np.where(link, centre[:, 1:] - centre[:, :-1], 1.0)
    has_left = np.zeros((n, n_bins), dtype=bool)
    has_right = np.zeros((n, n_bins), dtype=bool)
    has_left[:, 1:] = link
    has_right[:, :-1] = link
    # The slope map: an interior bin averages its two secants, an end bin of a segment takes
    # the secant to its one neighbour, and a bin outside every segment has no slope.
    from_left_weight = np.where(has_left, np.where(has_right, 0.5, 1.0), 0.0)
    from_right_weight = np.where(has_right, np.where(has_left, 0.5, 1.0), 0.0)
    inverse_gap = np.where(link, 1.0 / gap, 0.0)
    slope = np.zeros((n, n_bins))
    todo = np.flatnonzero(link.any(axis=1))
    work_level, work_width = level[todo], width[todo]
    work_gap = inverse_gap[todo]
    work_left, work_right = from_left_weight[todo][:, 1:], from_right_weight[todo][:, :-1]
    work_slope = np.zeros((todo.size, n_bins))
    previous = np.full((todo.size, n_bins), np.inf)
    used = 0
    while todo.size and used < passes:
        used += 1
        knot = work_level - _phi(work_slope * work_width)
        secant = np.diff(knot, axis=1) * work_gap
        work_slope = np.zeros_like(knot)
        work_slope[:, 1:] += work_left * secant
        work_slope[:, :-1] += work_right * secant
        moving = np.max(np.abs(knot - previous), axis=1) > _KNOT_TOLERANCE
        previous = knot
        if not moving.all():
            slope[todo[~moving]] = work_slope[~moving]
            todo = todo[moving]
            work_level, work_width = work_level[moving], work_width[moving]
            work_gap, work_slope = work_gap[moving], work_slope[moving]
            work_left, work_right = work_left[moving], work_right[moving]
            previous = previous[moving]
    slope[todo] = work_slope
    capped = int(todo.size)
    knot = np.where(active, level - _phi(slope * width), 0.0)
    secant = np.where(link, np.diff(knot, axis=1) / gap, 0.0)
    from_left = np.zeros((n, n_bins))
    from_right = np.zeros((n, n_bins))
    from_left[:, 1:] = secant
    from_right[:, :-1] = secant
    multi = has_left | has_right
    slope_left = np.where(has_left, from_left, from_right)
    slope_right = np.where(has_right, from_right, from_left)
    mass = _exp_integral(knot, -slope_left, centre - left) + _exp_integral(
        knot, slope_right, right - centre
    )
    # Segment ids: a new segment starts wherever a bin is not linked to its left neighbour.
    segment = np.cumsum(~has_left, axis=1) - 1
    flat_index = (np.arange(n)[:, None] * n_bins + segment).ravel()
    integral = np.bincount(flat_index, np.where(multi, mass, 0.0).ravel(), minlength=n * n_bins)
    held = np.bincount(flat_index, np.where(multi, projected, 0.0).ravel(), minlength=n * n_bins)
    with np.errstate(divide="ignore", invalid="ignore"):
        shift = np.log(held) - np.log(integral)
    shift = np.where(np.isfinite(shift), shift, 0.0)
    value = np.where(multi, knot + shift[flat_index].reshape(n, n_bins), 0.0)
    return _Knots(p, e, lower, upper, value, secant, link, scale, used, capped)


def _knots_by_block(
    probabilities: FloatArray,
    edges: FloatArray,
    occupied: BoolArray,
    lower: FloatArray | None,
    upper: FloatArray | None,
) -> _Knots:
    """:func:`_knots` a block of rows at a time; every step is row-wise, so the values agree."""
    p = np.asarray(probabilities, dtype=float)
    n, n_bins = p.shape
    block = max(1, _BLOCK_ENTRIES // max(n_bins, 1))
    if n <= block:
        return _knots(p, edges, occupied, lower, upper)
    parts = [
        _knots(
            p[start : start + block],
            edges,
            occupied,
            None if lower is None else lower[start : start + block],
            None if upper is None else upper[start : start + block],
        )
        for start in range(0, n, block)
    ]
    return _Knots(
        p,
        np.asarray(edges, dtype=float),
        lower,
        upper,
        np.vstack([part.value for part in parts]),
        np.vstack([part.secant for part in parts]),
        np.vstack([part.link for part in parts]),
        np.concatenate([part.scale for part in parts]),
        max(part.passes for part in parts),
        sum(part.capped for part in parts),
    )


def _read(knots: _Knots, values: FloatArray) -> FloatArray:
    """The reading at ``(n, m)`` values, one row of values per row of the knots."""
    e = knots.edges
    n = knots.value.shape[0]
    v = np.asarray(values, dtype=float).reshape(n, -1)
    rows = np.repeat(np.arange(n), v.shape[1])
    flat = v.ravel()
    finite = np.isfinite(flat)
    safe = np.where(finite, flat, e[0])
    index = np.clip(np.searchsorted(e, safe, side="right") - 1, 0, e.size - 2)
    at = knots.take(rows, index)
    inside = finite & (safe >= e[0]) & (safe <= e[-1])
    inside &= (safe >= at["left"]) & (safe <= at["right"]) & (at["width"] > 0.0)
    slope = np.where(safe >= at["centre"], at["slope_right"], at["slope_left"])
    with np.errstate(over="ignore"):
        polygon = np.exp(at["value"] + slope * (safe - at["centre"]))
    out = np.where(at["multi"], polygon, at["flat"])
    out = np.where(inside & np.isfinite(out), out, 0.0)
    return np.asarray(out.reshape(v.shape), dtype=float)


def _mass(knots: _Knots, lower: FloatArray, upper: FloatArray) -> FloatArray:
    """``int_lower^upper`` of the reading, per row; zero where ``upper <= lower``."""
    n, n_bins = knots.value.shape
    lo = np.broadcast_to(np.asarray(lower, dtype=float).reshape(-1), (n,))
    hi = np.broadcast_to(np.asarray(upper, dtype=float).reshape(-1), (n,))
    out = np.zeros(n)
    block = max(1, _BLOCK_ENTRIES // max(n_bins, 1))
    columns = np.arange(n_bins)
    for start in range(0, n, block):
        rows = np.arange(start, min(n, start + block))
        k = rows.size
        r = np.repeat(rows, n_bins)
        at = knots.take(r, np.tile(columns, k))
        a = np.repeat(lo[rows], n_bins)
        b = np.repeat(hi[rows], n_bins)
        x0 = np.clip(np.maximum(a, at["left"]), at["left"], at["right"])
        x1 = np.clip(np.minimum(b, at["right"]), at["left"], at["right"])
        length = np.clip(x1 - x0, 0.0, None)
        centre = at["centre"]
        # The left half [left, centre] and the right half [centre, right] of a multi bin.
        l0, l1 = x0, np.minimum(x1, centre)
        r0, r1 = np.maximum(x0, centre), x1
        start_left = at["value"] + at["slope_left"] * (l0 - centre)
        start_right = at["value"] + at["slope_right"] * (r0 - centre)
        polygon = _exp_integral(
            start_left, at["slope_left"], np.clip(l1 - l0, 0.0, None)
        ) + _exp_integral(start_right, at["slope_right"], np.clip(r1 - r0, 0.0, None))
        piece = np.where(at["multi"], polygon, at["flat"] * length)
        out[rows] = np.sum(np.where(length > 0.0, piece, 0.0).reshape(k, n_bins), axis=1)
    return np.asarray(np.where(hi > lo, out, 0.0), dtype=float)


def _check_support(
    support: tuple[FloatArray, FloatArray] | None, n: int
) -> tuple[FloatArray, FloatArray] | None:
    if support is None:
        return None
    lower = np.asarray(support[0], dtype=float).reshape(-1)
    upper = np.asarray(support[1], dtype=float).reshape(-1)
    if lower.shape != (n,) or upper.shape != (n,):
        raise ValueError(
            f"a declared support needs (lower, upper) of shape ({n},); got "
            f"{lower.shape} and {upper.shape}"
        )
    # A row with no bound (NaN, such as a row that does not hold the node) is unbounded.
    lower = np.where(np.isnan(lower), -np.inf, lower)
    upper = np.where(np.isnan(upper), np.inf, upper)
    if np.any(np.logical_not(lower < upper)):
        raise ValueError("a declared support needs lower < upper on every row")
    return lower, upper


@dataclass(frozen=True)
class DensityDiagnostics:
    """What the density fit looked like, for the report and for the positivity check.

    Parameters
    ----------
    n_bins : int
        Bins actually used: the refined grid, after duplicate quantiles were dropped.
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
    knot_passes : tuple of int
        ``(passes, capped)`` of the reading's knot iteration: the most passes any row
        needed, and the rows still moving at the cap of 200, which keep their capped
        iterate with exact segment masses.
    core_bins : int or None
        The core count ``K`` of a grid the fit built (``density_bins``), before the outer
        refinement; ``None`` for explicit ``edges=``.
    """

    n_bins: int
    edges: FloatArray
    min_density: float
    integrated: tuple[float, float]
    learner: tuple[SuperLearnerDiagnostics, ...] = ()
    knot_passes: tuple[int, int] = (0, 0)
    core_bins: int | None = None


@dataclass(frozen=True)
class ConditionalDensity:
    """An evaluated conditional density: one row of bin probabilities per observation.

    Like :class:`~cleverly.interventions.RegimeSet`, this holds *evaluated arrays and no
    callables*.  The density is estimated once, out of fold, and stored as the ``(n, B)``
    matrix :math:`P(b(A) = b \\mid W_i)`; every later evaluation -- at the observed
    treatment, at a shifted one, at a grid for a plot -- reads row ``i`` by the corrected
    log polygon of the module docstring, not a model call.

    That is not only an efficiency: it is what makes "``g(A | W)`` and ``g(A - delta | W)``
    come from the same out-of-fold model" a structural fact rather than an invariant
    somebody has to maintain.  It is also what lets a fit survive
    :func:`~cleverly.load`, a bootstrap resample and every
    :meth:`~cleverly.estimators.TMLE.retarget` without the learner being refit.  The knots of
    the reading are computed once per instance and are not saved with it.

    Parameters
    ----------
    bin_probabilities : ndarray
        ``(n, B)``, rows summing to one.  Out-of-fold predictions.
    edges : ndarray
        ``(B + 1,)`` bin boundaries, strictly increasing.
    occupied : ndarray
        ``(B,)`` booleans: the bins that hold an observation of the sample the edges were
        read from.  An empty bin breaks every row's segment.
    support : tuple of ndarray or None
        The declared support ``(lower, upper)`` on each row, or ``None``.  The density is
        read over each bin's intersection with the row's support and is zero outside it.
    """

    bin_probabilities: FloatArray
    edges: FloatArray
    occupied: BoolArray
    support: tuple[FloatArray, FloatArray] | None = None

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
        occupied = np.asarray(self.occupied)
        if occupied.shape != (p.shape[1],) or occupied.dtype != bool:
            raise ValueError(
                f"occupied must be a boolean mask of shape ({p.shape[1]},); got "
                f"{occupied.dtype} of shape {occupied.shape}"
            )
        object.__setattr__(self, "support", _check_support(self.support, p.shape[0]))

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state.pop("_reading", None)
        return state

    def _knots(self) -> _Knots:
        """The reading's knots, computed once per instance."""
        found = self.__dict__.get("_reading")
        if found is None:
            lower, upper = (None, None) if self.support is None else self.support
            found = _knots_by_block(self.bin_probabilities, self.edges, self.occupied, lower, upper)
            object.__setattr__(self, "_reading", found)
        return cast(_Knots, found)

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
    def knot_passes(self) -> tuple[int, int]:
        """``(passes, capped)``: the knot iteration's passes and the rows left at its cap.

        A row still moving after the cap keeps its capped iterate.  Its segment masses are
        exact either way, because each segment is rescaled to its total probability.
        """
        knots = self._knots()
        return knots.passes, knots.capped

    def bin_of(self, values: FloatArray) -> IntArray:
        """Which bin each value falls in; ``-1`` outside the edges.

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
            ``(n,)`` bin index per row, and ``-1`` where the value is outside the edges.
        """
        a = np.asarray(values, dtype=float).reshape(-1)
        index = np.digitize(a, self.edges) - 1
        outside = (a < self.edges[0]) | (a > self.edges[-1])
        index = np.clip(index, 0, self.n_bins - 1)
        return np.asarray(np.where(outside, -1, index), dtype=np.int64)

    def density_at(self, values: FloatArray) -> FloatArray:
        """:math:`\\hat g(a_i \\mid W_i)` for one value per row; zero outside the support.

        The corrected log polygon of row ``i``'s bin probabilities (the module docstring),
        evaluated at ``values[i]``.  The knots are computed on the first call and kept, so
        this can be called as often as an estimand needs without refitting anything.

        Parameters
        ----------
        values : ndarray
            ``(n,)`` treatment values, one per row.

        Returns
        -------
        ndarray
            ``(n,)`` density at those values, and zero outside the edges and outside a
            declared support.
        """
        a = np.asarray(values, dtype=float).reshape(-1)
        if a.size != self.n:
            raise ValueError(f"density_at needs one value per row: got {a.size} for {self.n} rows")
        return _read(self._knots(), a).reshape(-1)

    def mass_between(self, lower: FloatArray, upper: FloatArray) -> FloatArray:
        """``(n,)`` values of :math:`\\int_{l_i}^{u_i} \\hat g(a \\mid W_i)\\, da`, in closed form.

        The reading is exponential-linear on each half of a bin, so the integral is a sum of
        closed forms over the pieces ``[lower, upper]`` overlaps.

        Parameters
        ----------
        lower : ndarray
            ``(n,)`` lower end per row, or a scalar.
        upper : ndarray
            ``(n,)`` upper end per row, or a scalar.

        Returns
        -------
        ndarray
            ``(n,)`` probability per row; zero where ``upper <= lower``.
        """
        return _mass(self._knots(), lower, upper)

    def integrated(self) -> FloatArray:
        """``(n,)`` values of :math:`\\int \\hat g(a \\mid W_i)\\, da`, which must all be one.

        Each segment of the reading keeps its total probability, and mass outside a
        declared support is projected away, so this is a check on the arithmetic rather than
        on the model; it is reported because a silent failure here would rescale every
        clever covariate by an amount no other diagnostic would see.

        Returns
        -------
        ndarray
            ``(n,)`` integral of the fitted density over the support, per row.
        """
        knots = self._knots()
        p = np.asarray(self.bin_probabilities, dtype=float)
        if self.support is None:
            return np.asarray(p.sum(axis=1), dtype=float)
        left, right = _effective(knots.edges, knots.lower, knots.upper, self.n)
        inside = np.where(right > left, p, 0.0).sum(axis=1)
        return np.asarray(inside / knots.scale, dtype=float)

    def flat_fraction(self, shifted: FloatArray, observed: FloatArray) -> float:
        """Share of moved rows whose observed and moved dose fall in one flat bin.

        A bin is *flat* for a row when the reading is constant on it: an empty bin, a bin
        with almost no mass, or a one-bin segment.  Where both doses share such a bin, the
        density ratio carries no information from the density (for a shift it is exactly
        one there, for ``Scale(c)`` it is ``1 / c``).

        Parameters
        ----------
        shifted : ndarray
            ``(n,)`` dose the policy assigns.
        observed : ndarray
            ``(n,)`` dose actually received.

        Returns
        -------
        float
            The share among rows whose dose moves; one when no dose moves, since a policy
            that moves nobody is invisible however the density reads.
        """
        moved_to = np.asarray(shifted, dtype=float).reshape(-1)
        a = np.asarray(observed, dtype=float).reshape(-1)
        moved = moved_to != a
        if not moved.any():
            return 1.0
        here = self.bin_of(a)
        there = self.bin_of(moved_to)
        same = moved & (here >= 0) & (here == there)
        if not same.any():
            return 0.0
        rows = np.flatnonzero(same)
        multi = self._knots().take(rows, here[rows])["multi"]
        return float(np.count_nonzero(np.logical_not(multi)) / np.count_nonzero(moved))

    def subset(self, index: Any) -> ConditionalDensity:
        """The same density on a row subset -- a bootstrap resample, a validation fold.

        The rows are sliced and the edges and occupancy kept, exactly as
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
        support = None if self.support is None else (self.support[0][idx], self.support[1][idx])
        return replace(self, bin_probabilities=self.bin_probabilities[idx], support=support)


@dataclass(frozen=True)
class MixedDensity:
    r"""The law of a dose with declared atoms, as a density against Lebesgue plus counting.

    Let the declared atoms be :math:`a_1, \ldots, a_K` and :math:`p_k(h) = P(A = a_k \mid h)`.
    The law of the dose is :math:`\sum_k p_k(h) \delta_{a_k} + (1 - \sum_k p_k(h)) f(\cdot
    \mid h)\,da`, with :math:`f` the density of the dose on the rows that hold no atom.
    With respect to Lebesgue measure plus counting measure on the atoms its density is
    :math:`p_k(h)` at an atom and :math:`(1 - \sum_k p_k(h)) f(a \mid h)` elsewhere.

    Like :class:`ConditionalDensity` it holds evaluated arrays and no model: one row per
    observation, out of fold.

    Parameters
    ----------
    atoms : tuple of float
        The declared atoms, in order.
    atom_probabilities : ndarray
        ``(n, K)`` fitted :math:`p_k(h_i)`, out of fold.
    continuous : ConditionalDensity or None
        The density :math:`f` of the dose given the history on the rows with no atom, or
        ``None`` at a node whose every observed dose is a declared atom.

    Attributes
    ----------
    n : int
    """

    atoms: tuple[float, ...]
    atom_probabilities: FloatArray
    continuous: ConditionalDensity | None

    def __post_init__(self) -> None:
        p = np.asarray(self.atom_probabilities, dtype=float)
        if p.ndim != 2 or p.shape[1] != len(self.atoms):
            raise ValueError(
                f"atom probabilities must be (n, {len(self.atoms)}); got shape {p.shape}"
            )
        if self.continuous is not None and self.continuous.n != p.shape[0]:
            raise ValueError("the continuous density and the atom probabilities differ in rows")

    @property
    def n(self) -> int:
        """Return the number of observations."""
        return int(np.asarray(self.atom_probabilities).shape[0])

    def _continuous_share(self) -> FloatArray:
        return np.clip(1.0 - np.asarray(self.atom_probabilities, dtype=float).sum(axis=1), 0.0, 1.0)

    def atom_of(self, values: FloatArray) -> IntArray:
        """The index of the declared atom each value equals, and ``-1`` elsewhere.

        Parameters
        ----------
        values : ndarray
            ``(n,)`` doses, one per row.

        Returns
        -------
        ndarray
            ``(n,)`` atom index per row.
        """
        v = np.asarray(values, dtype=float).reshape(-1)
        out = np.full(v.size, -1, dtype=np.int64)
        for index, atom in reversed(list(enumerate(self.atoms))):
            hit = np.abs(v - float(atom)) <= 1e-12 * max(1.0, abs(float(atom)))
            out = np.where(hit, index, out)
        return out

    def density_at(self, values: FloatArray) -> FloatArray:
        r"""The density against Lebesgue plus counting measure, at one value per row.

        Parameters
        ----------
        values : ndarray
            ``(n,)`` doses, one per row.

        Returns
        -------
        ndarray
            ``(n,)`` :math:`p_k(h_i)` at an atom and :math:`(1 - \sum_k p_k) f(a_i \mid h_i)`
            elsewhere.
        """
        v = np.asarray(values, dtype=float).reshape(-1)
        if v.size != self.n:
            raise ValueError(f"density_at needs one value per row: got {v.size} for {self.n} rows")
        index = self.atom_of(v)
        p = np.asarray(self.atom_probabilities, dtype=float)
        if self.continuous is None:
            elsewhere = np.zeros(v.size)
        else:
            elsewhere = self._continuous_share() * self.continuous.density_at(v)
        at_atom = p[np.arange(v.size), np.maximum(index, 0)] if p.shape[1] else np.zeros(v.size)
        return np.asarray(np.where(index >= 0, at_atom, elsewhere), dtype=float)

    def mass_between(self, lower: FloatArray, upper: FloatArray) -> FloatArray:
        r"""The probability that the dose has no atom and lies in ``[lower, upper]``, per row.

        :math:`(1 - \sum_k p_k(h_i)) \int_{l_i}^{u_i} f(s \mid h_i)\,ds`, with the integral
        of the continuous part in closed form (:meth:`ConditionalDensity.mass_between`).

        Parameters
        ----------
        lower : ndarray
            ``(n,)`` lower end per row.
        upper : ndarray
            ``(n,)`` upper end per row.

        Returns
        -------
        ndarray
            ``(n,)`` probability per row.
        """
        if self.continuous is None:
            return np.zeros(self.n)
        mass = self.continuous.mass_between(lower, upper)
        return np.asarray(self._continuous_share() * mass, dtype=float)

    def subset(self, index: Any) -> MixedDensity:
        """The same law on a row subset.

        Parameters
        ----------
        index : array_like
            Row positions or a boolean mask.

        Returns
        -------
        MixedDensity
            The same law over the selected rows.
        """
        idx = np.asarray(index)
        if idx.dtype == bool:
            idx = np.flatnonzero(idx)
        return MixedDensity(
            self.atoms,
            np.asarray(self.atom_probabilities)[idx],
            None if self.continuous is None else self.continuous.subset(idx),
        )


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


#: Out-of-fold densities at further conditioning designs: by key, ``(design, values)`` or
#: ``(design, values, support)``, with ``design`` the ``(n, p)`` covariates the density
#: conditions on, or a :class:`DesignRows` that builds them, and ``values`` the ``(n,)`` or
#: ``(n, m)`` doses it is evaluated at.  ``values`` may instead be an interval request
#: ``(lower, upper)`` of two such arrays, which returns the mean density over each interval.
#: ``support`` is the declared ``(lower, upper)`` on each row of that design.
PredictDesigns = Mapping[Hashable, tuple[Any, ...]]


@dataclass(frozen=True)
class _Grid:
    """The bins of one density fit, read from the rows it is fitted on.

    ``core`` is the core count of a grid the fit built itself and ``None`` for explicit
    edges.  ``values`` are the doses the grid was read from, for the refusal's advice.
    """

    edges: FloatArray
    bins: IntArray
    n_total: int
    n_hazards: int
    occupied: BoolArray
    core: int | None = None
    values: FloatArray | None = None


def _grid(
    treatment: FloatArray,
    mask: BoolArray | None,
    n_bins: int | None,
    edges: Sequence[float] | None,
    name: str | None,
    occupied: BoolArray | None = None,
) -> _Grid:
    """The edges, the occupancy and each row's bin, with the default and the edges checked.

    ``occupied`` travels with explicit ``edges``: a caller that reads its edges from another
    sample passes that sample's occupancy beside them.  Without it the occupancy is read
    from the rows the grid is read from.
    """
    grid_values = treatment if mask is None else treatment[mask]
    if occupied is not None and edges is None:
        raise TypeError("occupied= travels with edges=; pass both or neither")
    core = None
    if edges is None:
        core = default_density_bins(grid_values.size) if n_bins is None else n_bins
        grid = bin_edges(grid_values, core, name=name)
    else:
        grid = np.asarray(edges, dtype=float)
    if grid.ndim != 1 or grid.size < 3 or np.any(np.diff(grid) <= 0.0):
        raise DataError("edges= must be a strictly increasing sequence of at least 3 values")
    n_total = int(grid.size - 1)
    bins = np.clip(np.digitize(treatment, grid) - 1, 0, n_total - 1).astype(np.int64)
    mask_bins = (
        bin_occupancy(grid_values, grid)
        if occupied is None
        else np.asarray(occupied, dtype=bool).reshape(-1)
    )
    if mask_bins.shape != (n_total,):
        raise ValueError(f"occupied= needs one entry per bin ({n_total}); got {mask_bins.shape}")
    return _Grid(grid, bins, n_total, n_total - 1, mask_bins, core, grid_values)


def _table_at(
    learner: Learner, n_covariates: int, grid: _Grid, name: str | None
) -> Callable[[int], int] | None:
    """The bytes of the hazard table at another core count, on that count's refined grid."""
    if grid.core is None or grid.values is None:
        return None
    values = grid.values
    index_only = _index_only(learner)

    def table(k: int) -> int:
        edges = bin_edges(values, k, name=name)
        b = edges.size - 1
        bins = np.clip(np.digitize(values, edges) - 1, 0, b - 1)
        records = int(np.sum(np.minimum(bins, b - 2) + 1))
        width = _bin_block(np.zeros(1, dtype=np.int64), b - 1, index_only).shape[1]
        return records * (n_covariates + width) * 8

    return table


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
            table_at=_table_at(learner, n_covariates, grid, name),
            core=grid.core,
        )
        retained += a.size * grid.n_total * 8 + cost.kept


def _lookup(
    probabilities: FloatArray,
    edges: FloatArray,
    occupied: BoolArray,
    support: tuple[FloatArray, FloatArray] | None,
    values: FloatArray,
) -> FloatArray:
    """The reading of each row's bin probabilities at its ``(m,)`` values: ``(rows, m)``.

    The same reading as :meth:`ConditionalDensity.density_at`, for bin probabilities that
    are not kept, such as a fold's predictions at a counterfactual design.
    """
    p = np.asarray(probabilities, dtype=float)
    lower, upper = (None, None) if support is None else support
    knots = _knots_by_block(p, edges, occupied, lower, upper)
    return _read(knots, np.asarray(values, dtype=float).reshape(p.shape[0], -1))


def _lookup_mean(
    probabilities: FloatArray,
    edges: FloatArray,
    occupied: BoolArray,
    support: tuple[FloatArray, FloatArray] | None,
    lower: FloatArray,
    upper: FloatArray,
) -> FloatArray:
    """The mean of the reading over each row's ``(m,)`` intervals: ``(rows, m)``.

    ``mean * (upper - lower)`` is the closed-form integral of :func:`_lookup` over the
    interval.  A zero-length interval reads the density at its point.
    """
    p = np.asarray(probabilities, dtype=float)
    lo_support, hi_support = (None, None) if support is None else support
    knots = _knots_by_block(p, edges, occupied, lo_support, hi_support)
    lo = np.asarray(lower, dtype=float).reshape(p.shape[0], -1)
    hi = np.asarray(upper, dtype=float).reshape(p.shape[0], -1)
    out = np.zeros(lo.shape)
    point = _read(knots, lo)
    for j in range(lo.shape[1]):
        length = hi[:, j] - lo[:, j]
        mass = _mass(knots, lo[:, j], hi[:, j])
        safe = np.where(length > 0.0, length, 1.0)
        out[:, j] = np.where(length > 0.0, mass / safe, point[:, j])
    return out


@dataclass(frozen=True)
class _Request:
    """One entry of ``predict_designs``: a design, its values or interval, its support."""

    design: Any
    values: list[FloatArray]
    support: tuple[FloatArray, FloatArray] | None


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
    occupied: BoolArray | None = ...,
    support: tuple[FloatArray, FloatArray] | None = ...,
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
    occupied: BoolArray | None = ...,
    support: tuple[FloatArray, FloatArray] | None = ...,
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
    occupied: BoolArray | None = None,
    support: tuple[FloatArray, FloatArray] | None = None,
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

    ``edges=`` fixes the grid.  ``occupied=`` travels with it: a caller that reads the edges
    from another sample passes that sample's :func:`bin_occupancy` beside them, so that a
    density fitted on a subset breaks its segments at the same bins as the density fitted
    on every row.  ``edges=`` alone takes the occupancy of the rows the fit reads the grid
    from; ``occupied=`` without ``edges=`` is a :class:`TypeError`.

    ``support=`` is the declared support ``(lower, upper)`` on each row of ``covariates``.
    The hazard fit does not read it.  The returned density reads each row over its support
    (the module docstring), and is zero outside it.

    ``predict_designs`` evaluates the density at further conditioning designs, by key: each
    entry is ``(design, values)`` or ``(design, values, support)``, and the result holds the
    density at ``values`` given ``design``, one value per entry of ``values``, read with
    that design's declared support when it carries one.  ``values`` may also be an interval
    request ``(lower, upper)`` of two arrays of one shape; the result then holds the mean
    density over each interval, whose product with the interval's length is the exact
    integral of the reading.  Row ``i`` of every design is predicted by
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
    grid = _grid(a, mask, n_bins, edges, name, occupied)
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
        table_at=_table_at(learner, w.shape[1], grid, name),
        core=grid.core,
    )
    declared = _check_support(support, a.size)
    designs: dict[Hashable, _Request] = {}
    for key, entry in ({} if predict_designs is None else dict(predict_designs)).items():
        design, values = entry[0], entry[1]
        bounds = _check_support(entry[2] if len(entry) > 2 else None, a.size)
        interval = isinstance(values, tuple)
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
        parts = [np.asarray(v, dtype=float) for v in (values if interval else (values,))]
        if any(part.shape[0] != a.size for part in parts) or (
            interval and (len(parts) != 2 or parts[0].shape != parts[1].shape)
        ):
            raise ValueError(f"predict design {key!r} needs one row of values per unit")
        designs[key] = _Request(design, parts, bounds)

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
        for key, request in designs.items():
            design = request.design
            held_out = (
                np.asarray(design(test, w[test]), dtype=float)
                if callable(design)
                else np.asarray(design, dtype=float)[test]
            )
            probabilities = _probabilities_from_hazards(
                _hazard_matrix(model, held_out, n_hazards, index_only)
            )
            bounds = (
                None
                if request.support is None
                else (request.support[0][test], request.support[1][test])
            )
            if len(request.values) == 2:
                out[key] = _lookup_mean(
                    probabilities,
                    grid.edges,
                    grid.occupied,
                    bounds,
                    request.values[0][test],
                    request.values[1][test],
                )
            else:
                out[key] = _lookup(
                    probabilities, grid.edges, grid.occupied, bounds, request.values[0][test]
                )
        return out

    probabilities = np.empty((a.size, n_total), dtype=float)
    predictions = {
        key: np.zeros((a.size, int(np.prod(request.values[0].shape[1:], dtype=int))))
        for key, request in designs.items()
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

    density = ConditionalDensity(probabilities, grid.edges, grid.occupied, declared)
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
        knot_passes=density.knot_passes,
        core_bins=grid.core,
    )
    if predict_designs is None:
        return density, report
    shaped = {
        key: predictions[key].reshape(request.values[0].shape) for key, request in designs.items()
    }
    return density, report, shaped


def warn_if_flat(
    density: ConditionalDensity,
    shifted: FloatArray,
    observed: FloatArray,
    component: str | None = None,
) -> float:
    """Warn when most moved rows keep their dose inside one flat bin, and return the share.

    Separated from the fit so the check happens where the *policy* is known: the density
    knows its own resolution, but only an intervention knows how far it means to move.  The
    share is :meth:`ConditionalDensity.flat_fraction`.

    ``component`` names the dose of a vector node.  Its caller passes only the rows whose
    dose the policy changes, so a gated map is not charged for the rows it leaves alone, and
    the message names the dose rather than the clever covariate: another component of the
    node can move on the same row, so that row's ratio need not be one.

    Parameters
    ----------
    density : ConditionalDensity
        The fitted density.
    shifted : ndarray
        ``(n,)`` dose the policy assigns.
    observed : ndarray
        ``(n,)`` dose received.
    component : str or None
        The dose's name at a vector node.

    Returns
    -------
    float
        The flat share.
    """
    flat = density.flat_fraction(shifted, observed)
    if flat > _FLAT_WARNING:
        grid = f"({density.n_bins} bins over [{density.edges[0]:.3g}, {density.edges[-1]:.3g}])"
        if not np.any(np.asarray(shifted, dtype=float) != np.asarray(observed, dtype=float)):
            _warn(
                "the policy changes no row's dose, so its density ratio is one on every row "
                "and the intervention is invisible: it estimates the natural course. Check the "
                "policy's cap or gate if it was meant to move some doses.",
                UserWarning,
                stacklevel=3,
            )
            return flat
        if component is None:
            message = (
                f"the policy keeps {flat:.1%} of the rows it moves inside one flat bin of the "
                f"estimated density {grid}: an empty bin, a bin with almost no mass, or an "
                "isolated bin, where the density is read as constant. For those rows the "
                "density ratio is fixed by the bin alone (exactly one for a shift), so the "
                "intervention is invisible rather than merely noisy."
            )
        else:
            message = (
                f"the policy changes {component!r} on {np.size(shifted)} rows, and keeps "
                f"{flat:.1%} of them inside one flat bin of its estimated density {grid}: an "
                "empty bin, a bin with almost no mass, or an isolated bin, where the density "
                f"is read as constant. On those rows the density of {component!r} cannot tell "
                "the changed dose from the natural one, and that change is invisible rather "
                "than merely noisy."
            )
        _warn(
            f"{message} Use more bins, or a larger change of dose, or read the estimate as the "
            "effect of a policy the bins can resolve.",
            UserWarning,
            stacklevel=3,
        )
    return flat
