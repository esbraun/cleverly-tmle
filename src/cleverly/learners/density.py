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

The construction is Diaz & van der Laan (2011); it is what R's ``haldensify`` fits, and
what ``txshift`` consumes.

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

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .._typing import FloatArray, IntArray, Learner
from ..exceptions import DataError
from ..utils.parallel import map_parallel
from ._fitting import fit_learner, predict_mean
from .crossfit import Folds
from .super_learner import SuperLearnerDiagnostics

__all__ = [
    "ConditionalDensity",
    "DensityDiagnostics",
    "bin_edges",
    "fit_conditional_density",
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


def bin_edges(values: FloatArray, n_bins: int) -> FloatArray:
    """Equal-mass bin edges for ``values``: ``n_bins + 1`` of them, strictly increasing.

    Empirical quantiles, so each bin holds about the same number of observations rather
    than the same width.  That is the right default for a density whose tails are the
    part positivity depends on: equal-width bins put almost no data in the tails, and the
    hazard there is then estimated from a handful of rows.

    Ties can make two quantiles coincide; duplicates are dropped and the bin count falls,
    which is reported rather than worked around.
    """
    a = np.asarray(values, dtype=float).reshape(-1)
    if n_bins < 2:
        raise DataError(f"a binned density needs at least 2 bins; got {n_bins}")
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(a, quantiles))
    if edges.size < 3:
        raise DataError(
            f"the treatment has too few distinct values to form {n_bins} bins; "
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

    Attributes
    ----------
    n_bins:
        Bins actually used, after duplicate quantiles were dropped.
    edges:
        The bin boundaries.
    min_density:
        Smallest :math:`\\hat g(A_i \\mid W_i)` over the sample.  The denominator of every
        clever covariate, so the number a positivity problem shows up in first.
    integrated:
        ``(min, max)`` of :math:`\\int \\hat g(a \\mid W_i)\\,da` over rows.  Both should be
        one to machine precision; anything else is a bug in the hazard product, not a
        modelling matter.
    learner:
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

    Attributes
    ----------
    bin_probabilities:
        ``(n, B)``, rows summing to one.  Out-of-fold predictions.
    edges:
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
        return int(self.bin_probabilities.shape[0])

    @property
    def n_bins(self) -> int:
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
        """
        return np.asarray(self.bin_probabilities.sum(axis=1), dtype=float)

    def crossing_fraction(self, shifted: FloatArray, observed: FloatArray) -> float:
        """Share of rows a shift actually moves into a different bin.

        The resolution guard described in the module docstring: a shift that moves nobody
        across an edge has a clever covariate of exactly one everywhere and estimates
        nothing, however clean the fit looks.
        """
        return float(np.mean(self.bin_of(shifted) != self.bin_of(observed)))

    def subset(self, index: Any) -> ConditionalDensity:
        """The same density on a row subset -- a bootstrap resample, a validation fold.

        The rows are sliced and the edges kept, exactly as
        :meth:`~cleverly.interventions.RegimeSet.subset` slices its densities: re-deriving
        the edges from the resample would give the subset a different discretisation from
        the fit it is meant to be a replicate of.
        """
        idx = np.asarray(index)
        if idx.dtype == bool:
            idx = np.flatnonzero(idx)
        return replace(self, bin_probabilities=self.bin_probabilities[idx])


# ------------------------------------------------------------------- fitting


def _bin_block(index: IntArray, n_hazards: int) -> FloatArray:
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
    indicators = (index.reshape(-1, 1) == np.arange(1, n_hazards).reshape(1, -1)).astype(float)
    return np.column_stack([index.astype(float), indicators])


def _long_expansion(
    covariates: FloatArray, bins: IntArray, n_hazards: int
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
    design = np.column_stack([covariates[rows], _bin_block(index, n_hazards)])
    target = (index == bins[rows]).astype(float)
    return design, target, rows


def _hazard_matrix(model: Learner, covariates: FloatArray, n_hazards: int) -> FloatArray:
    """``(n, n_hazards)`` predicted hazards, one column per bin the model covers."""
    n = covariates.shape[0]
    columns = [
        predict_mean(
            model,
            np.column_stack([covariates, _bin_block(np.full(n, b, dtype=np.int64), n_hazards)]),
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


def fit_conditional_density(
    learner: Learner,
    covariates: FloatArray,
    treatment: FloatArray,
    weights: FloatArray,
    folds: Folds,
    *,
    n_bins: int = 20,
    edges: Sequence[float] | None = None,
    groups: IntArray | None = None,
    n_jobs: int = 1,
) -> tuple[ConditionalDensity, DensityDiagnostics]:
    """Cross-fit :math:`g(a \\mid W)` and return it evaluated at every row.

    The fold loop is deliberately *not* :func:`cross_fit_predictions`.  That helper's
    contract is ``n`` rows in, ``n`` predictions out; a hazard model trains on a long
    expansion of :math:`\\sum_i (b_i + 1)` records and produces an ``(n, B)`` matrix, so
    bending the shared helper would put two orthogonal modes on one function.  What is
    shared is what matters: the same :class:`~cleverly.learners.Folds`, so a row's density
    and its outcome regression come from the same split, and
    :func:`~cleverly.learners.fit_learner`, so weights, cluster codes and screening
    pipelines route exactly as they do for every other nuisance.
    """
    w = np.asarray(covariates, dtype=float)
    a = np.asarray(treatment, dtype=float).reshape(-1)
    sample_weight = np.asarray(weights, dtype=float).reshape(-1)
    grid = bin_edges(a, n_bins) if edges is None else np.asarray(edges, dtype=float)
    if grid.ndim != 1 or grid.size < 3 or np.any(np.diff(grid) <= 0.0):
        raise DataError("edges= must be a strictly increasing sequence of at least 3 values")

    n_total = int(grid.size - 1)
    n_hazards = n_total - 1
    bins = np.clip(np.digitize(a, grid) - 1, 0, n_total - 1).astype(np.int64)

    def fit_on(rows: IntArray) -> Learner:
        design, target, source = _long_expansion(w[rows], bins[rows], n_hazards)
        return fit_learner(
            learner,
            design,
            target,
            sample_weight[rows][source],
            groups=None if groups is None else np.asarray(groups)[rows][source],
            warn_unweighted=False,
        )

    probabilities = np.empty((a.size, n_total), dtype=float)
    diagnostics: list[SuperLearnerDiagnostics] = []

    if folds.is_single:
        model = fit_on(np.arange(a.size, dtype=np.int64))
        probabilities[:] = _probabilities_from_hazards(_hazard_matrix(model, w, n_hazards))
        found = getattr(model, "diagnostics_", None)
        if found is not None:
            diagnostics.append(found)
    else:

        def run_fold(train: IntArray, test: IntArray) -> tuple[IntArray, FloatArray, Any]:
            model = fit_on(train)
            hazards = _hazard_matrix(model, w[test], n_hazards)
            return test, _probabilities_from_hazards(hazards), getattr(model, "diagnostics_", None)

        for test, values, found in map_parallel(
            run_fold, [(train, test) for train, test in folds], n_jobs=n_jobs
        ):
            probabilities[test] = values
            if found is not None:
                diagnostics.append(found)

    density = ConditionalDensity(probabilities, grid)
    integrated = density.integrated()
    observed = density.density_at(a)
    return density, DensityDiagnostics(
        n_bins=n_total,
        edges=grid,
        min_density=float(observed.min()),
        integrated=(float(integrated.min()), float(integrated.max())),
        learner=tuple(diagnostics),
    )


def warn_if_unresolved(
    density: ConditionalDensity, shifted: FloatArray, observed: FloatArray
) -> float:
    """Warn when a shift moves too few rows across a bin edge, and return the share.

    Separated from the fit so the check happens where the *policy* is known: the density
    knows its own resolution, but only an intervention knows how far it means to move.
    """
    crossing = density.crossing_fraction(shifted, observed)
    if crossing < _CROSSING_WARNING:
        warnings.warn(
            f"the shift moves only {crossing:.1%} of rows across a bin edge of the "
            f"estimated density ({density.n_bins} bins over "
            f"[{density.support[0]:.3g}, {density.support[1]:.3g}]). A binned density is "
            "constant within a bin, so for the rest the clever covariate is exactly one "
            "and the intervention is invisible rather than merely noisy. Use more bins, "
            "or a larger shift, or read the estimate as the effect of a policy the bin "
            "width can resolve.",
            UserWarning,
            stacklevel=3,
        )
    return crossing
