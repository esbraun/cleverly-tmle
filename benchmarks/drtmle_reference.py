r"""The reduced regressions at their population limits, on a continuous law.

``docs/roadmap.md``'s **E2**.  ``tests/unit/test_oracle_reductions.py`` builds this on the
exact law, where the conditioning variable takes three values and a conditional expectation
is a finite sum.  Here it is built on ``linear_dgp``, where it is not -- and the whole
difficulty is that the result is a **numerical reference and not an oracle**.  Nothing in
this module should be called one.

**What the object is, and it is univariate.**
:mod:`cleverly.estimators.reduced` states it: *"Each is univariate: a regression on one
column, that column being the other nuisance's out-of-fold prediction."*  So for outer fold
:math:`k` and arm :math:`a`, relative to the **current targeted** pair,

.. math::

    Q_r(a, w)    &= E_0[\, \bar Q_0(a, W) - \bar Q^{*(k)}(a, W)
                        \mid A = a,\ g^{*(k)}(a|W) = g^{*(k)}(a|w) ] \\
    g_{r,1}(a|w) &= E_0[\, g_0(a|W) \mid \bar Q^{*(k)}(a, W) = \bar Q^{*(k)}(a, w) ] \\
    g_{r,2}(a|w) &= E_0[\, \{g_0(a|W) - g^{*(k)}_b(a|W)\}/g^{*(k)}_b(a|W)
                        \mid \bar Q^{*(k)}(a, W) = \bar Q^{*(k)}(a, w) ]

one conditioning column each.  The two-scalar conditioning in
:func:`~benchmarks.drtmle_remainder.branch_products` is appendix A's and B's ``0n`` limits,
which are a **different object** -- ``docs/roadmap.md``'s E2 paragraph asserted otherwise and
is corrected there.

**Why this needs no change under** ``src/``.  Item 24 said an oracle here "needs the fitted
learners, and nothing here keeps them".  That has been false since C2:
:class:`~cleverly.estimators._nuisance.CompanionEstimates` holds ``outcome`` and
``propensity`` one copy per outer fold at rows declared before the fit, produced by the same
model objects the production arrays came from, and
:attr:`~cleverly.estimators.targeting.ReductionSpec.refit` is handed them at the *current*
targeted pair on every round.  So the reference is fitted where :math:`P_0` is known and
predicted at the production design values the fit already carries.  The residue is a
**declaration** constraint -- the grid is fixed before the fit -- and that is a virtue, since
it makes the reference conditionally fixed given the fit.

**Two of the three integrals are exact and only one is estimated.**  Under
:func:`~benchmarks.drtmle_remainder.quadrature_frame` every Sobol point contributes two rows,
one per arm, carrying weight :math:`g_0(a|W)/\text{points}`.  So :math:`A` is integrated in
closed form -- ``qr``'s ``| A = a`` is a *weight* and not a subsample, and ``gr1``'s target
:math:`E[1_a \mid W]` is :math:`g_0(a|W)` exactly -- and :math:`Y` never appears, because the
frame carries :math:`\bar Q_0(a, W)` in its outcome column.  What is left is a **smoothing
over one index** and a finite point count over :math:`W`.

**Those two are what a fidelity gate still has to bound, and this module does not bound
them.**  It builds the instrument; the gate is the next piece of E2 and is owed before any
paired comparison is read.  What it has to be is stated here so that a later revision cannot
quietly settle for less: an exact-law control where the conditioning is discrete and the
answer is a finite sum; a **held-out weighted risk** on an independent, finer randomised-QMC
companion, whose cross term vanishes because the reference is a weighted :math:`L_2`
projection -- which is why :class:`SplineProjection` is pinned linear in its target; and a
randomisation budget over independent scrambles.  Until those exist, a paired
reference-against-``glm`` comparison is not evidence about the reduction learner.

**What must not be reused as the gate.**  Not the movement between two resolutions.  That is
the statistic ``docs/roadmap.md`` withdrew for the quadrature ladder and then for the
branches themselves, and E2 inheriting it would rebuild the mistake it exists to repair.  A
knot ladder is printed as a *stability* column and never as an error.

**And randomisation gates only half of it.**  Independent scrambles make the reference's
**quadrature** error mean-zero and estimable by replication -- E1b's device, transported.
They cannot see the **smoothing** bias: every randomisation shares the knot count and the
basis, so the across-scramble spread is orthogonal to a bias in the basis.  That half is what
the analytic-index control and the held-out risk are for.

**What E2R adds here, and why it is an addition to the instrument rather than to the rule.**
E2 shipped one rung for three regressions and used gate B to check it; three cells of four
came back ``unresolved`` because a coarser rung beat it.  So the rung is now **selected**
against a measured ranking, and two things follow that this module has to supply.

*The reference is a mapping rather than a single object.*  :func:`reference_reductions` takes
either, and a mapping routes one :class:`Reference` per reduced regression -- which subsumes
the case where they agree and does not assume it.

*The ranking has to be taken on the objects the fit consumes, not only on the three
regressions.*  :func:`~cleverly.inference.influence.reduced_correction_parts` divides: it
builds ``H_2 = g_{r,2}/g_{r,1}`` and ``H_3 = q_r/g``, per arm, at the **bounded**
denominators.  :data:`METRICS` names those two beside the three componentwise ones and
:func:`composite_denominators` supplies each divisor with the two columns item 25 reads --
its margin and its truncation rate -- at the same ``g_bounds`` the fit used, since at any
other bound it is a loss on a different object.

**A composite loss is the same held-out risk under a different weight, and that is exactly
why it decomposes.**  :func:`held_out_risk`'s cross term vanishes for *any* weight that is
measurable in the conditioning index, because ``E_0[w(T - m) | U] = 0`` is what defines
:math:`m` under :math:`w`; multiplying by :math:`\varphi(U) \ge 0` leaves it zero.  Both
divisors are such functions: ``qr``'s index **is** the mechanism ``H_3`` divides by, and
``gr1`` is a regression on ``gr2``'s own index.  So scoring ``qr`` under
:math:`w/g^{*2}_b` is scoring the composite ``H_3`` itself, with the irreducible term still
common to every candidate -- which is the one property a ratio of two risks does not have.

**What a composite loss does *not* see is the divisor's own error**, and this is stated here
rather than left implicit.  The divisor is the one the **fit under measurement** used -- it is
the same array for every candidate the ranking compares, which is what keeps the difference a
difference of squared weighted errors rather than a difference of two irreducible terms.  So
the composite metric re-weights the *numerator's* error towards the rows where the estimator
actually divides by something small; the divisor's own quality is what the componentwise
``gr1`` metric and the margin columns beside it are for.  Neither substitutes for the other,
which is the same sentence gates B and C already carry about each other.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, Protocol

import numpy as np
from sklearn.preprocessing import SplineTransformer

from cleverly import DRTMLE
from cleverly.estimators.reduced import ReducedSet, reduced_designs

# The benchmarks package is a checkout rather than an installed distribution, so a plain
# `python benchmarks/drtmle_reference.py` has to find its siblings the way its neighbours do.
if __package__ in (None, ""):  # pragma: no cover - only on the direct-script path
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.drtmle_remainder import _arm_probability, _latent

__all__ = [
    "KNOT_LADDER",
    "METRICS",
    "POINTS_PER_PARAMETER",
    "ArmTruth",
    "Denominator",
    "EqualCountBins",
    "Metric",
    "Reference",
    "ReferenceReductionDRTMLE",
    "SaturatedCells",
    "SplineProjection",
    "arm_truth",
    "composite_denominators",
    "fit_mask",
    "fold_targets",
    "held_out_risk",
    "metric_weights",
    "per_reduction",
    "reference_reductions",
]

#: The knot counts the reference is read at.  A **ladder**, and read as a stability column
#: rather than as an error: the movement between two rungs of a refinement is what
#: ``docs/roadmap.md`` withdrew twice, and this module does not get to use it a third time.
#: What has to bound the smoothing half is an exact-law control and a held-out risk on an
#: independent companion, neither of which is a refinement difference -- see the module
#: docstring, and note that neither is built yet.
KNOT_LADDER = (8, 16, 32)

#: How many companion rows every basis column must have behind it.  A budget rather than a
#: taste: the reference's variance is what a near-interpolating basis buys in exchange for
#: its bias, and ``docs/drtmle/validation-plan.md`` records the same coupling for the binned
#: limits -- "a bin count raised without the rows behind it drives every branch toward its
#: own target and reports a spuriously small one".  Refused rather than warned about.
POINTS_PER_PARAMETER = 64


class Reference(Protocol):
    """A weighted projection of one target onto one index, evaluable at new index values.

    Deliberately not a :class:`~cleverly.learners.Learner`: nothing here is cross-fitted,
    selected or tuned.  A reference is a *deterministic* function of the population grid it
    is handed, which is what keeps A1b's stability condition free -- see
    ``docs/drtmle/theorem-concordance.md`` §15's ``(S)`` row, which is free for a fixed-basis
    smoother and not free for anything selecting a split or a bandwidth from the data.
    """

    def fit(
        self, index: np.ndarray, target: np.ndarray, weights: np.ndarray
    ) -> FittedReference: ...

    @property
    def label(self) -> str: ...


class FittedReference(Protocol):
    """What :meth:`Reference.fit` returns: a callable on new index values."""

    def __call__(self, index: np.ndarray) -> np.ndarray: ...


class SplineProjection:
    """Weighted least squares on a cubic B-spline basis in the index.  **The reference.**

    Chosen over a finer regressogram and over local-linear smoothing, and the reasons are
    recorded because each rejection is a claim:

    * a **regressogram**'s bias is :math:`O(B^{-1})` pointwise and no better at the boundary,
      which is where :func:`~benchmarks.drtmle_remainder.conditional_mean`'s own docstring
      records a fitted mechanism piles up.  The :math:`O(B^{-2})` inner-product argument in
      ``benchmarks/drtmle_tier2.py`` does **not** transfer: it is about a regressogram
      integrated against a fixed smooth weight, and a reference injected into equations (9)
      and (10) passes through a Newton solve weighted by :math:`1/\\hat g` and
      :math:`1_a g_{r,2}/g_{r,1}`, neither smooth at the bin scale nor sign-definite;
    * **local-linear** would fix the boundary but needs a declared bandwidth *sequence* with
      the status ``drtmle_tier2.bandwidth`` has -- and that module's own scan records that no
      bandwidth constant made its leading-order prediction hold on this law.  Taking on that
      liability buys nothing a spline does not already give;
    * a **growing spline basis** is what ``docs/roadmap.md`` already names as E2b's candidate,
      so sharing it makes any later comparison a point count against a sample size rather than
      a contest between estimator families.  Its ladder is over an integer, so there is no
      continuous constant to commit to.

    Built on the same :class:`~sklearn.preprocessing.SplineTransformer` the package's own
    ``SL.gam`` uses (``src/cleverly/learners/library.py``), whose ``n_knots`` is fixed at 4
    there and is the resolution knob here.
    """

    def __init__(self, n_knots: int = KNOT_LADDER[1], degree: int = 3) -> None:
        self.n_knots = int(n_knots)
        self.degree = int(degree)

    @property
    def label(self) -> str:
        return f"spline({self.n_knots})"

    @property
    def n_parameters(self) -> int:
        """Basis columns plus the intercept, which is what the budget is read against."""
        return self.n_knots + self.degree - 1 + 1

    def fit(self, index: np.ndarray, target: np.ndarray, weights: np.ndarray) -> _FittedSpline:
        index = np.asarray(index, dtype=float).reshape(-1, 1)
        target = np.asarray(target, dtype=float).reshape(-1)
        weights = np.asarray(weights, dtype=float).reshape(-1)
        if index.shape[0] < POINTS_PER_PARAMETER * self.n_parameters:
            raise ValueError(
                f"{self.label} has {self.n_parameters} parameters and would be fitted on "
                f"{index.shape[0]} rows, below the declared budget of "
                f"{POINTS_PER_PARAMETER} rows per parameter. A reference fitted thinner than "
                "its own budget reports its variance as the target's structure; refuse rather "
                "than widen the budget after the fact"
            )
        # Knots at quantiles of the index rather than uniformly: a fitted mechanism piles up,
        # which is the same reason `conditional_mean` takes equal-count bins.
        basis = SplineTransformer(
            n_knots=self.n_knots, degree=self.degree, knots="quantile", include_bias=True
        )
        design = basis.fit_transform(index)
        root = np.sqrt(np.maximum(weights, 0.0))
        coefficients, *_ = np.linalg.lstsq(design * root[:, None], target * root, rcond=None)
        return _FittedSpline(basis, np.asarray(coefficients, dtype=float))


class _FittedSpline:
    """:class:`SplineProjection`'s fitted half, kept apart so the spec stays a value."""

    def __init__(self, basis: SplineTransformer, coefficients: np.ndarray) -> None:
        self._basis = basis
        self._coefficients = coefficients

    def __call__(self, index: np.ndarray) -> np.ndarray:
        design = self._basis.transform(np.asarray(index, dtype=float).reshape(-1, 1))
        return np.asarray(design @ self._coefficients, dtype=float)


class EqualCountBins:
    """A regressogram, kept as the **negative control** the fidelity gate must reject.

    Not a candidate reference.  It is here so that "the gate rejects a reference that is too
    coarse" is a test rather than a hope, and so that a deliberately coarse arm can be shown
    to reach a *close final estimate* while failing the gate -- which is the one shape of
    result that would otherwise tempt a reader to skip the gate.
    """

    def __init__(self, bins: int = 8) -> None:
        self.bins = int(bins)

    @property
    def label(self) -> str:
        return f"bins({self.bins})"

    def fit(self, index: np.ndarray, target: np.ndarray, weights: np.ndarray) -> _FittedBins:
        index = np.asarray(index, dtype=float).reshape(-1)
        target = np.asarray(target, dtype=float).reshape(-1)
        weights = np.asarray(weights, dtype=float).reshape(-1)
        edges = np.quantile(index, np.linspace(0.0, 1.0, self.bins + 1)[1:-1])
        codes = np.searchsorted(edges, index, side="right")
        totals = np.bincount(codes, weights=target * weights, minlength=self.bins)
        counts = np.bincount(codes, weights=weights, minlength=self.bins)
        fallback = float(np.dot(target, weights) / weights.sum()) if weights.sum() > 0 else 0.0
        cells = np.where(counts > 0.0, totals / np.where(counts > 0.0, counts, 1.0), fallback)
        return _FittedBins(edges, cells)


class _FittedBins:
    def __init__(self, edges: np.ndarray, cells: np.ndarray) -> None:
        self._edges = edges
        self._cells = cells

    def __call__(self, index: np.ndarray) -> np.ndarray:
        codes = np.searchsorted(self._edges, np.asarray(index, dtype=float).reshape(-1), "right")
        return np.asarray(self._cells[codes], dtype=float)


class SaturatedCells:
    """An exact weighted cell mean over the **distinct** index values.

    The exact-law control's smoother, and exact there rather than approximate: when the index
    takes finitely many values, conditioning on it is conditioning on a discrete variable and
    the conditional expectation is a finite sum.  A spline is *not* exact in that case, which
    is why the control needs its own smoother rather than reusing the reference.
    """

    #: Distinct index values are matched to this many decimals.  The index is a nuisance
    #: prediction, so two rows of one cell agree to floating point and not to the bit.
    TOLERANCE = 12

    @property
    def label(self) -> str:
        return "saturated"

    def fit(self, index: np.ndarray, target: np.ndarray, weights: np.ndarray) -> _FittedCells:
        index = np.round(np.asarray(index, dtype=float).reshape(-1), self.TOLERANCE)
        target = np.asarray(target, dtype=float).reshape(-1)
        weights = np.asarray(weights, dtype=float).reshape(-1)
        levels, codes = np.unique(index, return_inverse=True)
        totals = np.bincount(codes, weights=target * weights, minlength=levels.size)
        counts = np.bincount(codes, weights=weights, minlength=levels.size)
        fallback = float(np.dot(target, weights) / weights.sum()) if weights.sum() > 0 else 0.0
        cells = np.where(counts > 0.0, totals / np.where(counts > 0.0, counts, 1.0), fallback)
        return _FittedCells(levels, cells, fallback, self.TOLERANCE)


class _FittedCells:
    def __init__(
        self, levels: np.ndarray, cells: np.ndarray, fallback: float, tolerance: int
    ) -> None:
        self._levels = levels
        self._cells = cells
        self._fallback = fallback
        self._tolerance = tolerance

    def __call__(self, index: np.ndarray) -> np.ndarray:
        wanted = np.round(np.asarray(index, dtype=float).reshape(-1), self._tolerance)
        position = np.searchsorted(self._levels, wanted)
        position = np.clip(position, 0, self._levels.size - 1)
        hit = self._levels[position] == wanted
        return np.where(hit, self._cells[position], self._fallback)


# --------------------------------------------------------------- the provider


def fit_mask(name: str, indicator: np.ndarray) -> np.ndarray | None:
    """Which rows the reference is *fitted* on, mirroring ``fit_reduced``'s own rule.

    ``qr`` alone carries a mask, and it is the whole of the ``| A = a`` in its definition --
    what makes the limit weighted by :math:`P(W) g_0(a|W)` rather than by :math:`P(W)`.
    ``gr1`` and ``gr2`` take every row, and their arm-integration happens through the row
    weights instead: under :func:`~benchmarks.drtmle_remainder.quadrature_frame` the two rows
    of one Sobol point carry :math:`g_0(1|W)` and :math:`g_0(0|W)`, so a weighted average of
    the indicator over a conditioning set *is* :math:`E[g_0(a|W) \\mid \\cdot]` exactly.
    Getting this backwards is silent: every array stays in range and the limit answers for
    the wrong measure.
    """
    return (indicator == 1.0) if name == "qr" else None


@dataclass(frozen=True)
class Metric:
    """One thing gate B ranks candidates on: a reduced regression under one weighting.

    ``reduction`` is which of the three regressions is fitted and scored, and ``divisor``
    names the composite whose denominator re-weights the risk -- ``None`` for the three
    componentwise metrics, which are the risks E2 ranked on and which stay.

    **A metric is not a regression**, and keeping the two words apart is what stops a table
    reading as though there were five reduced regressions.  There are three; ``h3`` is ``qr``
    scored where the estimator divides by :math:`g^*_b`, and ``h2`` is ``gr2`` scored where it
    divides by :math:`g_{r,1,b}`.
    """

    name: str
    reduction: str
    divisor: str | None = None


#: The five metrics, in the order every table prints them: the three componentwise risks, then
#: the two composites the fit's correction actually consumes.
#:
#: **Both halves are kept because neither contains the other.**  A componentwise risk is what
#: Theorem 1's premise is about -- it asks the three regressions to be consistent -- and it is
#: what detects an individually bad reduction and stops two errors cancelling.  A composite
#: risk is what :func:`~cleverly.inference.influence.reduced_correction_parts` divides by, so
#: it is what a *fit* is sensitive to, and it weights the same error towards the rows where the
#: denominator is small.  ``docs/roadmap.md``'s E2R states it as *componentwise risks are
#: theorem-relevant and incomplete, not wrong*.
METRICS = (
    Metric("qr", "qr"),
    Metric("gr1", "gr1"),
    Metric("gr2", "gr2"),
    Metric("h3", "qr", "h3"),
    Metric("h2", "gr2", "h2"),
)


@dataclass(frozen=True)
class Denominator:
    """One composite's divisor at the companion rows, and the two columns item 25 reads.

    Attributes
    ----------
    values:
        The **bounded** array, which is what the fit divides by.  A composite loss taken at
        any other bound is a loss on a different object, which is why this is built from the
        fit's own ``g_bounds`` rather than from a fresh choice.
    margin:
        How near either bound the **untruncated** array comes, as a fraction of the interval:
        ``min_i min(d_i - lo, hi - d_i) / (hi - lo)``.  Signed, exactly as
        :attr:`~cleverly.validation.drtmle.CorrectionRow.gr1_margin` is signed and for the
        same reason -- a value at or below zero says the truncation is doing something to the
        denominator, and a threshold belongs to whoever reads the column and not inside it.
    truncated:
        The share of this divisor's rows -- every companion row, since a divisor is predicted
        everywhere -- that the truncation moved.  Reported beside the margin because the two say
        different things: a margin at or below zero says *some* row was clipped and this says how
        many.
    """

    values: np.ndarray
    margin: float
    truncated: float


def _denominator(raw: np.ndarray, bounds: tuple[float, float]) -> Denominator:
    """One divisor, truncated, with its margin and truncation share off the raw array."""
    values = np.asarray(raw, dtype=float).reshape(-1)
    low, high = float(bounds[0]), float(bounds[1])
    span = high - low
    nearest = float(np.minimum(values - low, high - values).min()) if values.size else float("nan")
    outside = (values < low) | (values > high)
    return Denominator(
        values=np.clip(values, low, high),
        margin=nearest / span if span > 0 else float("nan"),
        truncated=float(outside.mean()) if values.size else float("nan"),
    )


def composite_denominators(
    current: Any,
    *,
    fold: int,
    arm: float,
    g_bounds: tuple[float, float],
    reduced: Any = None,
) -> dict[str, Denominator]:
    r"""What the fit divides by, per composite, at fold ``fold``'s companion rows.

    ``h3``'s divisor is :math:`g^{*(k)}_b(a|W)` and ``h2``'s is :math:`g^{(k)}_{r,1,b}(a|w)`,
    both read off **the state being measured** rather than rebuilt: the point of a composite
    metric is that it weights an error the way the estimator's own arrays do, and a divisor
    fitted fresh here would weight it the way this module would have.

    ``reduced`` is the companion :class:`~cleverly.estimators.reduced.ReducedSet` whose
    ``gr1`` supplies ``h2``'s divisor, defaulting to the one the state carries.  The caller
    passes it explicitly where the state's own copy is a refit behind the arrays the reported
    correction was built from -- which is the case at the alternation's exit, and is why
    ``benchmarks/drtmle_reference_study.py`` records the produced set beside the state.

    **The divisor is candidate-free by construction**, and every claim about the composite
    metrics rests on that: one array for every candidate the ranking compares, so the
    irreducible term of :func:`held_out_risk` stays common to them and the difference of two
    risks stays a difference of squared weighted errors.
    """
    companion = current.companion
    # The arm's own column of a per-arm array, which `ReducedSet` keys by position in `arms`
    # exactly as `Submodel` does -- never by a design column, which is a different index.
    index = tuple(current.arms).index(float(arm))
    mechanism = np.asarray(companion.propensity[fold].arm(arm), dtype=float)
    reductions = reduced if reduced is not None else companion.reduced[fold]
    return {
        "h3": _denominator(mechanism, g_bounds),
        "h2": _denominator(np.asarray(reductions.gr1, dtype=float)[:, index], g_bounds),
    }


def metric_weights(
    mass: np.ndarray, denominators: Mapping[str, Denominator]
) -> dict[str, np.ndarray]:
    r"""The scoring weight of every metric in :data:`METRICS`, from one row-weight vector.

    A componentwise metric scores at the law's own measure :math:`w`; a composite one scores
    at :math:`w/d^2`, which is what turns a risk on the numerator into a risk on the ratio.

    **The cross term still vanishes**, which is the only reason this is a gate.  Both
    divisors are functions of the conditioning index -- ``qr``'s index *is* the mechanism
    ``h3`` divides by, and ``gr1`` is a regression on ``gr2``'s index -- so
    :math:`w/d^2 = w\varphi(U)` and ``E_0[w \varphi(U) (T - m) | U] = \varphi(U) E_0[w (T - m)
    | U] = 0``.  A weight built from anything else on the row, the law's :math:`g_0(a|W)` for
    instance, would break that silently: every array would stay in range and the ranking would
    be against a different projection from the one a reduced regression is.
    """
    weights = np.asarray(mass, dtype=float).reshape(-1)
    out: dict[str, np.ndarray] = {}
    for metric in METRICS:
        if metric.divisor is None:
            out[metric.name] = weights
            continue
        divisor = np.asarray(denominators[metric.divisor].values, dtype=float).reshape(-1)
        out[metric.name] = weights / divisor**2
    return out


def _check_the_weights_are_the_laws(
    mass: np.ndarray,
    truth_g: np.ndarray,
    indicator: np.ndarray,
    window: Any,
    arm: float,
) -> None:
    r"""The reference block's weights must *be* :math:`g_0(a|W)`, and this checks it.

    Everything this module does with the arm rests on it.  ``gr1``'s target is the bare
    indicator and its limit is :math:`E[g_0(a|W) \mid \cdot]` **only** because a weighted
    average over the two rows of a Sobol point turns one into the other; ``qr``'s ``| A = a``
    is a weight for the same reason.  So a weight vector that is one grid stale, or a window
    pointing at a :func:`~benchmarks.drtmle_remainder.stacked_companion` **draw** block --
    where every weight is ``1.0`` -- would integrate against the wrong measure and report the
    answer to five decimals.

    Checked rather than documented because both failures are silent: every array stays in
    range, no solver complains, and the estimate moves by an amount indistinguishable from
    the thing being measured.  This is the guard that makes "the reference was fitted on the
    reference block" structural.
    """
    rows = slice(window.start, window.stop)
    taken = indicator[rows] == 1.0
    if not np.allclose(mass[rows][taken], truth_g[rows][taken], rtol=1e-9, atol=1e-12):
        raise ValueError(
            f"the reference block's weights are not the law's own g_0({arm:g} | W): a "
            "quasi-random block carries the mechanism as its weight and an i.i.d. draw block "
            "carries ones, so this window is either the wrong block or a stale weight vector. "
            "Both integrate against the wrong measure silently"
        )


@dataclass(frozen=True)
class ArmTruth:
    r"""The law's own arrays at the companion rows, for one arm.

    Split out because **two callers need exactly these three and must not spell them
    twice**: the provider below, which fits a reference against them, and the held-out risk
    in ``benchmarks/drtmle_reference_study.py``, which scores candidates against them on a
    block the provider never touched.  A second spelling would be free to drift, and what it
    would drift into is a gate scoring a different regression from the one it gates.

    ``outcome`` is on the **scaled** scale, because that is what
    :attr:`~cleverly.fluctuation.iterative.InitialFit.arms` carries and ``qr``'s target is a
    difference of the two.
    """

    indicator: np.ndarray
    mechanism: np.ndarray
    outcome: np.ndarray


def arm_truth(current: Any, *, dgp: Any, arm: float) -> ArmTruth:
    r""":math:`1_a`, :math:`g_0(a|W)` and :math:`\bar Q_0(a, W)` at the companion rows."""
    companion = current.companion
    latent = _latent(companion.data, dgp)
    treatment = np.asarray(companion.data.treatment, dtype=float)
    truth_g_one = np.asarray(dgp.propensity(latent), dtype=float)
    return ArmTruth(
        indicator=(treatment == float(arm)).astype(float),
        mechanism=_arm_probability(truth_g_one, float(arm)),
        outcome=current.scaler.scale(
            np.asarray(dgp.outcome_mean(latent, float(arm), None), dtype=float)
        ),
    )


def fold_targets(
    current: Any,
    *,
    fold: int,
    arm: float,
    truth: ArmTruth,
    g_bounds: tuple[float, float],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    r"""``(designs, targets)`` at the companion rows, for one arm and one outer fold.

    The designs are :func:`~cleverly.estimators.reduced.reduced_designs`' -- ``qr`` on the
    mechanism and the two reduced mechanisms on the outcome regression, the crossing the R
    source names the other way round from the paper -- taken from **fold** :math:`k`'s copy,
    which is the model that held fold :math:`k` out.

    :math:`Y` never appears in a target, and that is the construction rather than a
    convenience: :func:`~benchmarks.drtmle_remainder.quadrature_frame` carries
    :math:`\bar Q_0(a, W)` in the companion's outcome column, so ``qr``'s population target is
    a difference of two known functions of :math:`W` and the other two are functions of the
    indicator alone.
    """
    companion = current.companion
    column = companion.propensity[fold].column_for(arm)
    bounded = companion.propensity[fold].bounded(g_bounds)[:, column]
    fitted_q = np.asarray(companion.outcome[fold].arms[arm], dtype=float)
    designs = reduced_designs(companion.propensity[fold], companion.outcome[fold], arm)
    targets = {
        "qr": truth.outcome - fitted_q,
        "gr1": truth.indicator,
        "gr2": (truth.indicator - bounded) / bounded,
    }
    return designs, targets


def per_reduction(reference: Reference | Mapping[str, Reference]) -> dict[str, Reference]:
    """One :class:`Reference` per reduced regression, from either spelling.

    A single object is broadcast to the three, which is what E2 shipped and is the case a
    mapping subsumes rather than replaces.  A mapping missing a name is refused rather than
    defaulted: a reference silently falling back to a rung nobody selected is the shape of
    mistake ``docs/roadmap.md``'s E2R exists to remove, and it would leave every array in
    range.
    """
    if not isinstance(reference, Mapping):
        return dict.fromkeys(("qr", "gr1", "gr2"), reference)
    missing = [name for name in ("qr", "gr1", "gr2") if name not in reference]
    if missing:
        raise ValueError(
            f"a per-regression reference must name every reduced regression; {missing} "
            "missing. A regression left to a default is fitted at a rung nothing selected"
        )
    return {name: reference[name] for name in ("qr", "gr1", "gr2")}


def reference_reductions(
    current: Any,
    *,
    dgp: Any,
    reference: Reference | Mapping[str, Reference],
    window: Any,
    row_weights: np.ndarray,
    g_bounds: tuple[float, float],
    reduction: str = "univariate",
) -> tuple[ReducedSet, tuple[ReducedSet, ...]]:
    """The three reduced regressions at their population limits, at ``current``'s pair.

    A function of the *current* targeted nuisances and not of the fit's starting point,
    because equations (9) and (10) are stated at starred reductions -- and a provider that
    closed over the initial pair would answer a different question while still passing a
    great deal, which is the mutation
    ``tests/unit/test_drtmle_reference.py`` watches.

    ``window`` is the reference block of the stacked companion, and the fit happens **there
    alone** while the prediction happens everywhere.  The two must not be the same block the
    remainder is integrated on: the reference's error propagates into the fit
    deterministically, so sharing a scramble with :math:`P_0\\hat D` would make the two the
    same random variable with a covariance nobody can sign.

    ``reference`` is one object or one per reduced regression -- see :func:`per_reduction`.
    E2R selects the rung per regression, and routing them through one argument here is what
    keeps the provider's own logic identical between a shipped rung and a selected one.
    """
    references = per_reduction(reference)
    companion = current.companion
    if companion is None:
        raise ValueError(
            "a population reference needs the companion rows the fit declared; fit with "
            "evaluation= so that every fold's nuisance is evaluated on the reference grid"
        )
    arms = tuple(current.arms)
    n_folds = companion.n_folds
    assignment = np.asarray(current.folds.assignment, dtype=int)

    treatment = np.asarray(companion.data.treatment, dtype=float)
    mass = np.asarray(row_weights, dtype=float).reshape(-1)
    if mass.size != treatment.size:
        raise ValueError(
            f"the companion holds {treatment.size} rows and the rule supplied {mass.size} "
            "weight(s); a stale weight vector integrates against the wrong measure and "
            "reports the answer to five decimals"
        )

    production: dict[str, list[np.ndarray]] = {"qr": [], "gr1": [], "gr2": []}
    elsewhere: dict[str, list[list[np.ndarray]]] = {"qr": [], "gr1": [], "gr2": []}

    for arm in arms:
        truth = arm_truth(current, dgp=dgp, arm=arm)
        _check_the_weights_are_the_laws(mass, truth.mechanism, truth.indicator, window, arm)
        at_production = {name: np.zeros(assignment.size) for name in production}
        at_folds: dict[str, list[np.ndarray]] = {name: [] for name in production}
        # The production design is the fit's own out-of-fold array and does not depend on
        # which fold's reference is about to read it -- row `i`'s value already came from the
        # model that held `i` out, which is exactly the pairing restored below.
        production_designs = reduced_designs(current.propensity, current.outcome, arm)
        # Rows of the reference block. Fitting happens here alone; prediction happens
        # everywhere, because the companion `ReducedSet` has to cover every companion row.
        block = np.zeros(treatment.size, dtype=bool)
        block[window.start : window.stop] = True

        for fold in range(n_folds):
            mine = assignment == fold
            designs, targets = fold_targets(
                current, fold=fold, arm=arm, truth=truth, g_bounds=g_bounds
            )
            for name in production:
                keep = fit_mask(name, truth.indicator)
                inside = block if keep is None else (block & keep)
                fitted = references[name].fit(
                    designs[name][inside], targets[name][inside], mass[inside]
                )
                at_folds[name].append(fitted(designs[name]))
                at_production[name][mine] = fitted(production_designs[name][mine])

        for name in production:
            production[name].append(at_production[name])
            elsewhere[name].append(at_folds[name])

    bounds = (float(g_bounds[0]), float(g_bounds[1]))
    at_companion = tuple(
        ReducedSet(
            qr=np.column_stack([column[fold] for column in elsewhere["qr"]]),
            gr1=np.column_stack([column[fold] for column in elsewhere["gr1"]]),
            gr2=np.column_stack([column[fold] for column in elsewhere["gr2"]]),
            arms=arms,
            g_bounds=bounds,
            reduction=reduction,
        )
        for fold in range(n_folds)
    )
    return (
        ReducedSet(
            qr=np.column_stack(production["qr"]),
            gr1=np.column_stack(production["gr1"]),
            gr2=np.column_stack(production["gr2"]),
            arms=arms,
            g_bounds=bounds,
            reduction=reduction,
        ),
        at_companion,
    )


class ReferenceReductionDRTMLE(DRTMLE):
    """``DRTMLE`` with the reduced regressions replaced by their population reference.

    **Both hooks, because both matter**, exactly as
    :class:`tests.unit.test_oracle_reductions.OracleReductionDRTMLE` does it: ``_nuisances``
    supplies the set the first round's mechanism covariate reads, and ``_reduction`` supplies
    the closure every later refit goes through.  Overriding only the second leaves one fitted
    set inside the fit, which is the half-substitution that makes a comparison mean nothing in
    particular.

    ``super()._nuisances`` still fits a ``glm`` reduction once before it is replaced.  That is
    accepted rather than optimised away: it keeps the override to a few lines, and it is one
    cheap fit against a kernel regression.

    **The attributes are underscored because two of the obvious names are already taken**, and
    both collisions are silent until they are not.  ``TMLE.reference`` is the **reference
    arm** -- assigning a projection there makes the fit raise ``reference=<SplineProjection>
    is not a level of A``, which is the loud one -- and ``TMLE._reference_arm`` is the method
    beside it.  ``tests/unit/test_drtmle_reference.py`` pins that a fitted instance's
    ``reference`` is still an arm.
    """

    def __init__(
        self,
        *,
        dgp: Any,
        reference: Reference | Mapping[str, Reference],
        window: Any,
        row_weights: np.ndarray,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._dgp = dgp
        self._reference = reference
        self._window = window
        self._row_weights = row_weights

    def _reference_set(self, current: Any, g_bounds: tuple[float, float]) -> Any:
        return reference_reductions(
            current,
            dgp=self._dgp,
            reference=self._reference,
            window=self._window,
            row_weights=self._row_weights,
            g_bounds=g_bounds,
        )

    def _nuisances(
        self,
        data: Any,
        folds: Any,
        scaler: Any,
        config: Any,
        intermediate_value: Any,
        seed: Any = None,
    ) -> Any:
        base, extra = super()._nuisances(data, folds, scaler, config, intermediate_value, seed)
        if base.reduced is None:
            return base, extra
        produced, at_companion = self._reference_set(base, base.reduced.g_bounds)
        base = replace(base, reduced=produced)
        if base.companion is not None:
            base = replace(base, companion=replace(base.companion, reduced=at_companion))
        return base, extra

    def _reduction(self, data: Any, nuisance: Any) -> Any:
        spec = super()._reduction(data, nuisance)
        if spec is None:
            return None
        bounds = nuisance.reduced.g_bounds
        return replace(spec, refit=lambda current: self._reference_set(current, bounds))


# ------------------------------------------------------- the one gate that is not a ladder


def held_out_risk(
    fitted: FittedReference,
    index: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
) -> float:
    r"""``E_0[w (T - m-hat(U))^2]`` on rows the reference was **not** fitted to.

    **Why a difference of these is a difference of squared errors**, which is the whole reason
    this is a gate and a knot ladder is not.  Write :math:`m` for the weighted conditional
    expectation -- which is what a reduced regression *is* -- and expand:

    .. math::

        E_0[w (T - \hat m)^2] = E_0[w (T - m)^2] + E_0[w (m - \hat m)^2]
                                + 2 E_0[(m - \hat m)\, E_0[w (T - m) \mid U]]

    The cross term **vanishes identically**, because :math:`E_0[w(T - m) \mid U] = 0` is the
    definition of :math:`m` under :math:`w`.  The first term does not depend on the candidate.
    So the difference of two candidates' held-out risks estimates the difference of their
    squared weighted :math:`L_2` errors, with nothing assumed about either.

    Three properties follow, and the third is the one that matters:

    * it is evaluated on rows **neither candidate saw**, from an independent scramble stream;
    * it estimates a difference of *squared errors* rather than of two estimates of the same
      target -- so a candidate that is **less** accurate ranks below one that is more accurate,
      whatever the common irreducible term is.  *An earlier revision of this bullet said "two
      equally wrong candidates do not look equal", and that is false under the identity displayed
      above*: two candidates with equal :math:`\|m - \hat m\|^2_w` have equal risk, however wrong
      both are.  The difference is **oriented**, which is a statement about ranking unequally
      accurate candidates and not about detecting shared inadequacy -- and shared inadequacy is
      exactly what this column cannot see, which is why a gate built on it is a *relative*
      instrument.  See ``docs/roadmap.md``'s piece F;
    * **it is oriented, and a refinement difference is not.**  A near-interpolating reference
      has *higher* held-out risk, so the column **ranks** candidates.  :math:`|fine - coarse|`
      is a magnitude with no orientation: it says two rungs disagree and cannot say which is
      nearer, and it is undefined for the first rung of any ladder.

      *Measured here rather than asserted*, on a logistic mean with `sigma = 0.35` noise and
      6,000 rows a block, scoring on an independent block::

          knots   held-out risk   |move vs prev|   true weighted L2
              8        0.120629                -            0.000120
             16        0.120815         0.013170            0.000360
             45        0.121401         0.014415            0.000773
             90        0.122985         0.025477            0.001950

      The risk orders the four correctly.  The movement does rise with them -- so the naive
      claim that it cannot see over-fitting at all is **wrong, and is not made here** -- but it
      **overstates the true error by an order of magnitude at every rung**, which is the same
      failure ``delta`` was withdrawn for, and it offers no reading at all for the rung a
      ladder starts on.

    The linearity :class:`SplineProjection` is pinned on is what makes the decomposition hold
    for it; a clip or a robust loss entering later would break the cross term silently, which
    is why that test exists.
    """
    residual = np.asarray(target, dtype=float).reshape(-1) - np.asarray(
        fitted(index), dtype=float
    ).reshape(-1)
    mass = np.asarray(weights, dtype=float).reshape(-1)
    total = float(mass.sum())
    if total <= 0.0:
        raise ValueError("a held-out risk needs positive total weight on the scoring block")
    return float(np.dot(residual**2, mass) / total)
