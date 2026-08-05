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
"""

from __future__ import annotations

from dataclasses import replace
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
    "POINTS_PER_PARAMETER",
    "EqualCountBins",
    "Reference",
    "ReferenceReductionDRTMLE",
    "SaturatedCells",
    "SplineProjection",
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


def _fit_mask(name: str, indicator: np.ndarray) -> np.ndarray | None:
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


def reference_reductions(
    current: Any,
    *,
    dgp: Any,
    reference: Reference,
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
    """
    companion = current.companion
    if companion is None:
        raise ValueError(
            "a population reference needs the companion rows the fit declared; fit with "
            "evaluation= so that every fold's nuisance is evaluated on the reference grid"
        )
    arms = tuple(current.arms)
    scaler = current.scaler
    n_folds = companion.n_folds
    assignment = np.asarray(current.folds.assignment, dtype=int)

    latent = _latent(companion.data, dgp)
    treatment = np.asarray(companion.data.treatment, dtype=float)
    mass = np.asarray(row_weights, dtype=float).reshape(-1)
    if mass.size != treatment.size:
        raise ValueError(
            f"the companion holds {treatment.size} rows and the rule supplied {mass.size} "
            "weight(s); a stale weight vector integrates against the wrong measure and "
            "reports the answer to five decimals"
        )
    truth_g_one = np.asarray(dgp.propensity(latent), dtype=float)

    production: dict[str, list[np.ndarray]] = {"qr": [], "gr1": [], "gr2": []}
    elsewhere: dict[str, list[list[np.ndarray]]] = {"qr": [], "gr1": [], "gr2": []}

    for arm in arms:
        indicator = (treatment == float(arm)).astype(float)
        truth_g = _arm_probability(truth_g_one, float(arm))
        truth_q = scaler.scale(np.asarray(dgp.outcome_mean(latent, float(arm), None), dtype=float))
        _check_the_weights_are_the_laws(mass, truth_g, indicator, window, arm)
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
            designs = reduced_designs(companion.propensity[fold], companion.outcome[fold], arm)
            column = companion.propensity[fold].column_for(arm)
            bounded = companion.propensity[fold].bounded(g_bounds)[:, column]
            fitted_q = np.asarray(companion.outcome[fold].arms[arm], dtype=float)
            # `Y` never appears: the frame carries `Qbar_0(a, W)` in its outcome column, so
            # the residual's population value is a difference of two known functions of `W`.
            targets = {
                "qr": truth_q - fitted_q,
                "gr1": indicator,
                "gr2": (indicator - bounded) / bounded,
            }
            for name in production:
                keep = _fit_mask(name, indicator)
                inside = block if keep is None else (block & keep)
                fitted = reference.fit(designs[name][inside], targets[name][inside], mass[inside])
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
        reference: Reference,
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
      target -- so two equally wrong candidates do not look equal;
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
