"""Cross-fitted estimation of the nuisance parameters.

TMLE needs three or four regressions before any targeting happens:

``g(a | W) = P(A = a | W)``
    the treatment mechanism, one column per arm, which enters the clever covariate;
``Qbar(A, W) = E[Y | A, W, Delta = 1]``
    the outcome regression, which is what gets fluctuated;
``P(Delta = 1 | A, W)``
    the missingness mechanism, when outcomes can be unobserved;
``P(Z = 1 | A, W)``
    the intermediate mechanism, for controlled direct effects.

All of them are fit **out of fold**: each observation's prediction comes from a
model that never saw it.  With machine-learning nuisance estimators this is not
optional -- in-sample predictions are overfit, the residuals they produce are too
small, and the targeting step cannot repair the resulting bias.  Cross-fitting is
also what lets the influence-curve variance stay valid without a Donsker
condition on the nuisance estimators.

The *raw*, un-truncated propensity is retained alongside the bounded one so that
sensitivity analyses can re-truncate and re-target without refitting anything.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np

from .._typing import BoolArray, FloatArray, IntArray, Learner
from ..data.causal_data import CausalData
from ..fluctuation.iterative import Fluctuation, InitialFit
from ..interventions import Incremental, IPSISet, RegimeSet, Shift, ShiftSet
from ..learners._fitting import (
    Task,
    as_target,
    fit_learner,
    predict_mean,
    predict_probabilities,
)
from ..learners.crossfit import Folds
from ..learners.density import ConditionalDensity, fit_conditional_density
from ..learners.screeners import CorrelationScreener
from ..learners.super_learner import SuperLearnerDiagnostics
from ..msm import MSMSet
from ..utils.bounds import OutcomeScaler, bound
from ..utils.parallel import map_parallel
from .direct_effect import check_level

if TYPE_CHECKING:  # `reduced` imports this module, so the dependency only goes one way
    from .reduced import MissingOutcomeReducedSet, ReducedSet

__all__ = [
    "CompanionEstimates",
    "InnerDesigns",
    "NuisanceEstimates",
    "Propensity",
    "RepeatFit",
    "UnfittedPropensity",
    "cross_fit_companion",
    "cross_fit_predictions",
    "fit_inner_designs",
    "fit_nuisances",
    "fit_on_rows",
]

#: A design a *companion* prediction is taken at: either one matrix every fold's model
#: predicts at, or one matrix per outer fold.  The second form is what a reduced regression
#: needs, since fold ``k``'s companion design is fold ``k``'s own primary prediction --
#: see :func:`~cleverly.estimators.reduced.fit_reduced`.
CompanionDesign: TypeAlias = "FloatArray | Sequence[FloatArray]"

#: How far a two-arm mechanism's rows may depart from summing to one before
#: :class:`Propensity` refuses to call it a distribution over the arms.  Loose enough for
#: the float error in a classifier's ``[1 - p, p]`` and in a weighted arm proportion, tight
#: enough that no real second mechanism slips through.
_SIMPLEX_TOLERANCE = 1e-9


@dataclass(frozen=True)
class Propensity:
    r"""The treatment mechanism :math:`g(a \mid W)`, one column per arm.

    ``values`` is ``(n, K)`` with column ``j`` holding :math:`P(A = \text{arms}[j] \mid W)`
    out of fold and **untruncated**; truncation happens at targeting time via
    :meth:`bounded`, because the ATT tolerates far less extrapolation than the ATE and
    so uses a tighter bound, and because a sensitivity sweep must be able to re-truncate
    without refitting.

    A matrix rather than the single :math:`g_1(W)` vector this used to be, even for two
    arms -- where column 0 is exactly ``1 - g1`` and the arithmetic is unchanged, *because*
    the two columns are a distribution over the arms.  With more than two arms there is no
    margin to be the propensity: the mechanism is a distribution over the arms, and every
    arm needs its own denominator.

    ``simplex`` says whether the rows are that distribution. The flag also supports
    deliberately armwise collaborative mechanisms: :meth:`bounded`'s two-arm shortcut is
    valid only on the simplex, while a non-simplex mechanism is clipped column by column.
    """

    values: FloatArray
    arms: tuple[float, ...]
    #: Whether the rows sum to one.  ``False`` marks a mechanism that is not a
    #: distribution over the arms, which is what :meth:`bounded` keys its two-arm
    #: shortcut off.
    simplex: bool = True

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(self.arms):
            raise ValueError(
                f"propensity must be (n, {len(self.arms)}) for arms {list(self.arms)}; "
                f"got shape {values.shape}"
            )
        # Checked rather than trusted, because `bounded` derives arm 0 from arm 1 whenever
        # this holds: a mechanism that is off the simplex and does not say so gets the
        # complement of the wrong column as its other denominator, which is a wrong
        # estimate rather than an error. Only two arms, since more are deliberately allowed
        # off the simplex, and only when every value is finite, which excludes
        # `UnfittedPropensity`'s NaN staging array.
        if len(self.arms) == 2 and self.simplex and bool(np.all(np.isfinite(values))):
            deviation = float(np.max(np.abs(values.sum(axis=1) - 1.0)))
            if deviation > _SIMPLEX_TOLERANCE:
                raise ValueError(
                    f"a two-arm propensity's rows must sum to one, and these depart by "
                    f"{deviation:.3g}. Pass simplex=False if this is deliberately not a "
                    "distribution over the arms -- a joint mechanism g_a(W) pi_a(W), say -- "
                    "so that bounded() clips both columns instead of taking one as the "
                    "complement of the other."
                )

    @property
    def n(self) -> int:
        return int(np.asarray(self.values).shape[0])

    @property
    def n_arms(self) -> int:
        return len(self.arms)

    def column_for(self, arm: float) -> int:
        """Index of the column holding ``P(A = arm | W)``."""
        match = [j for j, level in enumerate(self.arms) if level == float(arm)]
        if not match:
            raise KeyError(f"arm {float(arm)!r} is not one of {list(self.arms)}")
        return match[0]

    def arm(self, arm: float) -> FloatArray:
        """``P(A = arm | W)``, untruncated."""
        return np.asarray(self.values, dtype=float)[:, self.column_for(arm)]

    def bounded(self, bounds: tuple[float, float]) -> FloatArray:
        r"""The ``(n, K)`` mechanism truncated into ``bounds``.

        **Two arms on the simplex keep the complement form.**  ``g1`` is clipped and arm 0
        is taken as ``1 - g1``, which is exactly what the estimator has always done -- and
        it is not the same as clipping both columns when ``bounds`` is asymmetric, so this
        is what keeps every binary regression fixture valid.  It is valid only because the
        two columns sum to one; a ``simplex=False`` mechanism takes the column-by-column
        branch below, and :meth:`__post_init__` refuses the combination that would silently
        get this one.

        **More than two arms are clipped column by column, and are not renormalised.**
        Flooring a row of a multinomial breaks :math:`\sum_a g_a = 1`, and the obvious
        repair -- rescale the row -- would undo the only thing truncation is for, since
        rescaling can push a column back below the floor.  Nothing downstream needs the
        simplex: arm ``a``'s clever covariate reads only :math:`\tilde g_a`, and the
        plug-in is an average of targeted predictions containing no mechanism at all, so
        no bound can move :math:`\Psi`.  What a binding bound moves is the second-order
        remainder :math:`R_2`, exactly as :mod:`cleverly.fluctuation.submodel` sets out.
        :meth:`~cleverly.sensitivity.PositivityReport` reports how far the truncated rows
        depart from summing to one.
        """
        lower, upper = float(bounds[0]), float(bounds[1])
        values = np.asarray(self.values, dtype=float)
        if self.n_arms == 2 and self.simplex:
            one = bound(values[:, self.column_for(1.0)], lower, upper)
            columns = {self.column_for(1.0): one, self.column_for(0.0): 1.0 - one}
            return np.column_stack([columns[j] for j in range(2)])
        return bound(values, lower, upper)


@dataclass(frozen=True)
class UnfittedPropensity(Propensity):
    """The ``fit_treatment=False`` staging value: a mechanism that was never estimated.

    :func:`fit_nuisances` has to return the ordinary container even when it skipped the
    treatment model, because the outcome and missingness fits travel in it.  What it must
    *not* return is an array a consumer can quietly use.  Zeros would be usable: they clip
    to :meth:`Propensity.bounded`'s floor and give a finite, plausible, wrong estimate.  So
    the values are ``NaN`` and both read accessors raise -- the collaborative caller
    replaces the whole object before targeting, and any path that does not is a defect
    rather than a slightly worse fit.
    """

    def _unfitted(self) -> ValueError:
        return ValueError(
            "this treatment mechanism was never fitted: it is the fit_treatment=False "
            "staging value that a collaborative estimator must replace with its own g "
            "before any targeting, diagnostic or sensitivity code reads it"
        )

    def arm(self, arm: float) -> FloatArray:
        raise self._unfitted()

    def bounded(self, bounds: tuple[float, float]) -> FloatArray:
        raise self._unfitted()


@dataclass(frozen=True)
class InnerDesigns:
    r"""One fold-free copy of the primary nuisances per outer fold: **leave two folds out**.

    Entry ``k`` holds, at row ``j`` in fold ``m``, the prediction of a model fitted on
    ``assignment ∉ {k, m}``.  So it never saw outer fold ``k``, and it never saw row ``j``
    either -- both properties at once, which is what the nested reduced regression needs and
    what neither an independent split nor per-outer-fold designs give.  The ``m = k`` slice
    is the production model and is never read: the nested construction takes its
    **evaluation** designs from the production arrays exactly as the pooled one does, and
    differs only in what fold ``k``'s reduced regression **trains** on.

    **The inner split is the outer split**, used a second time, which is not an economy.  A
    freshly drawn one would add randomness the fit's ``random_state`` does not determine and
    would have to re-establish the cluster integrity :func:`~cleverly.learners.crossfit.
    make_folds` already checked; reusing the assignment inherits both.  It also fixes the
    inner training size at ``(K-2)/K`` of the sample against production's ``(K-1)/K`` --
    the smallest mismatch reachable without more than ``K(K-1)`` fits, and the one respect
    in which this is not the production estimator at a different split.

    **That mismatch is not the covariate shift** :func:`~cleverly.estimators.reduced.
    fit_reduced` rejects per-outer-fold designs for.  There the training designs would be a
    model's **in-sample** predictions and the test design its out-of-sample one, which is a
    first-order difference between two things that do not converge to each other.  Here both
    are out of sample and the gap is between two cross-fitted models of the same nuisance,
    which vanishes with the stabilisation the expansion already assumes.

    Attributes
    ----------
    outcome, propensity:
        One per outer fold, in fold order, each full length.  Both are the ordinary types
        rather than bare arrays, so a fold's copy is shape-checked on construction and its
        arm keys cannot silently disagree with the fit's.

    Carried on :attr:`NuisanceEstimates.inner` and used only by
    :class:`~cleverly.DRTMLE` with ``reduced_crossfit="nested"``, which is a reference
    construction used to diagnose generated-regressor leakage rather than a production path.
    """

    outcome: tuple[InitialFit, ...]
    propensity: tuple[Propensity, ...]

    def __post_init__(self) -> None:
        if len(self.outcome) != len(self.propensity):
            raise ValueError(
                "the inner outcome and mechanism designs must cover the same outer folds; "
                f"got {len(self.outcome)} and {len(self.propensity)}"
            )
        if not self.outcome:
            raise ValueError("inner designs must cover at least one outer fold")

    @property
    def n_folds(self) -> int:
        return len(self.outcome)


@dataclass(frozen=True)
class CompanionEstimates:
    r"""The fit's nuisances evaluated at rows it never saw, one copy per outer fold.

    What this is for is a benchmark-only fitted nuisance object exposing a prediction at new
    rows, *per fold*, so that :math:`P_0\hat D` -- the
    population mean of the **fitted** doubly-robust curve, which
    the remainder diagnostic needs and which :math:`P_n\hat D` is refused as a
    substitute for -- can be integrated against an independent draw.  This is that object,
    and it holds arrays rather than models for the reason
    :class:`~cleverly.estimators.reduced.ReducedSet` does: a model cannot be serialised
    with a result, and a *replay* of the alternation outside the library would be a second
    implementation of a state map whose whole difficulty is that it is not
    ``expit(logit Q + eps*H)``.

    Every array here is therefore produced by the **same fitted model object** the
    production array came from (:func:`cross_fit_companion`) and moved by the **same step
    sequence** the production arrays took
    (:attr:`~cleverly.fluctuation.iterative.Fluctuation.carried`).  What makes that
    checkable rather than asserted is that handing the fitting frame back in as the
    companion must reproduce the fit's own out-of-fold arrays exactly, fold by fold --
    ``tests/unit/test_drtmle_companion.py``.

    Attributes
    ----------
    data:
        The companion rows themselves, as a :class:`~cleverly.data.CausalData`, so that
        every clever-covariate builder can be reused verbatim at them rather than
        reimplemented.  They contribute to no fit, no score and no fold.
    outcome, propensity:
        One per outer fold, in fold order, each ``m`` rows long: entry ``k`` is fold ``k``'s
        model evaluated at every companion row.
    fold_sizes:
        Rows of the **fitting** sample held out by each fold.  §5's averaging convention is
        the fold-weighted mean, :math:`\sum_k (n_k/n) P_0 \hat D^{(k)}`, and this is what
        weights it; a study that averaged the slabs equally would be reporting a different
        convention without saying so.
    reduced:
        The three reduced regressions at the companion rows, one set per outer fold, filled
        in by :func:`~cleverly.estimators.reduced.fit_reduced`.  Empty until then.
    """

    data: CausalData
    outcome: tuple[InitialFit, ...]
    propensity: tuple[Propensity, ...]
    fold_sizes: tuple[int, ...]
    reduced: tuple[ReducedSet, ...] = ()

    def __post_init__(self) -> None:
        if len(self.outcome) != len(self.propensity) or len(self.outcome) != len(self.fold_sizes):
            raise ValueError(
                "the companion outcome, mechanism and fold sizes must cover the same outer "
                f"folds; got {len(self.outcome)}, {len(self.propensity)} and "
                f"{len(self.fold_sizes)}"
            )
        if not self.outcome:
            raise ValueError("companion estimates must cover at least one outer fold")

    @property
    def n_folds(self) -> int:
        return len(self.outcome)

    @property
    def n(self) -> int:
        """Companion rows."""
        return int(self.outcome[0].n)

    @property
    def fold_weights(self) -> FloatArray:
        """``n_k / n``: fold ``k``'s share of the rows *it held out*, which is the weighting
        a fold-conditional average over the slabs is taken with.  Equal weights are the same
        thing only on a balanced split.
        """
        sizes = np.asarray(self.fold_sizes, dtype=float)
        return sizes / float(sizes.sum())


@dataclass(frozen=True)
class NuisanceEstimates:
    """Cross-fitted nuisance predictions and the diagnostics that came with them.

    Attributes
    ----------
    propensity:
        Out-of-fold ``g(a | W)`` for every arm, *not* truncated -- see
        :class:`Propensity`.  On a **continuous** treatment there are no arms and no
        propensity: the mechanism is ``density`` instead, and this holds an ``(n, 0)``
        placeholder so that ``n`` and ``arms`` keep answering.  Every reader that would
        misread that placeholder as a real mechanism refuses a continuous fit by name --
        :func:`~cleverly.sensitivity.positivity_report` is the one that would otherwise
        report a spurious simplex deviation.
    outcome:
        Initial outcome regression on the ``[0, 1]`` scale, at the observed
        treatment and at every counterfactual arm.
    missingness, intermediate:
        ``(n, K)`` arrays indexed by treatment arm, or ``None`` when not applicable.
        On a ``shifts=`` fit a dose has no arms to index by, so they are ``(n, S + 1)``
        instead: column ``0`` at the observed dose and column ``s + 1`` at
        :math:`d_s(A, W)`, which is
        :attr:`~cleverly.interventions.ShiftSet.design`'s first axis exactly.  Both stay
        **untruncated** here and are bounded at targeting time by
        :meth:`bounded_missingness` and :meth:`intermediate_density`, which is what keeps
        ``nuisance_bound=`` a choice ``retarget`` can revisit without refitting.
    scaler:
        The transformation used to put the outcome on ``[0, 1]``.
    diagnostics:
        Per-nuisance Super Learner weights and cross-validated risks, when a
        Super Learner was used.
    """

    propensity: Propensity
    outcome: InitialFit
    scaler: OutcomeScaler
    folds: Folds
    #: Optional outcome state from which targeting must continue.  Ordinarily the
    #: targeting step starts at ``outcome``.  A collaborative estimator is different:
    #: model selection returns a *targeted* candidate Qbar and the reported estimator
    #: must continue from that selected state, while ``outcome`` remains the genuine
    #: initial learner fit used by nuisance diagnostics.
    targeting_outcome: InitialFit | None = None
    missingness: FloatArray | None = None
    intermediate: FloatArray | None = None
    treatment_covariates: tuple[str, ...] = ()
    diagnostics: dict[str, Any] = field(default_factory=dict)
    outcome_task: Task = "regression"
    #: Outcome regression evaluated at every requested level of the intermediate
    #: variable, when more than one was asked for.  ``outcome`` is the entry for the
    #: level this object currently targets; :meth:`at_level` swaps it.
    #:
    #: Populated because the controlled direct effect at ``z = 0`` and at ``z = 1``
    #: share *every* nuisance model -- the only difference is which counterfactual
    #: design the outcome regression is predicted onto -- so fitting them separately
    #: refits all four models to get two extra prediction vectors.
    outcome_by_level: dict[float, InitialFit] = field(default_factory=dict)
    #: The regimes this fit targets, evaluated at every row, or ``None`` for an
    #: arm-indexed fit.  Carried with the nuisances rather than recomputed because
    #: everything that reuses a fit reuses them: ``retarget``, and so the truncation
    #: curve, the MNAR tilt and the omitted-variable bound, all take a
    #: :class:`NuisanceEstimates` and must target the same regimes the fit declared --
    #: including on a result read back from disk, where the rules that built the
    #: densities are no longer callable.
    regimes: RegimeSet | None = None
    #: The conditional treatment density ``g(a | W)`` on a continuous fit, evaluated at
    #: every row and every bin, or ``None`` when the treatment has arms.  This is the
    #: mechanism half of double robustness there, standing where ``propensity`` stands
    #: for a discrete treatment.
    density: ConditionalDensity | None = None
    #: The shifts this fit targets, evaluated against ``density``, or ``None`` for a fit
    #: that declared none.  Carried for the reason ``regimes`` is, and built *inside*
    #: :func:`fit_nuisances` rather than beside it -- unlike a regime, a shift's clever
    #: covariate is a ratio of densities, so it cannot be evaluated until the density
    #: exists, and evaluating it here is what makes "g(A | W) and g(A - delta | W) come
    #: from the same out-of-fold model" structural rather than an invariant to maintain.
    shifts: ShiftSet | None = None
    #: The incremental interventions this fit targets, evaluated against ``propensity``,
    #: or ``None`` for a fit that declared none.  Built *inside* :func:`fit_nuisances`
    #: for the reason ``shifts`` is, and a sharper one: the tilt is a functional of the
    #: mechanism, so evaluating it where the mechanism is made is what keeps "the tilt
    #: and the g it tilts came from one out-of-fold model" structural rather than an
    #: invariant to maintain.  A regime, by contrast, is attached *outside* -- and could
    #: not be built here, since it needs no mechanism at all.
    #:
    #: After targeting this is replaced by :meth:`retilted` with the *fluctuated*
    #: mechanism, so that the influence curve's ``(A - g)`` and the plug-in agree.  The
    #: copy on ``result.nuisance`` is the initial one, exactly as ``outcome`` is.
    incremental: IPSISet | None = None
    #: The working model this fit projects the counterfactual means onto, evaluated at
    #: every row and every arm, or ``None`` for a fit that declared none.  Carried for the
    #: reason ``regimes`` is, and built *beside* :func:`fit_nuisances` rather than inside
    #: it -- unlike a shift, a working model's design is a function of the covariates
    #: alone and needs no mechanism to evaluate.
    msm: MSMSet | None = None
    #: The reduced-dimension regressions of the doubly-robust-inference variant, or
    #: ``None`` for every fit that is not one.  Carried for the reason ``regimes`` is,
    #: and built *outside* :func:`fit_nuisances` -- unlike ``shifts`` and ``incremental``,
    #: which are here because every fit that declares them needs them, this belongs to
    #: one variant and is fitted in its ``_nuisances`` override.  The invariant those two
    #: are built inside for survives anyway:
    #: :func:`~cleverly.estimators.reduced.fit_reduced` takes a whole
    #: :class:`NuisanceEstimates` and reads ``folds`` off it, so it cannot be handed a
    #: mechanism and a split that did not come from one construction.
    reduced: ReducedSet | MissingOutcomeReducedSet | None = None
    #: Fold-free copies of the primary nuisances, or ``None`` for every fit that did not
    #: ask for them -- which is every fit but a :class:`~cleverly.DRTMLE` with
    #: ``reduced_crossfit="nested"``.  Carried here rather than in that estimator for the
    #: reason ``reduced`` is: :func:`~cleverly.estimators.reduced.fit_reduced` takes a whole
    #: :class:`NuisanceEstimates` and reads ``folds`` off it, so the designs and the split
    #: they were built against cannot come apart.  See :class:`InnerDesigns`.
    inner: InnerDesigns | None = None
    #: The same nuisances evaluated at an independent draw, one copy per outer fold, or
    #: ``None`` for every fit that declared no ``evaluation=``.  Carried here for the reason
    #: ``inner`` is -- :func:`~cleverly.estimators.reduced.fit_reduced` reads it off the
    #: object it is handed, so the companion designs and the split cannot come apart -- and
    #: it is what :math:`P_0\\hat D` is integrated from.  See :class:`CompanionEstimates`.
    companion: CompanionEstimates | None = None

    def retilted(self, mechanism: FloatArray) -> NuisanceEstimates:
        """The same nuisances with every tilt re-evaluated at ``mechanism``.

        Named for the invariant rather than for its one caller: nothing derived from
        ``g`` may be left stale when ``g`` moves.  The mechanism fluctuation calls this
        after each update, and the object it returns is what the influence curve and the
        next clever covariate are built from.
        """
        if self.incremental is None:
            raise ValueError("this fit declared no incremental interventions to re-tilt")
        return replace(self, incremental=self.incremental.at(mechanism))

    @property
    def n(self) -> int:
        return self.propensity.n

    @property
    def arms(self) -> tuple[float, ...]:
        """The arm codes every per-arm array here is keyed by."""
        if not self.propensity.arms and self.msm is not None and self.msm.continuous:
            return self.msm.arms
        return self.propensity.arms

    def at_level(self, value: float) -> NuisanceEstimates:
        """The same nuisances, with the outcome regression evaluated at ``Z = value``."""
        fit = self.outcome_by_level.get(check_level(value))
        if fit is None:
            raise KeyError(
                f"no outcome regression was computed at intermediate level {value!r}; "
                f"have {sorted(self.outcome_by_level)}"
            )
        return replace(self, outcome=fit)

    def bounded_propensity(self, bounds: tuple[float, float]) -> FloatArray:
        """``g(a | W)`` truncated into ``bounds``, ``(n, K)`` -- see :meth:`Propensity.bounded`."""
        return self.propensity.bounded(bounds)

    def bounded_missingness(self, lower: float) -> FloatArray | None:
        """``P(Delta = 1 | A, W)`` truncated away from zero."""
        if self.missingness is None:
            return None
        return bound(self.missingness, lower, 1.0)

    def intermediate_density(self, value: float, lower: float) -> FloatArray | None:
        """``P(Z = z | A = a, W)`` for the targeted ``z``, truncated away from zero.

        Validates ``value`` because the branch below is silent about a level it does not
        recognise: anything other than ``1`` would fall through to the complement and
        return a perfectly plausible ``P(Z = 0 | A, W)`` for a parameter nobody asked
        for.  This accessor is the one place the ``z`` / ``1 - z`` convention lives, and
        every diagnostic that reports a denominator has to come through it.
        """
        if self.intermediate is None:
            return None
        level = check_level(value)
        probs = self.intermediate if level == 1.0 else 1.0 - self.intermediate
        return bound(probs, lower, 1.0)


@dataclass(frozen=True)
class RepeatFit:
    """One draw of the cross-fitting split, with everything that draw produced.

    A fit with ``repeats=R`` runs the whole construction ``R`` times over independent
    fold draws and averages the estimates (see
    :func:`~cleverly.inference.median_estimates`). The four things a draw produces --
    its folds, its nuisance predictions, the ``epsilon`` its targeting step solved, and
    the targeted ``Qbar`` that came out -- are *not* interchangeable between draws, so
    they are held together here rather than in parallel tuples on the result.  The pairing
    is the point: an analysis that took one draw's targeted ``Qbar`` and another draw's
    mechanism would be describing a fit that never happened, and the two arrays would give
    no sign of it.

    ``fluctuations`` is keyed by target group exactly as
    :attr:`~cleverly.estimators.base.TMLEResult.fluctuations` is -- that attribute now
    reads through to the first repeat.

    Attributes
    ----------
    nuisance:
        The out-of-fold nuisance predictions from this draw.  ``nuisance.folds`` is the
        draw itself, so a repeat carries the split that made it.
    fluctuations:
        The solved fluctuation per target group, holding this draw's ``epsilon`` and its
        targeted outcome regression.
    psi:
        This draw's own point estimates, per estimand -- the numbers the reported estimate
        is the mean of.  The point estimates alone and not the whole
        :class:`~cleverly.ParameterEstimate`: what they exist for is
        :meth:`~cleverly.estimators.base.TMLEResult.repeat_spread`, and keeping ``R``
        further copies of every influence curve to compute a standard deviation of ``R``
        scalars would multiply the memory a fit holds for nothing.
    """

    nuisance: NuisanceEstimates
    fluctuations: dict[str, Fluctuation]
    psi: Mapping[str, float] = field(default_factory=dict)

    @property
    def folds(self) -> Folds:
        """The split this draw realised."""
        return self.nuisance.folds


def cross_fit_predictions(
    learner: Learner,
    design: FloatArray,
    target: FloatArray,
    weights: FloatArray,
    folds: Folds,
    *,
    task: Task,
    predict_designs: dict[str, FloatArray],
    fit_mask: BoolArray | None = None,
    groups: IntArray | None = None,
    clip: tuple[float, float] | None = None,
    classes: Sequence[float] | None = None,
    n_jobs: int = 1,
) -> tuple[dict[str, FloatArray], list[SuperLearnerDiagnostics]]:
    """Out-of-fold predictions of one nuisance regression.

    Parameters
    ----------
    design, target, weights:
        Training data for the regression.
    predict_designs:
        Named design matrices to predict on -- for the outcome regression these are
        the observed treatment and every counterfactual arm, so a single pass over the
        folds produces everything the fluctuation needs.
    fit_mask:
        Rows eligible for *training*.  The outcome regression is fit only where the
        outcome is observed, but must still predict everywhere.
    groups:
        Cluster codes, forwarded to any learner that cross-validates internally so its
        inner folds keep clusters intact too -- see :func:`fit_on_rows`.
    classes:
        Set for a nuisance that is a conditional *distribution* over these classes
        rather than a single conditional mean -- the treatment mechanism of a ``K``-armed
        treatment.  Each named prediction then comes back ``(n, K)`` instead of ``(n,)``,
        with columns in ``classes`` order.

    Returns
    -------
    Predictions per named design, and the Super Learner diagnostics per fold (empty
    when the learner is not a Super Learner).
    """
    predictions, _, diagnostics = cross_fit_companion(
        learner,
        design,
        target,
        weights,
        folds,
        task=task,
        predict_designs=predict_designs,
        companion_designs={},
        fit_mask=fit_mask,
        groups=groups,
        clip=clip,
        classes=classes,
        n_jobs=n_jobs,
    )
    return predictions, diagnostics


def cross_fit_companion(
    learner: Learner,
    design: FloatArray,
    target: FloatArray,
    weights: FloatArray,
    folds: Folds,
    *,
    task: Task,
    predict_designs: dict[str, FloatArray],
    companion_designs: Mapping[str, CompanionDesign],
    fit_mask: BoolArray | None = None,
    groups: IntArray | None = None,
    clip: tuple[float, float] | None = None,
    classes: Sequence[float] | None = None,
    n_jobs: int = 1,
) -> tuple[dict[str, FloatArray], dict[str, FloatArray], list[SuperLearnerDiagnostics]]:
    """:func:`cross_fit_predictions`, and each fold's model evaluated at further rows.

    The one fold loop, so that a companion prediction comes from the **same fitted model
    object** the production prediction came from rather than from a refit of the same
    learner on the same rows.  Two of those would agree for a deterministic learner and
    would not for a seeded one, which is exactly the kind of "usually equal" this variant
    has been caught by before.

    ``predict_designs`` is scattered by fold into ``(n,)`` arrays, as it always was:
    row ``i``'s value comes from the model that held fold ``i`` out.  ``companion_designs``
    cannot be, because a companion row belongs to no fold -- so it comes back stacked,
    ``(K, m)`` or ``(K, m, C)``, entry ``k`` being fold ``k``'s model evaluated at every
    companion row.  Which of those slabs to read, or how to average them, is the caller's
    convention and is stated where the caller states it.  The convention this package's own
    readers use is the **fold-weighted average**: slab ``k`` is weighted by the number of rows
    fold ``k`` held out, so the weights are the held-out counts and sum to ``n``.  Equal
    weights are the same thing only when the split is balanced, which it is not in general.

    A companion design may be one matrix, used for every fold, or one matrix **per fold** --
    see :data:`CompanionDesign`.

    Returns ``(predictions, companion, diagnostics)``.  With no companion designs the first
    and last are what :func:`cross_fit_predictions` returns and the middle is empty; that
    path is the production one and is bit for bit what it was.
    """
    n = design.shape[0]
    mask = np.ones(n, dtype=bool) if fit_mask is None else np.asarray(fit_mask, dtype=bool)
    if not mask.any():
        raise ValueError("no rows are eligible for fitting this nuisance model")

    def predict(model: Learner, matrix: FloatArray) -> FloatArray:
        if classes is None:
            return _clip(predict_mean(model, matrix, task), clip)
        return _clip(predict_probabilities(model, matrix, classes), clip)

    def companion_at(model: Learner, fold: int) -> dict[str, FloatArray]:
        return {
            name: predict(model, _companion_matrix(matrix, fold))
            for name, matrix in companion_designs.items()
        }

    if folds.is_single:
        rows = np.flatnonzero(mask)
        model = fit_on_rows(learner, design, target, weights, rows, task, groups)
        predictions = {name: predict(model, matrix) for name, matrix in predict_designs.items()}
        companion = {name: values[None, ...] for name, values in companion_at(model, 0).items()}
        diagnostics = getattr(model, "diagnostics_", None)
        return predictions, companion, [diagnostics] if diagnostics is not None else []

    jobs = [(fold, train, test) for fold, (train, test) in enumerate(folds)]

    def run_fold(
        fold: int, train: IntArray, test: IntArray
    ) -> tuple[int, IntArray, dict[str, FloatArray], dict[str, FloatArray], Any]:
        rows = train[mask[train]]
        if rows.size == 0:
            raise ValueError(
                "a cross-fitting fold has no trainable rows for a nuisance model; "
                "reduce n_folds, supply cluster-aware folds, or -- when the outcome is "
                "the rare thing rather than the arm -- pass "
                "stratify_folds='treatment+outcome'"
            )
        model = fit_on_rows(learner, design, target, weights, rows, task, groups)
        predictions = {
            name: predict(model, matrix[test]) for name, matrix in predict_designs.items()
        }
        return (
            fold,
            test,
            predictions,
            companion_at(model, fold),
            getattr(model, "diagnostics_", None),
        )

    results = map_parallel(run_fold, jobs, n_jobs=n_jobs)
    shape: tuple[int, ...] = (n,) if classes is None else (n, len(tuple(classes)))
    out = {name: np.empty(shape, dtype=float) for name in predict_designs}
    slabs: dict[str, list[FloatArray]] = {name: [] for name in companion_designs}
    order: list[int] = []
    diagnostics_list: list[SuperLearnerDiagnostics] = []
    for fold, test, predictions, companion_values, diagnostics in results:
        for name, values in predictions.items():
            out[name][test] = values
        order.append(fold)
        for name, values in companion_values.items():
            slabs[name].append(values)
        if diagnostics is not None:
            diagnostics_list.append(diagnostics)
    # Stacked in **fold** order rather than in completion order: `map_parallel` preserves
    # the job order today, and a slab indexed by position would silently answer for the
    # wrong fold if it ever stopped doing so.
    rank = np.argsort(np.asarray(order))
    companion = {
        name: np.stack([values[index] for index in rank]) for name, values in slabs.items()
    }
    return out, companion, diagnostics_list


def _companion_matrix(design: CompanionDesign, fold: int) -> FloatArray:
    """One matrix from a companion design: the design itself, or this fold's entry."""
    if isinstance(design, np.ndarray):
        return design
    return np.asarray(design[fold], dtype=float)


def fit_on_rows(
    learner: Learner,
    design: FloatArray,
    target: FloatArray,
    weights: FloatArray,
    rows: IntArray,
    task: Task,
    groups: IntArray | None,
) -> Learner:
    """Fit a learner on ``rows``, passing cluster codes on to its inner folds.

    ``rows`` is a subset of one outer training fold, so a learner that cross-validates
    internally -- a :class:`~cleverly.learners.SuperLearner` scoring its candidates --
    only ever splits rows this fold was already allowed to train on.  What it needs told
    is the *cluster* structure, since an inner fold that splits a cluster scores a
    candidate on rows correlated with its own training set.

    :func:`~cleverly.learners._fitting.fit_learner` does the routing, which matters
    because ``screen_treatment=True`` wraps the learner in a pipeline: this used to test
    ``isinstance(learner, SuperLearner)`` and so dropped the cluster codes for exactly
    the configuration that asked for both.  The codes are subset to ``rows`` first, to
    stay aligned with the design handed alongside them.
    """
    return fit_learner(
        learner,
        design[rows],
        as_target(target[rows], task),
        weights[rows],
        groups=None if groups is None else np.asarray(groups)[rows],
        warn_unweighted=False,
    )


def _clip(values: FloatArray, clip: tuple[float, float] | None) -> FloatArray:
    if clip is None:
        return np.asarray(values, dtype=float)
    return np.clip(np.asarray(values, dtype=float), clip[0], clip[1])


def _screened(learner: Learner, threshold: float, min_retain: int | None) -> Learner:
    """Wrap a learner so covariates are screened inside each training fold.

    Screening must happen *inside* the fold: choosing covariates on the full sample
    and then cross-fitting leaks the held-out rows into the model-selection step.
    """
    from sklearn.pipeline import Pipeline

    return Pipeline(
        [
            ("screen", CorrelationScreener(threshold=threshold, min_retain=min_retain)),
            ("model", learner),
        ]
    )


def _arm_key(arm: float) -> str:
    """Name of the counterfactual design for one arm, where no intermediate is involved."""
    return f"arm@{arm}"


def _design_key(arm: float, level: float | None) -> str:
    """Name of the counterfactual design for one arm at one intermediate level.

    A single flat namespace, because ``cross_fit_predictions`` takes one dict of designs
    and returns one dict of predictions; the key has to carry both coordinates.
    """
    return f"arm@{arm}|z@{level}"


#: Key of the prediction at the treatment each unit actually received.  Only a shift fit
#: needs it: an arm fit reads the mechanism at its realised arm through the indicator
#: ``1{A = a}``, and a dose has no indicator to read it with.
_OBSERVED_KEY = "observed"


def _mechanism_designs(
    data: CausalData, arms: tuple[float, ...], shift_set: ShiftSet | None, design: FloatArray
) -> dict[str, FloatArray]:
    r"""Where a per-treatment mechanism -- :math:`\pi` or :math:`q_z` -- must be evaluated.

    Both mechanisms divide the clever covariate, and the fluctuation updates
    :math:`\bar Q` as a function of the *treatment*, so each has to be known wherever
    :math:`\bar Q` is read.  On the arm path that is each counterfactual arm.  On a shift
    path it is the observed dose **and** each shifted one: obtaining
    :math:`\bar Q^*(d_s(A, W), W)` means evaluating the covariate at :math:`d_s(A, W)`,
    and the covariate carries :math:`1 / \pi`.  Reusing the observed dose's value there is
    the mistake this exists to prevent, and it is silent whenever the mechanism happens not
    to depend on the dose.

    ``design`` is the block's own training design, passed back as a named prediction so
    that column ``0`` is an *out-of-fold* :math:`\pi(A_i, W_i)` rather than an in-sample
    one -- the same trick the outcome regression already uses for its ``"observed"`` key.
    """
    if shift_set is None:
        return {_arm_key(arm): data.counterfactual_design(arm) for arm in arms}
    designs = {_OBSERVED_KEY: design}
    for index, code in enumerate(shift_set.codes):
        designs[_arm_key(code)] = data.counterfactual_design(shift_set.shifted[:, index])
    return designs


def _mechanism_columns(
    predictions: dict[str, FloatArray], arms: tuple[float, ...], shift_set: ShiftSet | None
) -> FloatArray:
    """Stack :func:`_mechanism_designs`' predictions into the array the submodel reads.

    ``(n, K)`` keyed by arm on the arm path, and ``(n, S + 1)`` observed-first on a shift
    path -- the same layout as :attr:`~cleverly.interventions.ShiftSet.design`'s first
    axis, so a builder that indexes one indexes the other with the same integer.
    """
    if shift_set is None:
        return np.column_stack([predictions[_arm_key(arm)] for arm in arms])
    columns = [predictions[_OBSERVED_KEY]]
    columns.extend(predictions[_arm_key(code)] for code in shift_set.codes)
    return np.column_stack(columns)


def _requested_levels(
    intermediate_value: float | None, extra_levels: Sequence[float]
) -> tuple[float, ...]:
    """The levels to evaluate the outcome regression at, primary one first."""
    if intermediate_value is None:
        return (None,)  # type: ignore[return-value]
    primary = check_level(intermediate_value)
    ordered = [primary]
    for level in extra_levels:
        checked = check_level(level)
        if checked not in ordered:
            ordered.append(checked)
    return tuple(ordered)


def fit_nuisances(
    data: CausalData,
    *,
    outcome_learner: Learner,
    treatment_learner: Learner,
    missingness_learner: Learner | None,
    intermediate_learner: Learner | None,
    folds: Folds,
    scaler: OutcomeScaler,
    intermediate_value: float | None = None,
    extra_levels: Sequence[float] = (),
    screen_treatment: bool = False,
    screen_threshold: float = 0.1,
    min_retain: int | None = None,
    shifts: Sequence[Shift] = (),
    incremental: Sequence[Incremental] = (),
    incremental_reference: str | None = None,
    shift_reference: str | None = None,
    density_bins: int = 20,
    msm: MSMSet | None = None,
    companion: CausalData | None = None,
    n_jobs: int = 1,
    fit_treatment: bool = True,
) -> NuisanceEstimates:
    """Fit every nuisance model this estimator needs.

    ``intermediate_value`` selects which controlled direct effect the outcome
    regression is evaluated at; it is required when the data carries an
    intermediate variable.  ``extra_levels`` asks for the outcome regression to be
    evaluated at further levels in the *same* pass over the folds, which is what lets
    both controlled direct effects be estimated from one set of nuisance fits.

    ``shifts`` declares modified treatment policies, which a continuous treatment
    requires and an arm-coded one refuses.  They are evaluated *here*, against the
    density fitted a few lines above, rather than by the caller: the clever covariate is
    the ratio :math:`g(a - \\delta \\mid W) / g(a \\mid W)`, so numerator and denominator
    come from one out-of-fold model by construction and there is no second model for a
    later step to get wrong.

    ``fit_treatment=False`` is the outcome-first staging path used by collaborative
    estimators, which replace the ordinary treatment model before any targeting or
    reporting occurs.  It is intentionally internal: the returned propensity is an
    :class:`UnfittedPropensity`, whose values are ``NaN`` and whose accessors raise, so a
    caller that fails to replace it stops rather than reporting a plausible number.

    ``companion`` is an independent draw at which every fold's mechanism and outcome
    regression is *also* evaluated, returned on
    :attr:`NuisanceEstimates.companion`.  It contributes to no fit, no fold and no score,
    so a call that passes ``None`` -- which is every call but a
    :class:`~cleverly.DRTMLE` with ``evaluation=`` -- goes down the identical path.  It is
    accepted only for the shape that estimator supports: an arm-coded treatment, a fully
    observed outcome and no intermediate, which are exactly the refusals ``DRTMLE`` already
    makes by name.
    """
    diagnostics: dict[str, Any] = {}
    groups = data.cluster
    if companion is not None:
        _check_companion(data, companion)
    if not fit_treatment and (data.is_continuous_treatment or incremental or companion is not None):
        raise ValueError(
            "fit_treatment=False only supports discrete point-treatment nuisances without "
            "incremental interventions or companion evaluation"
        )

    # --- treatment mechanism -------------------------------------------------
    treatment_model = (
        _screened(treatment_learner, screen_threshold, min_retain)
        if screen_treatment
        else treatment_learner
    )
    arms = data.arm_codes
    density: ConditionalDensity | None = None
    shift_set: ShiftSet | None = None
    ipsi_set: IPSISet | None = None
    if data.is_continuous_treatment:
        # A dose has no arms, so there is no P(A = a | W) to classify: the mechanism is a
        # density. The learner is the same one either way -- fit_conditional_density
        # factorises the density into bin hazards, each a conditional probability of a
        # binary event, so the classifier in treatment_learner= estimates all of them at
        # once. The propensity below is an (n, 0) placeholder; see NuisanceEstimates.
        density, density_diagnostics = fit_conditional_density(
            treatment_model,
            data.covariates,
            data.treatment,
            data.weights,
            folds,
            n_bins=density_bins,
            groups=groups,
            n_jobs=n_jobs,
        )
        diagnostics["density"] = density_diagnostics
        propensity = Propensity(np.zeros((data.n, 0)), ())
        if shifts:
            shift_set = ShiftSet.evaluate(tuple(shifts), data, density, reference=shift_reference)
    elif fit_treatment:
        propensity_out, propensity_companion, propensity_diagnostics = cross_fit_companion(
            treatment_model,
            data.covariates,
            data.treatment,
            data.weights,
            folds,
            task="classification",
            predict_designs={"g": data.covariates},
            companion_designs={} if companion is None else {"g": companion.covariates},
            groups=groups,
            clip=(0.0, 1.0),
            classes=arms,
            n_jobs=n_jobs,
        )
        propensity = Propensity(propensity_out["g"], arms)
        if propensity_diagnostics:
            diagnostics["propensity"] = propensity_diagnostics
        if incremental:
            # Evaluated *here*, against the mechanism fitted three lines above, for the
            # reason the shifts are: q_delta is a functional of g, so building it beside
            # g makes "both came from one out-of-fold model" structural. Untruncated, and
            # deliberately: g is inside the estimand on this axis, so a bound would move
            # the parameter -- see IPSISet.evaluate.
            ipsi_set = IPSISet.evaluate(
                tuple(incremental), data, propensity.values, reference=incremental_reference
            )
    else:
        # A staging value, not an estimated mechanism.  Keeping the arm metadata and
        # matrix shape valid lets the shared outcome/missingness pipeline return its
        # ordinary container; CTMLE replaces this before any consumer can read it, and
        # `UnfittedPropensity` is what turns "does not" into "cannot".
        propensity = UnfittedPropensity(np.full((data.n, len(arms)), np.nan), arms)

    retained = data.covariate_names
    if screen_treatment:
        retained = _retained_covariates(data, screen_threshold, min_retain)

    # --- missingness mechanism ----------------------------------------------
    missingness = None
    if data.has_missing_outcome:
        if missingness_learner is None:
            raise ValueError(
                "the data has missing outcomes but no missingness_learner was supplied"
            )
        missingness_design = data.missingness_design()
        missing_out, missing_diagnostics = cross_fit_predictions(
            missingness_learner,
            missingness_design,
            data.observed.astype(float),
            data.weights,
            folds,
            task="classification",
            predict_designs=_mechanism_designs(data, arms, shift_set, missingness_design),
            groups=groups,
            clip=(0.0, 1.0),
            n_jobs=n_jobs,
        )
        missingness = _mechanism_columns(missing_out, arms, shift_set)
        if missing_diagnostics:
            diagnostics["missingness"] = missing_diagnostics

    # --- intermediate mechanism ---------------------------------------------
    intermediate = None
    if data.has_intermediate:
        if intermediate_learner is None:
            raise ValueError(
                "the data has an intermediate variable but no intermediate_learner was supplied"
            )
        assert data.intermediate is not None
        # ``[A, W]``, and deliberately not ``missingness_design()`` even though the two
        # are the same array today.  ``q(Z | A, W)`` conditions on the propensity's set
        # extended by the treatment, which is the time ordering; it does not inherit the
        # missingness model's assumptions, and a longitudinal extension that added ``Z``
        # to the missingness design would silently make this ``P(Z | A, W, Z)``.
        intermediate_design = data.treatment_design()
        intermediate_out, intermediate_diagnostics = cross_fit_predictions(
            intermediate_learner,
            intermediate_design,
            data.intermediate,
            data.weights,
            folds,
            task="classification",
            predict_designs=_mechanism_designs(data, arms, shift_set, intermediate_design),
            groups=groups,
            clip=(0.0, 1.0),
            n_jobs=n_jobs,
        )
        intermediate = _mechanism_columns(intermediate_out, arms, shift_set)
        if intermediate_diagnostics:
            diagnostics["intermediate"] = intermediate_diagnostics

    # --- outcome regression --------------------------------------------------
    if data.has_intermediate and intermediate_value is None:
        raise ValueError("intermediate_value is required when the data carries an intermediate")
    include_z = data.has_intermediate
    outcome_design = data.treatment_design(include_intermediate=include_z)
    scaled = scaler.scale(data.outcome)
    outcome_task: Task = "classification" if data.family == "binomial" else "regression"

    # Every requested level of the intermediate in one pass over the folds. The
    # outcome *model* does not depend on the level -- its design uses the observed Z --
    # so the levels differ only in which counterfactual design it is predicted onto,
    # which is exactly what predict_designs is for. Fitting them separately refits all
    # four nuisance models to obtain two extra prediction vectors.
    levels = _requested_levels(intermediate_value, extra_levels)

    # What the counterfactual predictions are keyed by, and what treatment value each
    # one sets. An arm fit sets one level for everybody; a shift fit sets a *different*
    # dose per row, which is what makes a modified treatment policy modified -- so the
    # value here is an (n,) array rather than a scalar, and the keys are shift codes.
    counterfactual: dict[float, float | FloatArray]
    if msm is not None and msm.continuous:
        counterfactual = {
            code: float(dose) for code, dose in zip(msm.arms, msm.dose_values, strict=True)
        }
    elif shift_set is None:
        counterfactual = {arm: arm for arm in arms}
    else:
        counterfactual = {
            code: shift_set.shifted[:, index] for index, code in enumerate(shift_set.codes)
        }

    designs: dict[str, FloatArray] = {"observed": outcome_design}
    for level in levels:
        for code, value in counterfactual.items():
            designs[_design_key(code, level)] = data.counterfactual_design(
                value, intermediate_value=level
            )

    companion_designs: dict[str, FloatArray] = {}
    if companion is not None:
        companion_designs = {"observed": companion.treatment_design()}
        for arm in arms:
            companion_designs[_design_key(arm, levels[0])] = companion.counterfactual_design(arm)

    outcome_out, outcome_companion, outcome_diagnostics = cross_fit_companion(
        outcome_learner,
        outcome_design,
        scaled,
        data.weights,
        folds,
        task=outcome_task,
        predict_designs=designs,
        companion_designs=companion_designs,
        fit_mask=data.observed,
        groups=groups,
        clip=(0.0, 1.0),
        n_jobs=n_jobs,
    )
    if outcome_diagnostics:
        diagnostics["outcome"] = outcome_diagnostics

    by_level = {
        level: InitialFit(
            outcome_out["observed"],
            {code: outcome_out[_design_key(code, level)] for code in counterfactual},
        )
        for level in levels
    }
    primary = by_level[levels[0]]

    companion_estimates = None
    if companion is not None:
        counts = np.bincount(np.asarray(folds.assignment), minlength=folds.n_folds)
        companion_estimates = CompanionEstimates(
            data=companion,
            outcome=tuple(
                InitialFit(
                    outcome_companion["observed"][fold],
                    {
                        arm: outcome_companion[_design_key(arm, levels[0])][fold]
                        for arm in counterfactual
                    },
                )
                for fold in range(folds.n_folds)
            ),
            propensity=tuple(
                Propensity(propensity_companion["g"][fold], arms) for fold in range(folds.n_folds)
            ),
            fold_sizes=tuple(int(count) for count in counts),
        )

    return NuisanceEstimates(
        propensity=propensity,
        outcome=primary,
        outcome_by_level=by_level if data.has_intermediate else {},
        scaler=scaler,
        folds=folds,
        missingness=missingness,
        intermediate=intermediate,
        treatment_covariates=tuple(retained),
        diagnostics=diagnostics,
        outcome_task=outcome_task,
        density=density,
        shifts=shift_set,
        incremental=ipsi_set,
        msm=msm,
        companion=companion_estimates,
    )


def _check_companion(data: CausalData, companion: CausalData) -> None:
    """Refuse a companion this function cannot evaluate every nuisance at.

    Named rather than assumed, because each of these is a *silent* wrong answer rather than
    a crash: a companion on a continuous treatment would carry an ``(n, 0)`` mechanism, one
    with a missingness or intermediate model would be missing the factors its clever
    covariate divides by, and one whose covariates are named differently would be predicted
    at a design the model was not fitted on.  All three are already refused by name in
    :class:`~cleverly.DRTMLE`, which is the only caller; this is the second lock, on the
    function that would otherwise return an array for a fit nobody ran.
    """
    if data.is_continuous_treatment or companion.is_continuous_treatment:
        raise NotImplementedError(
            "an evaluation companion reads a per-arm mechanism g(a | W), and a continuous "
            "dose has none: its mechanism is a conditional density and evaluating it at "
            "new rows is a different object from predicting a probability."
        )
    if data.has_missing_outcome or data.has_intermediate:
        raise NotImplementedError(
            "an evaluation companion is not built for a fit with delta= or intermediate=: "
            "the curve at the companion rows would need the missingness and intermediate "
            "mechanisms there too, and DRTMLE -- the only estimator that asks for one -- "
            "refuses both by name."
        )
    if tuple(companion.covariate_names) != tuple(data.covariate_names):
        raise ValueError(
            f"the companion carries covariates {list(companion.covariate_names)} and the "
            f"fit was made on {list(data.covariate_names)}; a companion is the fit's own "
            "models evaluated elsewhere, so its design has to be the design they were "
            "fitted on"
        )
    if tuple(companion.treatment_levels) != tuple(data.treatment_levels):
        raise ValueError(
            f"the companion's treatment takes levels {list(companion.treatment_levels)} and "
            f"the fit's takes {list(data.treatment_levels)}; the arm codes index every "
            "per-arm array here, so the two must agree"
        )


def fit_inner_designs(
    data: CausalData,
    nuisance: NuisanceEstimates,
    *,
    outcome_learner: Learner,
    treatment_learner: Learner,
    n_jobs: int = 1,
) -> InnerDesigns:
    r"""Refit the two primary nuisances once per outer fold, leaving that fold out as well.

    The whole construction is one keyword of :func:`cross_fit_predictions`: passing
    ``fit_mask`` with fold ``k``'s rows removed makes every fold's model train on
    ``train(m) ∩ complement(k)``, which is leave-two-folds-out and needs no second split, no
    further randomness and no new fold machinery.  Predictions still come back for every
    row, and the ``m = k`` slice -- the production model, since removing fold ``k`` from a
    training set that already excludes it changes nothing -- is the one entry
    :class:`InnerDesigns` says is never read.

    Only the treatment mechanism and the outcome regression are refitted, because those are
    the two arrays a reduced regression conditions on and takes residuals of.  There is no
    missingness or intermediate model here and no shift or incremental set: a
    :class:`~cleverly.DRTMLE` refuses ``intermediate=`` and the other parameter axes by
    name, while its supported ``delta=`` surface requires ``cross_fit=False`` and therefore
    cannot enter this nested construction.

    **Cost is ``K`` times the primary nuisance fitting**, paid once at the initial fit --
    ``K²`` models of each nuisance against ``K``.  Every refit inside the alternation reuses
    these arrays, moved by the fluctuation that moved the production ones
    (:attr:`~cleverly.fluctuation.iterative.Fluctuation.carried`), so the alternation costs
    what it costs on a pooled fit.
    """
    if data.is_continuous_treatment:
        raise ValueError("the nested construction reads a per-arm mechanism; a dose has none")
    folds = nuisance.folds
    if folds.n_folds < 3:
        raise ValueError(
            "the nested construction leaves two folds out at a time, so it needs at least "
            f"three; this fit has {folds.n_folds}. Pass n_folds=3 or more, or "
            "reduced_crossfit='pooled'."
        )
    arms = nuisance.arms
    outcome_design = data.treatment_design()
    designs: dict[str, FloatArray] = {"observed": outcome_design}
    for arm in arms:
        designs[f"arm{arm}"] = data.counterfactual_design(arm)
    scaled = nuisance.scaler.scale(data.outcome)
    assignment = np.asarray(folds.assignment)
    observed = np.asarray(data.observed, dtype=bool)

    outcomes: list[InitialFit] = []
    propensities: list[Propensity] = []
    for fold in range(folds.n_folds):
        without = assignment != fold
        mechanism, _ = cross_fit_predictions(
            treatment_learner,
            data.covariates,
            data.treatment,
            data.weights,
            folds,
            task="classification",
            predict_designs={"g": data.covariates},
            fit_mask=without,
            groups=data.cluster,
            clip=(0.0, 1.0),
            classes=arms,
            n_jobs=n_jobs,
        )
        regression, _ = cross_fit_predictions(
            outcome_learner,
            outcome_design,
            scaled,
            data.weights,
            folds,
            task=nuisance.outcome_task,
            predict_designs=designs,
            fit_mask=observed & without,
            groups=data.cluster,
            clip=(0.0, 1.0),
            n_jobs=n_jobs,
        )
        propensities.append(Propensity(mechanism["g"], arms))
        outcomes.append(
            InitialFit(regression["observed"], {arm: regression[f"arm{arm}"] for arm in arms})
        )
    return InnerDesigns(outcome=tuple(outcomes), propensity=tuple(propensities))


def _retained_covariates(
    data: CausalData, threshold: float, min_retain: int | None
) -> tuple[str, ...]:
    """Which covariates a full-sample screen would keep, for reporting only.

    The fits themselves screen inside each fold; this is the whole-sample answer,
    reported so the user can see what the screen is doing.
    """
    from ..learners.screeners import screen_by_correlation

    keep = screen_by_correlation(
        data.covariates,
        data.treatment,
        threshold=threshold,
        min_retain=min_retain,
        sample_weight=data.weights,
    )
    return tuple(name for name, flag in zip(data.covariate_names, keep, strict=True) if flag)
