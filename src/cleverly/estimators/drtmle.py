r"""Doubly-robust nonparametric inference: a TMLE whose *interval* survives one bad nuisance.

.. warning::

   **This variant is in progress.**  The code is written and its tests pass; that is not the
   same as finished, and two of the outstanding items are the kind that decide whether the
   thing is right at all.  The full list, with what each would change, is in
   ``docs/roadmap.md`` under *What is still open*.  The five a caller's numbers depend on:

   1. **The influence curve is transcribed rather than derived, and where it has now been
      checked against a statement of Theorem 1 it disagrees -- on a sign.**  The curve is
      the whole of what this variant buys and it is read off the R package ``drtmle``'s
      implementation.  The 2016 working-paper version of Benkeser et al. defines
      :math:`D_A = -(Q_r/g)(A - g)` and *subtracts* it, so its net mechanism-side
      contribution is :math:`+(Q_r/g)(A - g)`, where this package and ``drtmle`` alike
      compute a positive :math:`D^*_g` and subtract that.  Nothing that reports a point
      estimate can see it -- all three empirical means are driven to zero -- and what it
      moves is the variance.  Open pending the *published* 2017 article, which is
      authoritative and is not in hand; that is item 21, and
      ``docs/drtmle-theorem-concordance.md`` carries the objects and the acceptance test.
      See :func:`~cleverly.inference.influence.reduced_corrections`.
   2. **No number here has been compared against ``drtmle``'s output.**  The cheapest check
      on item 1, and it has not been done.
   3. **Nothing demonstrates that the interval is better.**  A coverage study over the
      off-diagonal of the misspecification grid found no gap for this variant to close at
      the sizes it could reach; the regime it is for is out of reach of a nightly budget.
   4. **The alternation does not reliably converge.**  Equation (10)'s covariate is
      near-singular on exactly the fits anybody wants -- see
      :func:`~cleverly.estimators.targeting.solve_with_reduction` -- so some draws exit at
      the outer cap and report ``failure = "max_iter_reached"``.  Over a 96-fit sweep 8 did
      that, 86 stalled at a fixed point and 2 reached the tolerance.  Which of those a
      given fit did is on its own report: ``summary()`` ends with the score check whenever
      the check fails, and ``res.score_verdict`` carries the verdict either way.  It used
      to say "read ``res.validation.score_check()`` on every fit rather than assuming",
      which was documentation standing in for reporting -- an unlicensed interval was
      formatted exactly like a licensed one and the reader had to know to go looking.
   5. **The reported curve is not centred wherever the mechanism truncation binds.**  On
      ``weak_overlap_dgp`` the score check fails on 23 of 24 swept fits, with the worst
      score at rough parity with ``se/sqrt(n)`` rather than the ``1e-7`` every other process
      reports -- and on roughly a quarter of *ordinary* splits it fails by ``2e-5`` to
      ``7e-4``.  The cause is one defect and it is located: equation (9) is solved against
      the raw tilted :math:`g^*` while the :math:`D^*_g` the curve subtracts reads the
      truncated one, so the two agree on every row the bound leaves alone and part company
      on every row it clips.  A single clipped row of 600 is enough.  It is not the
      conditioning of item 4 -- ``ill_conditioned`` never fires on that process -- and it is
      *not* a stale array: recomputing the recorded score from the returned state reproduces
      it bit for bit.  Until ``docs/roadmap.md``'s piece B1a lands, read a ``DRTMLE``
      standard error as provisional on every process and check ``res.score_verdict``.  Which
      convention replaces it is piece B1b's and waits on the theorem: there are more than
      two candidates, and the theorem's own algorithm truncates nothing at all.

Every interval this package reports is valid when the second-order remainder is negligible,
and for a plain TMLE that remainder is the product
:math:`\|\hat g - g_0\| \cdot \|\hat{\bar Q} - \bar Q_0\|`.  A product goes to zero when one
factor does, which is why the *point estimate* is doubly robust; but the interval needs
:math:`\sqrt n R_2 \to 0`, and with one factor not shrinking :math:`R_2` is first order in
the other.  So **`TMLE` is doubly robust for consistency and singly robust for inference**,
and this class closes that second gap.

It does it by solving two further score equations (van der Laan 2014; Benkeser, Carone,
van der Laan & Gilbert 2017; Benkeser & Hejazi 2023), built from *reduced-dimension*
regressions of each nuisance's residual on the other:

.. math::

    (9)  \quad & P_n[\, Q_r(a, W)/g^*(a|W)\,\{1_a - g^*(a|W)\}\,] = 0 \\
    (10) \quad & P_n[\, 1_a\,g_{r,2}(a|W)/g_{r,1}(a|W)\,\{Y - \bar Q^*(a, W)\}\,] = 0

The reductions are univariate however many covariates the fit adjusted for, so they can be
estimated fast enough whether or not the primary nuisances can.
:mod:`cleverly.estimators.reduced` fits them and
:func:`~cleverly.estimators.targeting.solve_with_reduction` solves the three equations
together.

**What this is not.**  The point estimate is a plain TMLE's, to the precision the extra
fluctuations move it -- the three empirical means are all driven to zero, so the extra terms
cannot move :math:`\hat\Psi` and only move its variance.  Read a ``DRTMLE`` fit as the same
estimate with an interval that is entitled to be believed under weaker conditions, not as a
better estimate.

**And it is not the efficient one.**  Under misspecification the canonical gradient at
:math:`P_0` is still :math:`D^*`.  What the three equations leave is
:math:`D = D^* - D^*_Q - D^*_g`, the *estimator's* asymptotic influence function at the
nuisance limits, and the estimator is generally **not efficient** there -- so the interval is
one that stays valid where a plain TMLE's stops being valid, and nothing more than that.
When both nuisances are consistent the corrections vanish row by row, the two curves
coincide, and this is the ordinary efficient estimator; that is exactly the case the variant
is not for.  The distinction is easy to lose because the numbers point the other way: in the
guide's worked example the corrected standard error is the **smaller** of the two, 0.06828
against 0.06850, which is a fact about one draw and not a general narrowing.  A doubly-robust
fit's ``score_check`` says so in its own verdict rather than signing the fit off as having
solved the efficient score equation.

**What it costs.**  Two further learner fits per arm per round, refitted *inside* the
alternation, plus a mechanism fluctuation.  A truncation curve or an MNAR sweep on a
``DRTMLE`` result therefore costs about a fit per point rather than a fraction of one:
``retarget`` here is no longer arithmetic on cached arrays, which is the one contract this
variant breaks and the reason it is a class of its own rather than a keyword.

Scope is what the sources *derive*, which is narrower than what ``drtmle`` accepts: a binary
treatment, the ``mean`` group, and Benkeser et al.'s univariate reduction.  Everything else
is refused by name.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from ..data.causal_data import CausalData
from ..learners.crossfit import Folds
from ..learners.super_learner import SuperLearnerDiagnostics
from ..utils.bounds import OutcomeScaler
from ._nuisance import NuisanceEstimates
from .base import MEAN_GROUP_ESTIMANDS, TMLEConfig, resolve_estimands
from .ctmle import CTMLE
from .reduced import REDUCTIONS, ReducedSet, fit_reduced, refuse_unsupported
from .targeting import ReductionSpec
from .tmle import TMLE

__all__ = ["DRTMLE", "ReducedFit"]

#: The two guards, in ``drtmle``'s vocabulary.  **Crossed**, and the commonest thing to
#: transcribe backwards: ``"Q"`` guards against a misspecified *outcome regression* and adds
#: equation (9), which fluctuates ``g``; ``"g"`` guards against a misspecified *mechanism*
#: and adds equation (10), which fluctuates ``Qbar``.  The keyword names the nuisance you are
#: worried about, not the one the equation moves.
GUARDS = ("Q", "g")


@dataclass(frozen=True)
class ReducedFit:
    """What the reduced-dimension regressions were, reported under ``result.extra["drtmle"]``.

    Attributes
    ----------
    guard:
        Which extra equations the fit solved.  Empty means none, and such a fit *is* a plain
        TMLE -- it carries no reduced regressions at all rather than carrying them and
        declining to use them, which is what makes it bit-for-bit the ordinary estimator.
    reduction:
        ``"univariate"``, Benkeser et al.'s three regressions and ``drtmle``'s own default.
    g_bounds:
        The truncation :math:`g_{r,2}`'s target was formed at, which is fixed at fit time --
        see :func:`~cleverly.estimators.reduced.fit_reduced`.  On record because a reader of
        a truncation curve needs to know which parts of the sweep reached these arrays.
    diagnostics:
        Super Learner diagnostics per regression, keyed ``"qr"``, ``"gr1"`` and ``"gr2"``.
    """

    guard: tuple[str, ...]
    reduction: str
    g_bounds: tuple[float, float]
    diagnostics: dict[str, list[SuperLearnerDiagnostics]] = field(default_factory=dict)


class DRTMLE(TMLE):
    r"""TMLE with doubly-robust inference, for a binary point treatment.  **In progress.**

    Reports ``ey1``, ``ey0`` and ``ate`` under those names -- a different estimator behind
    the same parameters, exactly as :class:`~cleverly.CTMLE` is -- with an influence curve
    and therefore an interval that stays valid when only one of the two nuisances is
    consistently estimated.

    **Read the module docstring's warning before using this in anger.**  The curve it
    reports is transcribed from the R package rather than derived, nothing has been compared
    against that package's numbers, and no study here demonstrates the interval is better
    than a plain TMLE's.  What the module docstring says about what this does and does not
    buy is not hedging: it is the current state of the evidence.

    Every :class:`~cleverly.TMLE` keyword is accepted and behaves identically except the
    ones listed under *Notes*, which are refused rather than approximated.

    Parameters
    ----------
    guard:
        Which extra score equations to solve, in ``drtmle``'s vocabulary and **crossed** the
        way that package crosses it -- see :data:`GUARDS`.  Both by default.  An empty guard
        fits no reduced regressions and is a plain TMLE, bit for bit.
    reduction:
        ``"univariate"`` (default) is Benkeser et al. (2017)'s three univariate regressions.
        ``"bivariate"`` -- van der Laan (2014)'s original single bivariate reduced mechanism
        -- is derived but not written, and is refused by name.
    reduced_outcome_learner, reduced_treatment_learner:
        Learners for the reduced-dimension regressions, defaulting to the specifications the
        primary nuisances use.  Two rather than one because the tasks differ:
        :math:`g_{r,1}` is a conditional probability and the other two are conditional means
        of a signed quantity.  **A learner *instance* built for classification cannot serve**
        :math:`Q_r`, whose target is an outcome residual -- if ``outcome_learner=`` is an
        object rather than a name, name a regression learner here.

    Notes
    -----
    Refused by name, each because the derivation read here does not cover it rather than
    because the loop would not run:

    * a multi-valued or continuous treatment, and ``reduction="bivariate"``;
    * ``att``/``atc`` and the ``interventions=``, ``shifts=``, ``incremental=`` and ``msm=``
      axes -- each is a different score equation with no reduced-dimension derivation;
    * ``delta=`` and ``intermediate=`` -- the equations above carry no missingness or
      intermediate factor;
    * ``targeting_scheme="fold"`` and ``cv_evaluation=True`` -- each fold would need its own
      reduced regressions, fitted out of that fold, and its own alternation inside it;
    * combining with :class:`~cleverly.CTMLE`.  A reduced regression conditions on
      :math:`\\hat g` *as a covariate*, and C-TMLE's :math:`\\hat g` is deliberately not an
      estimate of :math:`g_0`; and C-TMLE scores its path by the loss of the targeted
      :math:`\\bar Q`, so the criterion choosing :math:`\\hat g` presupposes that
      :math:`\\bar Q` is informative -- which is precisely the case this variant insures
      against.

    ``weights=`` is supported for **fixed analysis weights**, meaning what
    :mod:`cleverly.data.weighting` says they mean: the estimand is the parameter of the
    tilted law :math:`dP_w = w\,dP / E[w]`.  It once said the keyword "needs nothing said
    about it", on the grounds that the reduced regressions are fitted by weighted loss and
    every score equation here is weighted -- both true, and neither the claim that needed
    making.  The derivation was read at an *unweighted* law, and transporting it to
    :math:`P_w` needs two things beyond weighted losses: the reduced regressions must be
    conditional expectations under :math:`P_w`, which weighted loss gives; and the mechanism
    they condition on and divide by must be the :math:`P_w`-mechanism rather than
    :math:`g_0`, which holds because they are built from ``nuisance.propensity`` and that
    *is* the weighted fit.  ``tests/unit/test_remainder_drtmle.py`` runs the whole expansion
    at two tilted laws and keeps the wrong transport as a test: reductions taken at the
    sampling law leave a first-order remainder a single guard no longer removes.

    ``repeats=`` is supported and varies exactly one thing here: the *primary* split.  Each
    draw fits its own reduced regressions against its own folds and runs its own
    alternation, and the report is the mean of the draws with the curves averaged
    elementwise.  ``_fit_reduced`` is deliberately unseeded so that a refit matches its fit
    -- see its docstring -- which is what leaves the primary split as the only source of
    draw-to-draw variation.  Two things to know.  ``result.extra["drtmle"]`` describes
    **draw 0 only**, as every read-through attribute on a repeated result does.  And
    checking this is what surfaced the centring defect
    ``tests/unit/test_drtmle_fit.py::TestTheReportedCurveIsNotAlwaysCentred`` records: on
    roughly a quarter of splits the reported curve is not centred while all three
    fluctuation rows report their scores solved.  That is a property of a *draw* and not of
    the averaging, so it is a defect in the fit rather than a reason to refuse ``repeats=``.

    Where it stops is an **estimated** weight.  Nothing read here says what the reduced
    regressions of a random tilt are, and the ordinary answer -- that the interval conditions
    on the weights, as ``weights_estimated=`` declares -- is an argument about :math:`D^*`
    rather than about :math:`Q_r`, :math:`g_{r,1}` and :math:`g_{r,2}`.  No **fitted**
    weighted ``DRTMLE`` run exists here either; that is an applied stress test and
    ``docs/roadmap.md`` keeps it open.
    """

    def __init__(
        self,
        *,
        guard: Sequence[str] = GUARDS,
        reduction: str = "univariate",
        reduced_outcome_learner: Any = None,
        reduced_treatment_learner: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.guard = tuple(guard)
        self.reduction = reduction
        self.reduced_outcome_learner = reduced_outcome_learner
        self.reduced_treatment_learner = reduced_treatment_learner
        self._validate_drtmle_settings()

    def _validate_drtmle_settings(self) -> None:
        unknown = [name for name in self.guard if name not in GUARDS]
        if unknown:
            raise ValueError(
                f"guard entries must be drawn from {list(GUARDS)}; got {unknown}. "
                "'Q' guards against a misspecified outcome regression and 'g' against a "
                "misspecified mechanism -- the keyword names the nuisance you are worried "
                "about, not the one the equation it adds fluctuates."
            )
        if len(set(self.guard)) != len(self.guard):
            raise ValueError(f"guard names a guard twice: {list(self.guard)}")
        if self.reduction not in REDUCTIONS:
            raise ValueError(f"reduction must be one of {list(REDUCTIONS)}; got {self.reduction!r}")
        if self.reduction != "univariate":
            refuse_unsupported(self.reduction)
        if self.targeting_scheme == "fold" or self.cv_evaluation:
            raise NotImplementedError(
                "DRTMLE targets pooled only. Fold-wise targeting would need each fold's "
                "reduced-dimension regressions fitted out of that fold and its own "
                "alternation run inside it -- the reductions are regressions *of* an "
                "out-of-fold prediction, so a per-fold alternation is a derivation rather "
                "than a loop over the one here. Use targeting_scheme='pooled'."
            )
        for keyword in ("interventions", "shifts", "incremental", "msm"):
            if getattr(self, keyword, None):
                raise NotImplementedError(
                    f"DRTMLE and {keyword}= are not combined. The reduced-dimension "
                    "regressions are derived for the counterfactual means of a binary "
                    f"treatment; {keyword}= is a different score equation, and no theorem "
                    "read here says what its reductions would be. Fit a plain TMLE, which "
                    "is derived there."
                )

    # --------------------------------------------------------------- the hook

    def _nuisances(
        self,
        data: CausalData,
        folds: Folds,
        scaler: OutcomeScaler,
        config: TMLEConfig,
        intermediate_value: float | None,
        seed: int | None = None,
    ) -> tuple[NuisanceEstimates, dict[str, Any]]:
        """Fit the nuisances, then the reductions *relative to* them.

        Built here rather than inside :func:`~cleverly.estimators._nuisance.fit_nuisances`,
        which is where a shift's or an incremental fit's derived arrays are built, because
        this belongs to one variant rather than to every fit.  The invariant those get
        structurally -- "the derived array and the mechanism it derives from came from one
        out-of-fold model" -- survives because
        :func:`~cleverly.estimators.reduced.fit_reduced` takes the whole object and reads
        ``folds`` off it.
        """
        self._check_drtmle(data)
        base = self._fit_nuisances(data, folds, scaler, intermediate_value, seed=seed)
        if not self.guard:
            # No extra equation to solve, so no reductions to fit and no alternation to
            # enter: `needs_reduction` is False and the fit goes down the ordinary path.
            # That is what makes `guard=()` the plain estimator rather than the plain
            # estimator recovered by a loop that happens to exit after one round.
            return base, {"drtmle": ReducedFit((), self.reduction, config.g_bounds)}

        reduced, diagnostics = self._fit_reduced(data, base, config.g_bounds)
        return (
            replace(base, reduced=reduced),
            {"drtmle": ReducedFit(self.guard, self.reduction, config.g_bounds, diagnostics)},
        )

    def _reduction(self, data: CausalData, nuisance: NuisanceEstimates) -> ReductionSpec | None:
        """The closure the alternation refits with, and the guards it solves.

        The bound comes off ``nuisance.reduced`` rather than off this estimator's settings,
        which is what keeps :math:`g_{r,2}`'s "chosen at fit time" true under a sweep: a
        truncation curve moves the clever covariate's denominator, and it must not silently
        move the array whose *target* was a quotient by a bound the fit declared.
        """
        if not self.guard or nuisance.reduced is None:
            return None
        bounds = nuisance.reduced.g_bounds

        def refit(current: NuisanceEstimates) -> ReducedSet:
            return self._fit_reduced(data, current, bounds)[0]

        return ReductionSpec(refit=refit, guard=self.guard)

    def _fit_reduced(
        self,
        data: CausalData,
        nuisance: NuisanceEstimates,
        g_bounds: tuple[float, float],
    ) -> tuple[ReducedSet, dict[str, list[SuperLearnerDiagnostics]]]:
        """One place resolves the reduced learners, so a refit matches the initial fit.

        Deliberately **not** threaded with a draw's seed, unlike the primary nuisances.  The
        initial fit and every refit inside the alternation go through here, and a seed that
        moved between them would make a ``retarget`` of a fit disagree with the fit itself --
        which is the contract the sensitivity analyses rest on.  What ``repeats=`` averages
        over is the primary nuisances' splits, which do redraw.
        """
        return fit_reduced(
            data,
            nuisance,
            regression_learner=self._resolve_learner(
                self.reduced_outcome_learner, task="regression", fallback=self.outcome_learner
            ),
            classification_learner=self._resolve_learner(
                self.reduced_treatment_learner,
                task="classification",
                fallback=self.treatment_learner,
            ),
            g_bounds=g_bounds,
            reduction=self.reduction,
            n_jobs=self.n_jobs,
        )

    def _check_drtmle(self, data: CausalData) -> None:
        """The refusals that need the data, each naming what the derivation would need."""
        if isinstance(self, CTMLE):
            raise NotImplementedError(
                "DRTMLE and CTMLE are not combined, and the reason is a derivation rather "
                "than plumbing. A reduced-dimension regression conditions on g-hat *as a "
                "covariate*, and C-TMLE's g-hat is deliberately not an estimate of g_0 -- "
                "the collaborative point being that g need only adjust for what Qbar "
                "missed. C-TMLE also scores its path by the cross-validated loss of the "
                "targeted Qbar, so the criterion choosing g-hat presupposes that Qbar is "
                "informative, which is precisely the case this variant insures against."
            )
        if data.is_continuous_treatment:
            refuse_unsupported("continuous")
        if data.n_arms != 2:
            refuse_unsupported(
                "multi_arm",
                f"{data.treatment_name} has {data.n_arms} levels {list(data.treatment_levels)}.",
            )
        if data.has_missing_outcome or data.has_intermediate:
            missing = "delta=" if data.has_missing_outcome else "intermediate="
            raise NotImplementedError(
                f"DRTMLE and {missing} are not combined. Equations (9) and (10) are stated "
                "for a fully observed outcome and no intermediate; a further mechanism "
                "factor would sit inside the reduced regressions' own definitions, not "
                "merely in the clever covariate, and no theorem read here says what it is. "
                "Fit a plain TMLE, which is derived there."
            )
        estimands = resolve_estimands(self.estimands, data.family, data.n_arms)
        outside = [name for name in estimands if name not in MEAN_GROUP_ESTIMANDS]
        if outside:
            raise NotImplementedError(
                f"DRTMLE does not support estimand(s) {outside}: the reduced-dimension "
                "regressions are derived for the counterfactual means, and the ATT and ATC "
                "clever covariates are a propensity odds conditioning on a random event -- "
                "a different score equation with its own reductions to derive. Request them "
                f"from a plain TMLE, or set estimands={sorted(MEAN_GROUP_ESTIMANDS)!r}."
            )
