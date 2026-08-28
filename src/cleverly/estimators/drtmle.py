r"""Doubly-robust nonparametric inference: a TMLE whose *interval* survives one bad nuisance.

.. warning::

   **What this variant ships under is *conditional validity*.**  The default algorithm computes
   what Benkeser et al.'s Theorem 1 derives, and the bivariate option computes van der
   Laan's Theorem 3 construction armwise -- checked against the theorem's appendices, against
   the Gateaux derivative of the parameter, against exact finite-support laws, and against
   the remainder identities -- and the interval it reports is valid **conditional on** the
   caller obtaining
   adequate primary *and reduced-regression* fits.  Those are method-specific rate conditions on
   estimated functions; they are not verifiable from a fit's own output, and in particular
   **numerical score convergence does not verify them**.  The contract is under
   ``docs/technical-reference/dr-tmle/``
   in full; three things a caller's numbers depend on are here.

   1. **Solved score equations validate targeting, not nuisance adequacy**, and this is the
      one that is easy to get backwards.  A fit with badly wrong reduced regressions returns
      a ``psi``, an ``se`` and an interval formatted exactly like a good one, with every
      score green.  ``tests/unit/test_oracle_reductions.py`` is the evidence rather than the
      caveat: with **exact** reductions the estimator recovers the truth despite misspecified
      primary nuisances, and with **wrong** ones the estimate moves while every score
      equation still passes.  Inspect the reduced fits themselves --
      ``result.extra["drtmle"].diagnostics``, keyed ``"qr"``, ``"gr1"``, ``"gr2"`` on the
      univariate reduction, ``"qr"``, ``"gr1"`` on the bivariate reduction, and
      ``"gamma_a"``, ``"gamma_m"``, ``"r_a"``, ``"r_m"``, ``"e"`` on the missing-outcome one.
   2. **The interval is demonstrably better than a plain TMLE's, and it is not nominal.**
      Over 6,000 fits in two independent seed batches, the plain interval covers
      ``0.532``/``0.472`` against this estimator's ``0.844``/``0.848`` at ``n = 2,400`` in the
      cell built for it -- a paired ``+0.312`` and ``+0.376``.  **And it reaches ``0.95``
      nowhere**, the best reading anywhere being ``0.880``.  Two measured quantities account
      for the shortfall and are one premise measured twice: the second-order remainder
      Theorem 1 assumes negligible does not vanish at these sizes, and the reported ``se``
      runs about 10% short of the spread it covers in ``q-drift`` and about 16% *long* in
      ``g-drift`` -- so the second is **not** a separate defect in the variance estimator.
      ``sigma^2_n`` is Theorem 1's own ``P_n{D* - D_A - D_Y}^2``, valid to first order
      exactly when the condition the first quantity fails holds.  The reductions in that
      study were fitted by ``glm``, so it measures a configuration rather than the theorem's
      condition.  Use this where you have a reason to think one nuisance is badly estimated;
      do not treat the interval as settled.
   3. **The alternation is not guaranteed to converge, though it mostly does.**
      Equation (10)'s covariate becomes small on exactly the fits anybody wants -- see
      :func:`~cleverly.estimators.targeting.solve_with_reduction` -- so a draw can exit at
      the outer cap or an inner solve can stop at working precision. The original 96-fit
      sweep had cross-fitting disabled and therefore did not cover the shipped 10-fold
      default. A fixed-seed default-path check observed numerical difficulty in 3 of 46
      rounds at ``n=200``, 2 of 16 at ``n=1000``, and none of 8 at ``n=3000``; every final
      score check passed. No argument here *proves* the iterates approach a common zero,
      which is why ``validate()`` warns on those rounds even when the score verdict passes.

   Two things that were open and are closed, kept because both are the kind of defect that
   returns.  The **sign** of the mechanism correction is the appendices' orientation and not
   the §3.1 display's -- nothing reporting a point estimate could have caught a flip, since
   all three empirical means are driven to zero and what it moves is the variance;
   ``tests/unit/test_theorem_drtmle.py`` pins it at a **nonzero** :math:`Q_r`, which is the
   only place a sign is visible.  And the reported curve is **centred where the mechanism
   truncation binds**: equation (9) is solved at the truncated tilt, which is the expression
   the curve carries (:func:`~cleverly.fluctuation.mechanism.solve_bounded_mechanism`), so a
   fit whose bound never binds is bit for bit what it was, and one whose bound does binds
   holds every state identity at ``1e-17``.

   **The instrument that found the second stays**, and is why that fix is checkable rather
   than asserted: :func:`~cleverly.validation.drtmle.correction_check` recomputes each arm's
   :math:`P_n[w D^*_g]` and :math:`P_n[w D^*_Q]` from the exact returned state and reports the
   residual against the score the targeting step recorded, per arm and per equation, on every
   doubly-robust fit.  No threshold in it was loosened to make those rows pass.

Subject to the method's remaining regularity conditions, an interval is valid when the
second-order remainder is negligible.  For a plain TMLE, the absolute remainder has a
constant-times-product bound on the nuisance errors under positivity.  That bound explains
why the *point estimate* is doubly robust; but the interval needs
:math:`\sqrt n R_2 \to 0`, and with one factor not shrinking :math:`R_2` is first order in
the other.  So **`TMLE` is doubly robust for consistency and singly robust for inference**,
and this class closes that second gap.

It does it by solving two further score equations (van der Laan 2014; Benkeser, Carone,
van der Laan & Gilbert 2017; Benkeser & Hejazi 2023), built from *reduced-dimension*
regressions of each nuisance's residual on the other:

.. math::

    (9)  \quad & P_n[\, Q_r(a, W)/g^*(a|W)\,\{1_a - g^*(a|W)\}\,] = 0 \\
    (10) \quad & P_n[\, 1_a\,g_{r,2}(a|W)/g_{r,1}(a|W)\,\{Y - \bar Q^*(a, W)\}\,] = 0

Those are the default univariate equations.  The bivariate alternative replaces the
two reduced mechanisms by :math:`g_r=P(A=a\mid\hat{\bar Q}_a,\hat g_a)` and equation (10)'s
covariate by :math:`1_a(g_r-g)/(g g_r)`, separately for every discrete arm.
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
When both nuisances are consistent, the corrections converge to zero and the curve approaches
the ordinary efficient curve.  At the true nuisance functions, they vanish row by row.  The
distinction is easy to lose because the numbers point the other way: in the
guide's worked example the corrected standard error is the **smaller** of the two, 0.06828
against 0.06850, which is a fact about one draw and not a general narrowing.  A doubly-robust
fit's ``score_check`` says so in its own verdict rather than signing the fit off as having
solved the efficient score equation.

**What it costs.** Three reduced-family fits per arm per round for the univariate reduction,
or two for the bivariate reduction, plus a mechanism fluctuation. Each family is fitted once,
at the nuisance vintage its next equation consumes; the former implementation fitted every
family at both refit sites and discarded or overwrote half the work. A truncation curve or
an MNAR sweep on a
``DRTMLE`` result therefore costs about a fit per point rather than a fraction of one:
``retarget`` here is no longer arithmetic on cached arrays, which is the one contract this
variant breaks and the reason it is a class of its own rather than a keyword.

Scope follows the vetted R implementation for arbitrary discrete treatment levels with either
complete-outcome reduction, on the ``mean`` group.  The multi-arm bivariate construction is the
pinned implementation's armwise extension of van der Laan's binary theorem, not a claim that the
theorem itself was stated for multiple arms.  It also includes Díaz & van der
Laan (2017)'s binary randomized-trial construction for MAR outcomes, without cross-fitting;
there five reductions and separate treatment, observation and outcome tilts replace the
complete-data pair. Continuous treatment, observational missing outcomes, missing treatment,
and other target groups remain refused by name.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import copy
from dataclasses import dataclass, field, replace
from typing import Any, cast

import numpy as np

from ..data.causal_data import CausalData
from ..learners.crossfit import Folds
from ..learners.library import _validate_learner
from ..learners.super_learner import SuperLearnerDiagnostics
from ..utils.bounds import OutcomeScaler
from ._nuisance import NuisanceEstimates, Propensity, fit_inner_designs
from .base import MEAN_GROUP_ESTIMANDS, TMLEConfig, resolve_estimands
from .ctmle import CTMLE
from .reduced import (
    REDUCED_CROSSFITS,
    REDUCTIONS,
    MissingOutcomeReducedSet,
    ReducedFamily,
    ReducedSet,
    fit_missing_outcome_reduced,
    fit_reduced,
    refuse_unsupported,
)
from .targeting import DEFAULT_MAX_OUTER, ReductionOrder, ReductionSpec
from .tmle import TMLE

__all__ = ["DRTMLE", "ReducedFit"]

#: The two source-defined routes through a round. ``"drtmle"`` is the canonical R
#: package's loop and the default; ``"benkeser"`` is the published six-step recursion.
#: The theorem's exit is a fixed point rather than a route, so the update-order comparison
#: measures whether the two reach the same returned collection.
UPDATE_ORDERS = ("drtmle", "benkeser")

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
        The construction that was actually fitted, read off the reduced set rather than off
        the constructor argument so the two cannot drift: ``"univariate"`` for Benkeser et
        al.'s three regressions and ``drtmle``'s own default, ``"bivariate"`` for van der
        Laan's joint reduced probability, and ``"missing_outcome"`` for Díaz & van der
        Laan's five.  ``guard=()`` fits none and reports the setting asked for, there being
        no fit to report instead.
    g_bounds:
        The truncation :math:`g_{r,2}`'s target was formed at, which is fixed at fit time --
        see :func:`~cleverly.estimators.reduced.fit_reduced`.  On record because a reader of
        a truncation curve needs to know which parts of the sweep reached these arrays.
    missingness_bound:
        The truncation :math:`\\gamma_\\Delta` and :math:`g_{r,\\Delta}`'s targets were formed
        at, for the same reason ``g_bounds`` is recorded -- a missing-outcome fit forms two
        of its five reductions at *this* bound and not at ``g_bounds``, so a reader with
        only the latter cannot say which sweep reached them.  ``None`` where no reduction
        was formed at it.
    diagnostics:
        Super Learner diagnostics per regression.  Keyed ``"qr"``, ``"gr1"`` and ``"gr2"``
        on the univariate reduction, ``"qr"`` and ``"gr1"`` on the bivariate one, and
        ``"gamma_a"``, ``"gamma_m"``, ``"r_a"``, ``"r_m"`` and ``"e"`` on the missing-outcome
        one -- the constructions do not fit the same regressions, so they cannot report under
        the same names.
    """

    guard: tuple[str, ...]
    reduction: str
    g_bounds: tuple[float, float]
    diagnostics: dict[str, list[SuperLearnerDiagnostics]] = field(default_factory=dict)
    missingness_bound: float | None = None

    @staticmethod
    def evaluation(result: Any) -> Any:
        """The targeted companion of a fitted result, or ``None`` if it declared none.

        A one-line accessor rather than a field, because the companion is not known until
        the alternation has finished and this object is built before it: it lives on
        :attr:`~cleverly.estimators.targeting.ReductionFluctuation.evaluation`, which is
        where every other array the alternation produced lives, and reading it off the
        fluctuation is what keeps one copy of it.
        """
        fluctuation = result.repeats[0].fluctuations.get("mean")
        reduction = getattr(fluctuation, "reduction", None)
        return None if reduction is None else reduction.evaluation


class DRTMLE(TMLE):
    r"""TMLE with doubly-robust inference, for a discrete point treatment.

    Reports treatment-specific means and reference-arm contrasts -- a different estimator behind
    the same parameters, exactly as :class:`~cleverly.CTMLE` is -- with an influence curve
    and therefore an interval that stays valid when only one of the two nuisances is
    consistently estimated.

    **Read the module docstring's warning before using this in anger**, and
    ``docs/technical-reference/dr-tmle/`` carries the contract in full.  The curve it
    reports began as an *equation* read off the R package rather than as one derived
    here -- that is its *provenance*, and its *evidence* is that
    it has since been checked against Theorem 1's appendices and against the Gateaux
    derivative of the parameter, and agrees with both.  The registered canonical DR-TMLE study
    separately compares its binary complete-data numbers with the pinned R package under the
    paper's nuisance-correctness regimes; that bounded comparison is not the derivation.  A
    controlled study here **does**
    demonstrate the interval is better than a plain TMLE's, by a wide and reproduced margin,
    and also that it does not attain nominal coverage anywhere in that study.  What the
    module docstring says about what this does and does not buy is not hedging: it is what
    the evidence establishes, and the interval is offered as valid *conditionally* on
    nuisance conditions a fit cannot check for itself.

    Every :class:`~cleverly.TMLE` keyword is accepted and behaves identically except the
    ones listed under *Notes*, which are refused rather than approximated.

    Parameters
    ----------
    guard:
        Which extra score equations to solve, in ``drtmle``'s vocabulary and **crossed** the
        way that package crosses it -- see :data:`GUARDS`.  Both by default.  An empty guard
        fits no reduced regressions and is a plain TMLE, bit for bit.

        Missing-outcome fits require both guards because Díaz & van der Laan's algorithm
        targets all three correction blocks together; partial guards are refused there.

        It also says which corrections the reported curve subtracts -- one per equation
        solved, so ``guard=("g",)`` reports :math:`D = D^* - D^*_Q` and the score check's
        verdict names that curve.  The other equation's correction is still recomputed and
        reported, held to no threshold, because it is what says what the guard did not buy.
        Subtracting both regardless of the guard is a correctness defect.
    reduction:
        ``"univariate"`` (default) is Benkeser et al. (2017)'s three univariate regressions.
        ``"bivariate"`` -- van der Laan (2014)'s original single bivariate reduced mechanism
        -- fits :math:`P(A=a\mid\hat{\bar Q}(a,W),\hat g(a\mid W))` and uses its distinct
        outcome-drift score, as in the pinned canonical R package.  It is available for discrete
        complete-outcome fits; univariate remains the default.
    update_order:
        On complete-outcome fits, which route a round of the alternation takes,
        ``"drtmle"`` (default) or
        ``"benkeser"``.  **A diagnostic keyword rather than a tuning one**, and the reason is
        the update-order question: the 2016 working paper's step 7 states
        its own termination as the three empirical means being approximately zero, so its
        six-step order is one route to a fixed point rather than something Theorem 1
        assumes about the collection returned. Missing-outcome fits always use their
        dedicated published cycle. ``"benkeser"`` implements the complete-data order beside
        R ``drtmle``'s -- equation (8), then :math:`g_{r,1}` and :math:`g_{r,2}` at the
        **once-updated** outcome regression, then equation (10), then :math:`Q_r` at the
        **twice-updated** one, then equation (9) -- so that *whether the two reach the same
        fixed point on real data* is something a sweep measures rather than something a
        document asserts.  Both share the stopping rule, the stall test and the closing
        pass, deliberately: what is in question is the route.

        Two cautions carry over from ``docs/technical-reference/dr-tmle/targeting.md``.
        Compare the **scores and the estimates**,
        never the fluctuation coefficients: the submodels a round passes through differ, so
        an ``epsilon`` from one is not an ``epsilon`` from the other.  And compare the two at
        the **same nuisances** -- the same data, the same ``random_state`` -- since the
        initial fits are all either route has in common.
    reduced_crossfit:
        How fold ``k``'s reduced regressions get their **training** rows' design and target:
        ``"pooled"`` (default) reuses the primary split as it stands, and ``"nested"`` takes
        them from primary models fitted with fold ``k`` left out as well.  **A diagnostic
        keyword rather than a tuning one**, exactly as ``update_order`` is, and for the same
        kind of reason: the diagnostic asks whether the cheap construction's
        induced dependence is higher order, the argument for it needs one quantity to
        vanish, and that quantity *is* the difference between these two.  So the expensive
        one exists to be measured rather than to be used.

        It is refused below ``n_folds=3`` and under ``cross_fit=False`` -- there is no
        complement to leave a fold out of -- and, with ``targeting="one_step"``, by name.
        It costs `K` times the primary nuisance fitting, paid once; what actually dominates
        is that the nested reductions are noisier, so equation (10)'s near-singular solve
        takes more rounds.  Measured at 1.3x to 17x a pooled fit's wall clock over four
        draws, on two of which it reached the outer cap.
    evaluation:
        An independent draw -- a dataframe or a prepared
        :class:`~cleverly.data.CausalData` -- at which this fit's nuisances are **also**
        evaluated, one copy per outer fold, and moved by the same targeting steps the
        fitted arrays take.  **A third diagnostic keyword**, alongside ``update_order`` and
        ``reduced_crossfit``, and for the same kind of reason: the remainder diagnostic
        asks whether :math:`\\sqrt n R_{\\text{remaining}} \\to 0`, that needs
        :math:`P_0\\hat D` -- the population mean of the **fitted** curve, for which
        :math:`P_n\\hat D` is refused since targeting drove it to zero -- and a curve is a
        function of :math:`(W, A, Y)` that no array of out-of-fold predictions defines
        anywhere else.

        The companion contributes to no fit, no fold and no score, so a fit that declares
        one is **bit for bit** a fit that does not; ``tests/unit/test_drtmle_companion.py``
        is what holds that rather than the sentence.  What it costs is one further
        prediction per fold per nuisance per round, and no further learner fit.

        It arrives on
        :attr:`~cleverly.estimators._nuisance.NuisanceEstimates.companion` and, once the
        alternation has moved it, on :attr:`ReducedFit.evaluation`.  Refused with
        ``repeats=``, ``targeting="one_step"`` and ``target_weights=True``, each by name.
    reduced_outcome_learner, reduced_treatment_learner:
        Learners for the reduced-dimension regressions, defaulting to the specifications the
        primary nuisances use.  Two rather than one because the tasks differ:
        :math:`g_{r,1}` is a conditional probability and the other two are conditional means
        of a signed quantity.  **A learner *instance* built for classification cannot serve**
        :math:`Q_r`, whose target is an outcome residual -- if ``outcome_learner=`` is an
        object rather than a name, name a regression learner here.
    randomized:
        Declare that treatment was randomized for a fit with ``delta=``.  The treatment
        learner is still fitted, following Díaz & van der Laan's finite-sample recommendation.
        To use known probabilities instead, pass row-aligned ``treatment_probabilities=``
        to :meth:`fit`; doing so bypasses the treatment learner.
    max_outer:
        How many rounds the three-equation alternation may run.  **Not** ``max_iter``, which
        is the cap on the Newton steps *inside* one fluctuation and which every estimator
        here carries; the two were one keyword away from being confused and the confusion had
        already happened.  The registered DR-TMLE study published ``max_iter: 100`` as its
        alternation setting -- R ``drtmle``'s ``maxIter`` -- while the loop ran at a
        hard-coded 50 that no caller could reach.  The value that applied is now on
        :attr:`~cleverly.estimators.targeting.ReductionFluctuation.max_outer`, so a manifest
        cannot claim one cap while another ran.

        Raising it changes only the fits that reached it, and on those it can matter.  Measured
        on the paper law at ``n = 3000`` over 120 draws, going from 50 to 100 left every fit
        that exited on tolerance bit for bit and moved the 7 that had hit the cap by up to
        ``1.3e-3``, which is 7% of one sampling standard deviation.  So a ``"cap"`` exit is
        worth acting on rather than noting: it says this draw had not settled, and the
        estimate it reports is the one the loop happened to stop at.

    Notes
    -----
    With ``cross_fit=True``, the supported complete-outcome construction is the pinned R
    ``drtmle`` package's ``cvFolds`` path: the primary and reduced regressions predict out
    of fold, their row-aligned predictions enter one pooled alternation, and the report is
    the whole-sample plug-in with covariance from the rowwise corrected curve.  In this
    package's more explicit vocabulary that is ``targeting_scheme="pooled"`` and
    ``cv_evaluation=False``.  This source mapping is implementation provenance for the
    package's separate cross-fitting argument; the published theorem itself is not
    cross-fitted.

    Refused by name where the derivation read here does not cover it rather than because the
    loop would not run:

    * a continuous treatment, whose density-based equations are not derived here;
    * ``att``/``atc`` and the ``interventions=``, ``shifts=``, ``incremental=`` and ``msm=``
      axes -- each is a different score equation with no reduced-dimension derivation;
    * observational treatment with ``delta=``, missing treatment, and ``intermediate=``.
      Díaz & van der Laan (2017) covers binary randomized treatment with MAR outcomes;
      the other compositions need their own corrected curve and remainder;
    * ``targeting_scheme="fold"`` -- each fold would need its own reduced regressions and
      alternation; and ``cv_evaluation=True`` -- the common-update construction would need
      the corrected parameter and curve derived under fold-wise evaluation;
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
    alternation, and the report is the median of the draws with split dispersion included
    elementwise.  ``_fit_reduced`` is deliberately unseeded so that a refit matches its fit
    -- see its docstring -- which is what leaves the primary split as the only source of
    draw-to-draw variation.  Two things to know.  ``result.extra["drtmle"]`` describes
    **draw 0 only**, as every read-through attribute on a repeated result does.  And
    checking this is what surfaced the centring defect
    ``tests/unit/test_drtmle_fit.py::TestTheReportedCurveIsNotAlwaysCentred`` records: on
    roughly a quarter of splits the reported curve is not centred while all three
    fluctuation rows report their scores solved.  That is a property of a *draw* and not of
      the aggregation, so it is a defect in the fit rather than a reason to refuse ``repeats=``.

    Where it stops is an **estimated** weight.  Nothing read here says what the reduced
    regressions of a random tilt are, and the ordinary answer -- that the interval conditions
    on the weights, as ``weights_estimated=`` declares -- is an argument about :math:`D^*`
    rather than about :math:`Q_r`, :math:`g_{r,1}` and :math:`g_{r,2}`.  ``docs/roadmap.md``,
    *D2. Other refused DR-TMLE compositions*, keeps estimated weights refused until a paper
    supplies the influence contribution for estimating them.  No **fitted** weighted
    ``DRTMLE`` run exists here either; that is an applied stress test nothing has run, and
    no document tracks it as scheduled work.
    """

    def __init__(
        self,
        *,
        guard: Sequence[str] = GUARDS,
        reduction: str = "univariate",
        reduced_outcome_learner: Any = None,
        reduced_treatment_learner: Any = None,
        reduced_crossfit: str = "pooled",
        update_order: ReductionOrder = "drtmle",
        evaluation: Any = None,
        randomized: bool = False,
        max_outer: int = DEFAULT_MAX_OUTER,
        **kwargs: Any,
    ) -> None:
        _validate_learner(reduced_outcome_learner, "reduced_outcome_learner")
        _validate_learner(reduced_treatment_learner, "reduced_treatment_learner")
        super().__init__(**kwargs)
        self.max_outer = int(max_outer)
        self.guard = tuple(guard)
        self.reduction = reduction
        self.reduced_outcome_learner = reduced_outcome_learner
        self.reduced_treatment_learner = reduced_treatment_learner
        self.reduced_crossfit = reduced_crossfit
        self.update_order = update_order
        self.evaluation = evaluation
        self.randomized = bool(randomized)
        self._treatment_probabilities: Any = None
        self._validate_drtmle_settings()

    def fit(
        self,
        data: Any,
        *,
        treatment_probabilities: Any = None,
        **roles: Any,
    ) -> Any:
        """Fit, optionally using known randomized-treatment probabilities.

        ``treatment_probabilities`` is accepted at fit time because it is row-aligned data.
        Three forms:

        - a **mapping keyed by treatment level**, ``{"placebo": p0, "active": p1}``, with
          one ``(n,)`` column per arm and every arm named.  Prefer this: it says which arm
          each column belongs to instead of relying on the caller and the encoder agreeing
          about which level sorts first.
        - ``(n, 2)`` in encoded arm order, which is the levels sorted ascending.
        - ``(n,)``, read as the probability of the arm whose code is ``1`` -- the *second*
          sorted level, so ``"placebo"`` in a trial labelled ``active``/``placebo``.

        Supplying any of them implies ``randomized=True`` for the missing-outcome theorem
        and bypasses the treatment learner.  A shallow per-fit copy keeps an unfitted
        estimator reusable.
        """
        fitted = copy(self)
        fitted._treatment_probabilities = treatment_probabilities
        return TMLE.fit(fitted, data, **roles)

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
        if self.max_outer < 1:
            raise ValueError(
                f"max_outer must be at least one round; got {self.max_outer}. It caps the "
                "three-equation alternation, not the Newton steps inside one fluctuation -- "
                "that is max_iter."
            )
        if self.update_order not in UPDATE_ORDERS:
            raise ValueError(
                f"update_order must be one of {list(UPDATE_ORDERS)}; got "
                f"{self.update_order!r}. 'drtmle' is the canonical R package's loop and "
                "the default; 'benkeser' is the published six-step recursion. The former "
                "names 'cleverly' and 'paper' are no longer accepted."
            )
        if self.reduction not in REDUCTIONS:
            raise ValueError(f"reduction must be one of {list(REDUCTIONS)}; got {self.reduction!r}")
        if self.reduced_crossfit not in REDUCED_CROSSFITS:
            raise ValueError(
                f"reduced_crossfit must be one of {list(REDUCED_CROSSFITS)}; got "
                f"{self.reduced_crossfit!r}. 'pooled' reuses the primary split for the "
                "reduced regressions and is what ships; 'nested' refits the primary "
                "nuisances leaving each outer fold out as well, so that whether the cheap "
                "construction's induced dependence matters is measured rather than argued "
                "(see the reduced-cross-fitting diagnostic)."
            )
        if self.reduced_crossfit == "nested":
            if not self.cross_fit:
                raise ValueError(
                    "reduced_crossfit='nested' needs cross-fitting: it trains each fold's "
                    "reduced regression on primary models that left that fold out, and "
                    "cross_fit=False has one fold and no complement to leave it out of. "
                    "Pass cross_fit=True, or reduced_crossfit='pooled'."
                )
            if self.n_folds < 3:
                raise ValueError(
                    "reduced_crossfit='nested' leaves two folds out at a time -- the fold "
                    "being predicted and the fold being trained on -- so it needs at least "
                    f"three; got n_folds={self.n_folds}."
                )
        if self.evaluation is not None:
            # Each refused because the companion would come back describing a fit nobody
            # ran, and none of the three would raise on its own.
            if self.repeats > 1:
                raise ValueError(
                    "evaluation= and repeats= are not combined. Each draw of the split "
                    "targets its own alternation, so there would be one companion per draw "
                    "and no single state for P_0 D-hat to be the mean of -- and "
                    "result.extra['drtmle'] already describes draw 0 only. Fit each draw "
                    "separately, or drop repeats=."
                )
            if self.targeting == "one_step":
                raise NotImplementedError(
                    "evaluation= and targeting='one_step' are not combined, on cost rather "
                    "than on derivation -- the same refusal reduced_crossfit='nested' takes. "
                    "The companion is moved by the same steps the fitted arrays take, and "
                    "the one-step walk takes up to 20,000 of them with an adaptive length. "
                    "Use targeting='iterative', which is the default."
                )
            if self.target_weights:
                raise NotImplementedError(
                    "evaluation= and target_weights=True are not combined. The weighted form "
                    "of the submodel divides the covariate by the *fitting* sample's weights, "
                    "and a companion row has none of them -- so the companion would travel "
                    "along a covariate belonging to other rows while reporting itself as "
                    "that fit's."
                )
        if self.targeting_scheme == "fold" or self.cv_evaluation:
            raise NotImplementedError(
                "DRTMLE supports the canonical cvFolds mapping only: cross-fitted primary "
                "and reduced regressions followed by one pooled alternation and report. "
                "targeting_scheme='fold' would "
                "need each fold's reduced regressions and alternation; cv_evaluation=True "
                "would need the corrected parameter and influence curve derived under "
                "fold-wise evaluation. Neither follows by looping over the pooled "
                "reduction. Use targeting_scheme='pooled', cv_evaluation=False."
            )
        for keyword in ("interventions", "shifts", "incremental", "msm"):
            if getattr(self, keyword, None):
                raise NotImplementedError(
                    f"DRTMLE and {keyword}= are not combined. The reduced-dimension "
                    "regressions are derived for counterfactual means under static treatment "
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
        known = self._known_treatment_probabilities(data)
        base = self._fit_nuisances(
            data,
            folds,
            scaler,
            intermediate_value,
            seed=seed,
            companion=self._companion(data),
            fit_treatment=known is None,
        )
        if known is not None:
            base = replace(base, propensity=known)
        if self.guard and self.reduced_crossfit == "nested":
            # Before the reductions, because they read it. Once per fit rather than once
            # per round: every refit inside the alternation moves these arrays by the
            # fluctuation the production ones take rather than re-learning them.
            base = replace(
                base,
                inner=fit_inner_designs(
                    data,
                    base,
                    outcome_learner=self._resolve_learner(
                        self.outcome_learner, task=base.outcome_task, seed=seed
                    ),
                    treatment_learner=self._resolve_learner(
                        self.treatment_learner, task="classification", seed=seed
                    ),
                    n_jobs=self.n_jobs,
                ),
            )
        if not self.guard:
            # No extra equation to solve, so no reductions to fit and no alternation to
            # enter: `needs_reduction` is False and the fit goes down the ordinary path.
            # That is what makes `guard=()` the plain estimator rather than the plain
            # estimator recovered by a loop that happens to exit after one round.
            # Nothing was fitted, so `reduction` is the setting that was asked for rather
            # than a construction that ran -- there is no reduced set to read it off.
            return base, {
                "drtmle": ReducedFit(
                    (),
                    self.reduction,
                    config.g_bounds,
                    missingness_bound=(
                        config.missingness_bound if data.has_missing_outcome else None
                    ),
                )
            }

        reduced, diagnostics, at_companion = self._fit_reduced(data, base, config.g_bounds)
        if base.companion is not None:
            base = replace(
                base,
                companion=replace(
                    base.companion,
                    reduced=cast(tuple[ReducedSet, ...], at_companion),
                ),
            )
        missing_outcome = isinstance(reduced, MissingOutcomeReducedSet)
        return (
            replace(base, reduced=reduced),
            {
                "drtmle": ReducedFit(
                    self.guard,
                    # Read off the set that was fitted rather than off `self.reduction`,
                    # so the label and the construction cannot drift apart.
                    str(reduced.reduction),
                    config.g_bounds,
                    diagnostics,
                    missingness_bound=config.missingness_bound if missing_outcome else None,
                )
            },
        )

    def _known_treatment_probabilities(self, data: CausalData) -> Propensity | None:
        """The design probabilities as a mechanism, or ``None`` when none were supplied.

        Three accepted forms, and the mapping is the one to reach for.  A trial's arms are
        named, and the positional forms bind to the arm *codes* -- which are indices into
        the sorted levels, so ``1`` is ``"placebo"`` in a trial labelled
        ``active``/``placebo``.  A caller who reads ``(n,)`` as "the probability of
        treatment" and passes ``P(A = 'active' | W)`` inverts the design, and with unequal
        allocation nothing downstream contradicts them: the array is in range, its rows sum
        to one, and the treatment learner it replaces never runs.  So the mapping form
        exists to let the caller name the arm, exactly as ``arm_gamma`` does in
        :func:`~cleverly.sensitivity.missingness_tilt`, and the two positional forms name
        the level they resolved to in every message they raise.
        """
        supplied = self._treatment_probabilities
        if supplied is None:
            return None
        levels = list(data.treatment_levels)
        if isinstance(supplied, Mapping):
            values = self._probabilities_from_levels(data, supplied, levels)
        else:
            values = np.array(supplied, dtype=float, copy=True)
            if values.ndim == 1:
                if values.shape[0] != data.n:
                    raise ValueError(
                        f"treatment_probabilities has {values.shape[0]} rows; expected {data.n}"
                    )
                values = np.column_stack([1.0 - values, values])
        if values.shape != (data.n, 2):
            raise ValueError(
                f"treatment_probabilities must be (n,) for P({data.treatment_name} = "
                f"{levels[1]!r} | W), (n, 2) in the level order {levels}, or a mapping "
                f"keyed by those levels; got {values.shape}"
            )
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0) or np.any(values >= 1.0):
            raise ValueError("treatment_probabilities must be finite and strictly between 0 and 1")
        if not np.allclose(values.sum(axis=1), 1.0, rtol=0.0, atol=1e-12):
            raise ValueError("each row of treatment_probabilities must sum to one")
        return Propensity(values, data.arm_codes)

    @staticmethod
    def _probabilities_from_levels(
        data: CausalData, supplied: Mapping[Any, Any], levels: list[Any]
    ) -> np.ndarray:
        """One ``(n,)`` column per arm, keyed on the way in by the caller's own level.

        Keyed by level in and by arm code out, which is the convention every reported name
        follows -- and every arm must be named, because an arm left out would be filled in
        by a complement the caller never wrote, which is the assumption this form exists to
        state rather than inherit.
        """
        columns: dict[float, np.ndarray] = {}
        for label, probabilities in supplied.items():
            matches = [index for index, level in enumerate(levels) if level == label]
            if not matches:
                raise ValueError(
                    f"treatment_probabilities names {label!r}, which is not a level of "
                    f"{data.treatment_name}; its levels are {levels}"
                )
            column = np.asarray(probabilities, dtype=float).reshape(-1)
            if column.shape[0] != data.n:
                raise ValueError(
                    f"treatment_probabilities[{label!r}] has {column.shape[0]} rows; "
                    f"expected {data.n}"
                )
            columns[float(matches[0])] = column
        missing = [levels[int(code)] for code in data.arm_codes if code not in columns]
        if missing:
            raise ValueError(
                f"treatment_probabilities must name every arm, and {missing} are missing. "
                "The arms left out would take whatever is left over from the ones named, "
                "which is the design this form exists to state explicitly; give every arm "
                "its own column."
            )
        return np.column_stack([columns[code] for code in data.arm_codes])

    def _reduction(self, data: CausalData, nuisance: NuisanceEstimates) -> ReductionSpec | None:
        """The closure the alternation refits with, and the guards it solves.

        The ordinary reduction keeps the bound declared by its fitted state.  The missing-
        outcome construction instead receives the active treatment and observation bounds
        from targeting, because those bounds define two of its five regression targets and
        therefore must move with a truncation sweep.
        """
        if not self.guard or nuisance.reduced is None:
            return None
        bounds = nuisance.reduced.g_bounds

        def refit(
            current: NuisanceEstimates,
            families: tuple[ReducedFamily, ...],
        ) -> tuple[
            ReducedSet | MissingOutcomeReducedSet,
            tuple[ReducedSet | MissingOutcomeReducedSet, ...],
        ]:
            production, _, at_companion = self._fit_reduced(
                data, current, bounds, families=families
            )
            return production, at_companion

        def missing_refit(
            current: NuisanceEstimates,
            current_bounds: tuple[float, float],
            current_missingness_bound: float,
        ) -> tuple[
            ReducedSet | MissingOutcomeReducedSet,
            tuple[ReducedSet | MissingOutcomeReducedSet, ...],
        ]:
            production, _, at_companion = self._fit_reduced(
                data,
                current,
                current_bounds,
                missingness_bound=current_missingness_bound,
            )
            return production, at_companion

        return ReductionSpec(
            refit=refit,
            missing_refit=(
                missing_refit if isinstance(nuisance.reduced, MissingOutcomeReducedSet) else None
            ),
            guard=self.guard,
            order=self.update_order,
        )

    def _companion(self, data: CausalData) -> CausalData | None:
        """The declared evaluation rows, prepared the way the fitting rows were.

        ``None`` on every fit that declared no ``evaluation=``, which is the ordinary one.
        The frame goes through :meth:`~cleverly.TMLE._prepare` with the *same* roles the fit
        was given, so the companion's design is the design the models were fitted on rather
        than one that merely resembles it -- and a mismatch is refused by
        :func:`~cleverly.estimators._nuisance.fit_nuisances` rather than predicted through.
        """
        if self.evaluation is None:
            return None
        if isinstance(self.evaluation, CausalData):
            return self.evaluation
        # The roles come off the *fitted* container rather than being asked for again, so
        # a companion cannot be prepared under a different reading of the same frame --
        # which would predict the fit's models at a design they were not fitted on and
        # look entirely ordinary doing it.
        return self._prepare(
            self.evaluation,
            outcome=data.outcome_name,
            treatment=data.treatment_name,
            covariates=data.covariate_names,
            delta=None,
            weights=None,
            id=None,
            intermediate=None,
            strata=None,
            treatment_kind="discrete",
        )

    def _fit_reduced(
        self,
        data: CausalData,
        nuisance: NuisanceEstimates,
        g_bounds: tuple[float, float],
        *,
        missingness_bound: float | None = None,
        families: tuple[ReducedFamily, ...] | None = None,
    ) -> tuple[
        ReducedSet | MissingOutcomeReducedSet,
        dict[str, list[SuperLearnerDiagnostics]],
        tuple[ReducedSet | MissingOutcomeReducedSet, ...],
    ]:
        """One place resolves the reduced learners, so a refit matches the initial fit.

        Deliberately **not** threaded with a draw's seed, unlike the primary nuisances.  The
        initial fit and every refit inside the alternation go through here, and a seed that
        moved between them would make a ``retarget`` of a fit disagree with the fit itself --
        which is the contract the sensitivity analyses rest on. What ``repeats=`` takes a median
        over is the primary nuisances' splits, which do redraw.
        """
        regression = self._resolve_learner(
            self.reduced_outcome_learner, task="regression", fallback=self.outcome_learner
        )
        classification = self._resolve_learner(
            self.reduced_treatment_learner,
            task="classification",
            fallback=self.treatment_learner,
        )
        if data.has_missing_outcome:
            reduced, diagnostics = fit_missing_outcome_reduced(
                data,
                nuisance,
                regression_learner=regression,
                classification_learner=classification,
                g_bounds=g_bounds,
                missingness_bound=(
                    self.nuisance_bound if missingness_bound is None else missingness_bound
                ),
                n_jobs=self.n_jobs,
            )
            return reduced, diagnostics, ()
        return fit_reduced(
            data,
            nuisance,
            regression_learner=regression,
            classification_learner=classification,
            g_bounds=g_bounds,
            reduction=self.reduction,
            crossfit=self.reduced_crossfit,
            families=families,
            # Off the object handed in rather than off ``self``, so that a refit inside the
            # alternation predicts at the companion designs that round's state implies --
            # `_reduction_inputs` is what writes them there.
            companion=nuisance.companion,
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
        if data.has_intermediate:
            raise NotImplementedError(
                "DRTMLE and intermediate= are not combined. Equations (9) and (10) are stated "
                "without a controlled intermediate; its mechanism factor would sit inside "
                "the reduced regressions' own definitions, not merely in the clever "
                "covariate, and no theorem read here says what it is. "
                "Fit a plain TMLE, which is derived there."
            )
        if self.guard and self.reduction == "bivariate" and data.has_missing_outcome:
            raise NotImplementedError(
                "reduction='bivariate' is the complete-outcome construction. The "
                "randomized missing-outcome theorem uses its own five reductions; use "
                "reduction='univariate' (the setting is replaced by that construction)."
            )
        # Both of these are about the *array*, not about the extra score equations, so
        # they are asked outside the guarded block: with `guard=()` the fit is bit for bit
        # a plain TMLE, and a trial's known design mechanism is exactly what such a fit
        # should divide by. The bootstrap refusal in particular has to fire at every guard
        # -- it used to sit inside the block below, so lifting the outer refusal without
        # moving it would let an unguarded replicate refit on resampled rows the
        # row-aligned array cannot be reindexed to, silently.
        if self._treatment_probabilities is not None:
            if not data.has_missing_outcome:
                raise ValueError(
                    "treatment_probabilities= is currently only used with delta=. It "
                    "replaces the treatment learner outright, and nothing read here "
                    "states a complete-data construction that reads a known design "
                    "mechanism differently from a fitted one."
                )
            if self.n_bootstrap:
                raise NotImplementedError(
                    "treatment_probabilities= and n_bootstrap= are not combined. The array "
                    "is row-aligned to the data as passed, and a replicate refits on "
                    "resampled rows it cannot be reindexed to from here -- an n-out-of-n "
                    "resample even passes the length check, so the misalignment would be "
                    "silent. Pass randomized=True instead, so each replicate estimates the "
                    "mechanism from its own rows."
                )
        if data.has_missing_outcome and self.guard:
            if data.n_arms != 2:
                raise NotImplementedError(
                    "missing-outcome DRTMLE currently supports a binary randomized treatment; "
                    "the per-arm multi-level assembly has not been certified against the "
                    "published missing-data theorem"
                )
            if not self.randomized and self._treatment_probabilities is None:
                raise NotImplementedError(
                    "DRTMLE with delta= is supported only for a randomized trial. Pass "
                    "randomized=True to estimate the treatment mechanism for chance-imbalance "
                    "adjustment, or pass treatment_probabilities= to fit(). Observational "
                    "treatment remains unsupported by the published theorem."
                )
            if set(self.guard) != {"Q", "g"}:
                raise NotImplementedError(
                    "missing-outcome DRTMLE requires guard=('Q', 'g'): Díaz & van der "
                    "Laan's algorithm jointly targets the treatment, observation and "
                    "outcome correction blocks, and no partial-guard theorem is claimed"
                )
            if self.cross_fit:
                raise NotImplementedError(
                    "the published missing-outcome DR-TMLE theorem uses Donsker conditions and "
                    "does not establish its cross-validated extension; pass cross_fit=False"
                )
            if data.is_weighted:
                raise NotImplementedError(
                    "missing-outcome DRTMLE is not certified for a weight-tilted target law; "
                    "drop weights= or fit a plain TMLE"
                )
            if self.repeats != 1:
                raise NotImplementedError(
                    "repeats= is a cross-fitting construction and is not supported by the "
                    "published missing-outcome theorem"
                )
            if self.evaluation is not None or self.reduced_crossfit != "pooled":
                raise NotImplementedError(
                    "missing-outcome DRTMLE supports the published pooled construction only; "
                    "evaluation= and nested reduced cross-fitting are not certified"
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
