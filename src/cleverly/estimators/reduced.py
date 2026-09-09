r"""Reduced-dimension regressions for doubly-robust nonparametric inference.

Every interval this package reports is valid when the second-order remainder is
negligible, which needs *both* nuisances converging fast enough.  ``drtmle``
(van der Laan 2014; Benkeser, Carone, van der Laan & Gilbert 2017; Benkeser & Hejazi
2023) buys an interval that stays valid when only one of them is consistent, by
estimating the first-order part of that remainder with regressions of each nuisance's
residual on the *other* nuisance, and solving their score equations too.

This module fits those regressions and nothing here solves anything: the extra score
equations, the alternation they need and the influence curve they change live in
:mod:`cleverly.estimators.targeting` and :mod:`cleverly.inference.influence`, and
:class:`~cleverly.DRTMLE` is what assembles the three.  (This paragraph once ended *"so no
estimator reaches this yet"*, which was true for one commit and has not been since.)

**These three are the only nuisances a learner fits in the validation study**, whose two
primaries are injected analytic sequences -- and their consistency is the theorem premise
that study did not establish.  ``reduced_outcome_learner=`` and
``reduced_treatment_learner=`` are therefore not a tuning detail.  Under
``docs/technical-reference/dr-tmle/``, ``nuisance-conditions.md`` states the conditions
they have to meet, and ``validation-programme.md`` records that the reference study for
the reduced
regressions did not establish them.

Written in this package's notation, with :math:`1_a = 1\{A = a\}` and both reduced
mechanisms defined relative to a *given* :math:`\hat{\bar Q}` and :math:`\hat g`,

.. math::

    Q_r(a, w)    &= E[\, Y - \hat{\bar Q}(a, W) \mid A = a,\ \hat g(a|W) = \hat g(a|w) ] \\
    g_{r,1}(a|w) &= P(\, A = a \mid \hat{\bar Q}(a, W) = \hat{\bar Q}(a, w) ) \\
    g_{r,2}(a|w) &= E[\, \{1_a - \hat g(a|W)\}/\hat g(a|W)
                        \mid \hat{\bar Q}(a, W) = \hat{\bar Q}(a, w) ]

Each is univariate: a regression on one column, that column being the other nuisance's
out-of-fold prediction.  That is the whole construction, and
:mod:`tests.unit.test_remainder_drtmle` is where the arithmetic it has to satisfy is
pinned -- what survives a solved score equation is a **product** of the reduced
regression's error with a primary nuisance error, which is why a univariate rate suffices
however badly the primary nuisances do.

Randomized missing-outcome inference uses a different five-regression state, implemented by
:func:`fit_missing_outcome_reduced`.  It deliberately keeps the treatment and observation
mechanisms separate, as the Díaz & van der Laan (2017) targeting algorithm does.

**Why a residual regression is not degenerate here.**  :math:`Q_r` and :math:`g_{r,2}` are
identically zero when the nuisance they are residuals of is right -- row by row, not merely
on average.  So under an exact law with a saturated learner all of this vanishes and the
estimator reproduces ``TMLE`` exactly, which is why no ``test_influence_gateaux*`` module
can see whether it is built correctly and why the tests for this module hand it nuisances
that are wrong on purpose.  :math:`g_{r,1}` is the exception and the trap: it is a
probability, it does not vanish, and it sits in a denominator whose numerator does.  The R
source names it ``grn2`` where the paper calls it ``gr1``, and ``grn1`` where the paper
calls it ``gr2``.  So an *equation* read out of one and checked against the other is
inverted and still plausible.  The names here follow the paper.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np

from .._typing import BoolArray, FloatArray, IntArray, Learner
from ..data.causal_data import CausalData
from ..fluctuation.iterative import InitialFit
from ..learners._fitting import Task, predict_mean
from ..learners.super_learner import SuperLearnerDiagnostics
from ..utils.bounds import bound
from ..utils.parallel import map_parallel
from ._nuisance import (
    CompanionEstimates,
    InnerDesigns,
    NuisanceEstimates,
    Propensity,
    cross_fit_companion,
    fit_on_rows,
)

__all__ = [
    "REDUCED_CROSSFITS",
    "MissingOutcomeReducedSet",
    "ReducedFamily",
    "ReducedSet",
    "fit_missing_outcome_reduced",
    "fit_reduced",
    "reduced_designs",
    "refuse_unsupported",
]

#: The reductions the sources derive.  ``"univariate"`` is Benkeser et al.'s three
#: univariate regressions and ``drtmle``'s default; ``"bivariate"`` is van der Laan
#: (2014)'s original :math:`g_r(a|w) = P(A = a \mid \hat{\bar Q}, \hat g)` in place of
#: the pair of reduced mechanisms.
REDUCTIONS = ("univariate", "bivariate")

#: How fold ``k``'s reduced regression gets its **training** rows' design and target.
#: ``"pooled"`` reuses the primary split as it stands and is what ships; ``"nested"`` takes
#: them from models that left fold ``k`` out as well, which is the reference construction
#: The cross-fitting diagnostic measures the first against the second. A diagnostic keyword rather
#: than a tuning one, exactly as :data:`~cleverly.estimators.drtmle.UPDATE_ORDERS` is: what
#: is in question is whether the cheap construction's induced dependence is higher order,
#: and the expensive one exists so that is a run rather than an argument.
REDUCED_CROSSFITS = ("pooled", "nested")

#: One complete-outcome reduced-regression family.  A refit inside the alternation asks
#: for only the family its next score equation consumes; an initial fit asks for all of
#: them.  Kept separate from ``REDUCTIONS`` because that name selects the univariate or
#: bivariate construction, not one regression within it.
ReducedFamily = Literal["qr", "gr1", "gr2"]


@dataclass(frozen=True)
class ReducedFamilySpec:
    """Declarative learner and sampling contract for one reduced regression."""

    name: ReducedFamily
    learner: Literal["outcome", "treatment"]
    task: Task
    clip: tuple[float, float] | None
    observed_arm_only: bool = False


REDUCED_FAMILY_SPECS: tuple[ReducedFamilySpec, ...] = (
    ReducedFamilySpec("qr", "outcome", "regression", None, observed_arm_only=True),
    ReducedFamilySpec("gr1", "treatment", "classification", (0.0, 1.0)),
    ReducedFamilySpec("gr2", "outcome", "regression", None),
)


def _replace_families(base: ReducedSet, fresh: dict[ReducedFamily, FloatArray]) -> ReducedSet:
    """Carry omitted families without hiding their types behind dynamic keywords."""
    return replace(
        base,
        qr=fresh.get("qr", base.qr),
        gr1=fresh.get("gr1", base.gr1),
        gr2=fresh.get("gr2", base.gr2),
    )


def refuse_unsupported(kind: str, detail: str = "") -> None:
    """Raise for a reduction or a fit this module will not fake.

    Kept here rather than at the caller for reductions whose derivation does not exist.
    """
    if kind == "continuous":
        raise NotImplementedError(
            "the reduced-dimension regressions read a per-arm mechanism g(a | W), and a "
            "continuous dose has no arms to index by: the mechanism is a conditional "
            "density and the reduction would have to condition on it rather than on a "
            "probability. That is a different derivation, not a wider loop."
        )
    raise ValueError(f"unknown refusal {kind!r}")


@dataclass(frozen=True)
class ReducedSet:
    r"""The reduced-dimension regressions, evaluated at every row and every arm.

    Holds arrays and no callables, for the reason
    :class:`~cleverly.interventions.IPSISet` does: everything reached through
    :meth:`~cleverly.estimators.TMLE.retarget` -- the truncation curve, the MNAR tilt, a
    result read back from disk -- must target what the fit declared without any learner
    being refit.

    Attributes
    ----------
    qr:
        ``(n, K)``, :math:`Q_r(a, W_i)` with columns in :attr:`arms` order.  On the
        ``[0, 1]`` **scaled** outcome, because it is a residual of
        :class:`~cleverly.fluctuation.iterative.InitialFit`, which is.
    gr1:
        ``(n, K)``, :math:`g_{r,1}(a | W_i)`.  A probability, and the only array here
        with a denominator role -- stored **untruncated** and bounded at read time by
        :meth:`bounded_gr1`, exactly as :class:`~cleverly.estimators._nuisance.Propensity`
        is and for the same reason: a sensitivity sweep must be able to re-truncate
        without refitting.
    gr2:
        ``(n, K)``, :math:`g_{r,2}(a | W_i)`.  Signed, and not a probability -- clipping
        it into ``[0, 1]`` would floor every negative value at zero and return a perfectly
        plausible array.  This regression exists only for ``reduction="univariate"``;
        the bivariate construction stores ``NaN`` here so accidental use fails loudly.
    arms:
        The arm codes every column above is keyed by, in column order.  The same tuple
        :attr:`~cleverly.estimators._nuisance.NuisanceEstimates.arms` reports.
    g_bounds:
        The mechanism truncation :attr:`gr2`'s *target* was formed at.  On record because
        it is the one bound in this package chosen at fit time rather than at targeting
        time -- see :func:`fit_reduced` -- so a reader of a truncation curve has to be
        able to find out that the sweep did not reach these arrays.
    reduction:
        Which of :data:`REDUCTIONS` produced them.
    """

    qr: FloatArray
    gr1: FloatArray
    gr2: FloatArray
    arms: tuple[float, ...]
    g_bounds: tuple[float, float]
    reduction: str = "univariate"

    def __post_init__(self) -> None:
        if self.reduction not in REDUCTIONS:
            raise ValueError(f"reduction must be one of {list(REDUCTIONS)}; got {self.reduction!r}")
        k = len(self.arms)
        shapes = {
            name: np.asarray(getattr(self, name), dtype=float).shape
            for name in ("qr", "gr1", "gr2")
        }
        for name, shape in shapes.items():
            if len(shape) != 2 or shape[1] != k:
                raise ValueError(
                    f"{name} must be (n, {k}) for arms {list(self.arms)}; got shape {shape}"
                )
        if len({shape[0] for shape in shapes.values()}) != 1:
            raise ValueError(f"the three reduced regressions disagree about n: {shapes}")
        if not np.all(np.isfinite(self.qr)) or not np.all(np.isfinite(self.gr1)):
            raise ValueError("qr and gr1 must contain only finite values")
        gr2 = np.asarray(self.gr2, dtype=float)
        if self.reduction == "univariate" and not np.all(np.isfinite(gr2)):
            raise ValueError("gr2 must be finite for reduction='univariate'")
        if self.reduction == "bivariate" and not np.all(np.isnan(gr2)):
            raise ValueError("gr2 is absent for reduction='bivariate' and must be stored as NaN")

    @property
    def n(self) -> int:
        return int(np.asarray(self.qr).shape[0])

    def column_for(self, arm: float) -> int:
        """Index of the column holding the regressions for ``arm``."""
        match = [j for j, level in enumerate(self.arms) if level == float(arm)]
        if not match:
            raise KeyError(f"arm {float(arm)!r} is not one of {list(self.arms)}")
        return match[0]

    def bounded_gr1(self, bounds: tuple[float, float]) -> FloatArray:
        r""":attr:`gr1` truncated into ``bounds``, ``(n, K)``.

        Column by column and **not** complemented across the arms, which is where this
        departs from :meth:`~cleverly.estimators._nuisance.Propensity.bounded`'s two-arm
        rule: :math:`g_{r,1}(1|w)` and :math:`g_{r,1}(0|w)` condition on *different*
        designs -- :math:`\hat{\bar Q}(1, W)` and :math:`\hat{\bar Q}(0, W)` -- so they
        are two separate regressions rather than one probability and its complement, and
        they do not sum to one even before any truncation.
        """
        return bound(self.gr1, float(bounds[0]), float(bounds[1]))


@dataclass(frozen=True)
class MissingOutcomeReducedSet:
    r"""The five univariate regressions in Díaz & van der Laan (2017).

    Every array is ``(n, K)`` and indexed by :attr:`arms`.  ``gamma_a`` and
    ``gamma_m`` are probabilities; ``r_a``, ``r_m`` and ``e`` are signed
    conditional residual means.  The two bounds are recorded because ``r_a`` and
    ``r_m`` are fitted to targets that already divide by the primary mechanisms.
    """

    gamma_a: FloatArray
    gamma_m: FloatArray
    r_a: FloatArray
    r_m: FloatArray
    e: FloatArray
    arms: tuple[float, ...]
    g_bounds: tuple[float, float]
    missingness_bound: float
    reduction: str = "missing_outcome"

    def __post_init__(self) -> None:
        k = len(self.arms)
        shapes = {
            name: np.asarray(getattr(self, name), dtype=float).shape
            for name in ("gamma_a", "gamma_m", "r_a", "r_m", "e")
        }
        for name, shape in shapes.items():
            if len(shape) != 2 or shape[1] != k:
                raise ValueError(
                    f"{name} must be (n, {k}) for arms {list(self.arms)}; got shape {shape}"
                )
        if len({shape[0] for shape in shapes.values()}) != 1:
            raise ValueError(f"the five missing-outcome reductions disagree about n: {shapes}")

    @property
    def n(self) -> int:
        return int(np.asarray(self.e).shape[0])

    def bounded_gamma_a(self, bounds: tuple[float, float]) -> FloatArray:
        return bound(self.gamma_a, float(bounds[0]), float(bounds[1]))

    def bounded_gamma_m(self, lower: float) -> FloatArray:
        return bound(self.gamma_m, float(lower), 1.0)


def fit_missing_outcome_reduced(
    data: CausalData,
    nuisance: NuisanceEstimates,
    *,
    regression_learner: Learner,
    classification_learner: Learner,
    g_bounds: tuple[float, float],
    missingness_bound: float,
    n_jobs: int = 1,
) -> tuple[MissingOutcomeReducedSet, dict[str, list[SuperLearnerDiagnostics]]]:
    r"""Fit the five reduced nuisances for a randomized trial with MAR outcomes.

    This path is deliberately separate from :func:`fit_reduced`: the paper targets
    the treatment and observation mechanisms separately.  Collapsing them into the
    event ``A=a, Delta=1`` proves an algebraic identity for two correction terms but
    does not fit the five functions or solve the three mechanism-side scores the
    published theorem assumes.
    """
    if nuisance.missingness is None:
        raise ValueError("missing-outcome reductions need a fitted observation mechanism")
    if len(nuisance.arms) != 2:
        raise ValueError("missing-outcome reductions require binary treatment")

    scaled = nuisance.scaler.scale(data.outcome)
    observed = np.asarray(data.observed, dtype=bool)
    treatment = np.asarray(data.treatment, dtype=float)
    g_a = nuisance.bounded_propensity(g_bounds)
    raw_g_a = np.asarray(nuisance.propensity.values, dtype=float)
    bounded_m = nuisance.bounded_missingness(missingness_bound)
    assert bounded_m is not None
    g_m = np.asarray(bounded_m, dtype=float)
    raw_g_m = np.asarray(nuisance.missingness, dtype=float)
    diagnostics: dict[str, list[SuperLearnerDiagnostics]] = {
        name: [] for name in ("gamma_a", "gamma_m", "r_a", "r_m", "e")
    }
    columns: dict[str, list[FloatArray]] = {name: [] for name in diagnostics}

    for j, arm in enumerate(nuisance.arms):
        indicator = (treatment == float(arm)).astype(float)
        at_arm = np.asarray(indicator == 1.0, dtype=bool)
        complete = at_arm & observed
        q_a = np.asarray(nuisance.outcome.arms[arm], dtype=float)
        joint = np.asarray(g_a[:, j] * g_m[:, j], dtype=float)
        joint_design = np.asarray(raw_g_a[:, j] * raw_g_m[:, j], dtype=float)
        roles: tuple[
            tuple[
                str,
                Learner,
                Task,
                FloatArray,
                FloatArray,
                BoolArray | None,
                tuple[float, float] | None,
            ],
            ...,
        ] = (
            ("gamma_a", classification_learner, "classification", q_a, indicator, None, (0.0, 1.0)),
            (
                "gamma_m",
                classification_learner,
                "classification",
                q_a,
                observed.astype(float),
                at_arm,
                (0.0, 1.0),
            ),
            (
                "r_a",
                regression_learner,
                "regression",
                q_a,
                (indicator - g_a[:, j]) / g_a[:, j],
                None,
                None,
            ),
            (
                "r_m",
                regression_learner,
                "regression",
                q_a,
                (observed.astype(float) - g_m[:, j]) / joint,
                at_arm,
                None,
            ),
            ("e", regression_learner, "regression", joint_design, scaled - q_a, complete, None),
        )
        for name, learner, task, design, target, fit_mask, clip in roles:
            values, _ = _reduced_column(
                learner,
                design=design,
                target=target,
                training=None,
                companion=None,
                fit_mask=fit_mask,
                data=data,
                nuisance=nuisance,
                task=task,
                clip=clip,
                n_jobs=n_jobs,
                diagnostics=diagnostics[name],
            )
            columns[name].append(values)

    return (
        MissingOutcomeReducedSet(
            gamma_a=np.column_stack(columns["gamma_a"]),
            gamma_m=np.column_stack(columns["gamma_m"]),
            r_a=np.column_stack(columns["r_a"]),
            r_m=np.column_stack(columns["r_m"]),
            e=np.column_stack(columns["e"]),
            arms=nuisance.arms,
            g_bounds=(float(g_bounds[0]), float(g_bounds[1])),
            missingness_bound=float(missingness_bound),
        ),
        diagnostics,
    )


def fit_reduced(
    data: CausalData,
    nuisance: NuisanceEstimates,
    *,
    regression_learner: Learner,
    classification_learner: Learner,
    g_bounds: tuple[float, float],
    reduction: str = "univariate",
    crossfit: str = "pooled",
    families: Sequence[ReducedFamily] | None = None,
    companion: CompanionEstimates | None = None,
    n_jobs: int = 1,
) -> tuple[ReducedSet, dict[str, list[SuperLearnerDiagnostics]], tuple[ReducedSet, ...]]:
    r"""Fit selected reduced-dimension regressions out of fold, one set per arm.

    Parameters
    ----------
    nuisance:
        The primary fits the reductions are taken *relative to*.  Passing the whole
        object rather than two arrays is what keeps "the mechanism, the outcome
        regression and the split all came from one fit" structural: this reads
        ``nuisance.folds``, so it cannot be handed a mechanism and a split that did not
        come from the same construction.  A shift or an incremental fit builds its
        derived arrays *inside* :func:`~cleverly.estimators._nuisance.fit_nuisances` for
        that same invariant; this is built outside, because it belongs to one variant
        rather than to every fit.
    regression_learner, classification_learner:
        Already resolved, as :func:`~cleverly.estimators._nuisance.fit_nuisances` takes
        them.  Two rather than one because the tasks differ and an estimator is one or
        the other: :math:`g_{r,1}` is a conditional probability and the other two are
        conditional means of a signed quantity.
    g_bounds:
        The truncation :math:`g_{r,2}`'s target divides by -- see below.
    crossfit:
        One of :data:`REDUCED_CROSSFITS`.  ``"pooled"`` reuses the primary split as it
        stands and is what ships; ``"nested"`` reads fold ``k``'s training designs and
        targets off :attr:`~cleverly.estimators._nuisance.NuisanceEstimates.inner`, whose
        models left fold ``k`` out as well.  A reference construction rather than a
        production path -- see the last two paragraphs of the notes.
    families:
        Which regression families to fit. ``None`` fits every family in the selected
        construction and is the initial-fit behavior. A proper subset is a refit inside
        the alternation: omitted arrays are carried from ``nuisance.reduced`` (and from
        ``companion.reduced`` at evaluation rows), so the returned objects remain complete
        :class:`ReducedSet` values rather than partial objects with placeholder arrays.

    companion:
        The fit's primary nuisances at an independent draw, one copy per outer fold.  When
        given, fold ``k``'s reduced model also predicts at the design **fold ``k``'s own**
        companion primary arrays imply -- which is the pairing that makes the returned sets
        fold-conditional functions rather than a mixture of folds.  ``None`` on every fit
        that declared no ``evaluation=``, and that path is bit for bit what it was.

    Returns
    -------
    The evaluated set, the Super Learner diagnostics keyed by the families fitted, and one
    companion set per outer fold -- empty without a companion.

    Notes
    -----
    **Which folds.**  ``nuisance.folds``, the same split the primary nuisances used, and
    the argument for it is subtler than the obvious one.  A reduced regression's
    *design* is an out-of-fold prediction, so with the split reused, fold ``k``'s reduced
    regression trains on rows ``j`` outside ``k`` whose design :math:`\hat g(W_j)` came
    from a model that *did* see fold ``k`` -- and row ``i``'s own data therefore reaches
    its own prediction through the other rows' design values.  That is
    :mod:`tests.unit.test_crossfit_leakage`'s dependence, arriving through the design
    matrix rather than through the target.

    **Through the target as well**, which this paragraph used to leave out and which
    earlier descriptions of the construction left out after it. :math:`Q_r`'s target is a
    residual of
    :math:`\hat{\bar Q}` and :math:`g_{r,2}`'s is a quotient by :math:`\hat g`, so both
    halves of two of these three regressions are generated regressors.  Only
    :math:`g_{r,1}`'s target -- the arm indicator -- is data.  A construction that replaced
    the designs and left the targets alone would remove half the dependence and report
    itself as having removed it all; :func:`_roles` builds design and target off one pair of
    primary arrays so that cannot happen quietly.

    Drawing an independent split for these regressions removes **none** of it: the
    contamination is in what the training rows carry, not in which rows are trained on, so
    a second split changes nothing and loses the alignment with the fits it is a reduction
    of.  Per-fold designs -- predict :math:`\hat g^{(-k)}` at *every* row -- do remove it
    and cost more than they buy: the training designs would be that model's *in-sample*
    predictions and the test design its out-of-sample one, and a reduced regression is a
    regression **of** the design, so it trades a second-order dependence for a first-order
    covariate shift.  So the split is reused, which is also what ``drtmle`` does, and
    ``groups`` is forwarded so that the claim ``test_crossfit_leakage`` actually states --
    a model must not train on rows standing in for the ones it predicts -- holds at the
    level it is stated.

    **And here is the argument that the reuse is second order, which this docstring owed
    and did not have.**  Split fold ``k``'s empirical-process term into what the *nested*
    construction contributes and a residual :math:`(P_n - P_0)\Delta_k`, where
    :math:`\Delta_k` is the difference between the two.  The first is conditionally mean
    zero by the ordinary cross-fitting argument.  The second needs asymptotic
    equicontinuity, and the structural fact that supplies it is the one
    :func:`_reduced_column` opens with: **the reduction has fixed dimension** -- one for the
    univariate construction and at most two for the bivariate one.  Composing with a
    conditionally fixed primary fit transports brackets exactly, so the entropy requirement
    falls on a class of functions of one or two scalars and not on the primary nuisances'
    complexity at all -- and a fixed-dimension sieve or a bounded-variation ball satisfies it
    under *every* measure, which is what the random pushforward needs.  ``mean``, ``glm``,
    ``glmnet``, ``gam`` and ``boost`` are inside it; ``forest`` is not, because its fits have
    :math:`O(n)` pieces.

    What the argument does **not** settle is that :math:`\|\Delta_k\| \to 0`, which needs
    the fit to move continuously with its design column -- free for a fixed-basis smoother
    and not free for anything choosing a split point from the data.  That is why ``crossfit``
    exists: :math:`\Delta_k` *is* the pooled-minus-nested difference, so the open condition
    of the argument is the quantity the reference construction computes.
    ``docs/technical-reference/dr-tmle/targeting.md`` carries the argument in full, with
    both of its conditions and which learners meet them.

    **On the univariate construction, one bound is chosen here rather than at targeting
    time**, and it is the only one in
    this package that is.  :math:`g_{r,2}`'s *target* is a quotient by the mechanism, so
    it cannot be left raw and re-truncated later the way
    :meth:`~cleverly.estimators._nuisance.NuisanceEstimates.bounded_propensity` and
    :meth:`~cleverly.estimators._nuisance.NuisanceEstimates.bounded_missingness` are: the
    array *is* a regression of that quotient.  The fit's own declared ``g_bounds`` is
    what it divides by, and it is recorded on :attr:`ReducedSet.g_bounds`.  The
    consequence has to be said where a reader meets it rather than discovered:
    :meth:`~cleverly.assessment.DiagnosticsFacade.truncation_curve` moves the clever
    covariate's denominator and **does not** move these arrays, so the part of the curve
    that comes from the extra equations is flat by construction.  Flat-by-construction
    reads as insensitivity rather than as a limitation, which is exactly the mistake
    ``bounded_missingness`` exists to avoid on the shift axis.

    The *design* is the untruncated :math:`\hat g`, not the bounded one.  Truncation is
    for denominators, and here the mechanism is a conditioning variable: bounding it
    would collapse the extreme rows into ties and coarsen the sigma-algebra the reduction
    projects onto, which is the one thing that decides how much of the remainder the
    score equation removes.
    """
    if reduction not in REDUCTIONS:
        raise ValueError(f"reduction must be one of {list(REDUCTIONS)}; got {reduction!r}")
    if data.is_continuous_treatment:
        refuse_unsupported("continuous")
    if crossfit not in REDUCED_CROSSFITS:
        raise ValueError(f"crossfit must be one of {list(REDUCED_CROSSFITS)}; got {crossfit!r}")
    available: tuple[ReducedFamily, ...] = (
        ("qr", "gr1", "gr2") if reduction == "univariate" else ("qr", "gr1")
    )
    selected = available if families is None else tuple(families)
    if not selected:
        raise ValueError("families must name at least one reduced-regression family")
    unknown = [name for name in selected if name not in available]
    if unknown:
        raise ValueError(
            f"families must be drawn from {list(available)} for reduction={reduction!r}; "
            f"got {unknown}"
        )
    if len(set(selected)) != len(selected):
        raise ValueError(f"families names a reduced-regression family twice: {list(selected)}")
    partial = set(selected) != set(available)
    base = nuisance.reduced if isinstance(nuisance.reduced, ReducedSet) else None
    if partial and (base is None or base.reduction != reduction):
        raise ValueError(
            "a partial reduced-regression refit needs nuisance.reduced from the same "
            "construction so omitted families can be carried forward"
        )
    inner = nuisance.inner if crossfit == "nested" else None
    if crossfit == "nested":
        if inner is None:
            raise ValueError(
                "crossfit='nested' needs the fold-free primary designs on "
                "NuisanceEstimates.inner, and this object carries none. They are built by "
                "fit_inner_designs at the initial fit; a refit that dropped them would be "
                "pooled while reporting itself as nested."
            )
        if inner.n_folds != nuisance.folds.n_folds:
            # The designs are keyed to *a* split -- entry `k` is what left outer fold `k`
            # out -- so a mismatch means they were built against a different one, and every
            # fold would then train on arrays nested inside somebody else's partition. It
            # cannot happen through `DRTMLE`, where one `_nuisances` call builds both; it can
            # through `fit_reduced` directly, and a wrong answer here would look entirely
            # ordinary.
            raise ValueError(
                f"the fold-free designs cover {inner.n_folds} outer folds and this fit's "
                f"split has {nuisance.folds.n_folds}; they were built against a different "
                "split and reusing them would nest each fold inside the wrong partition"
            )
    arms = nuisance.arms

    if companion is not None and companion.n_folds != nuisance.folds.n_folds:
        raise ValueError(
            f"the companion covers {companion.n_folds} outer folds and this fit's split has "
            f"{nuisance.folds.n_folds}; fold k's reduced model predicts at fold k's own "
            "companion design, so a mismatch would pair a model with another fold's arrays"
        )
    if partial and companion is not None and len(companion.reduced) != companion.n_folds:
        raise ValueError(
            "a partial reduced-regression refit at evaluation rows needs one current "
            "companion reduced set per outer fold so omitted families can be carried forward"
        )
    scaled = nuisance.scaler.scale(data.outcome)
    names = selected
    diagnostics: dict[str, list[SuperLearnerDiagnostics]] = {name: [] for name in names}
    columns: dict[str, list[FloatArray]] = {name: [] for name in names}
    # ``(K, m)`` per role per arm, assembled into one ``ReducedSet`` per fold at the end.
    companion_columns: dict[str, list[FloatArray]] = {name: [] for name in names}

    for arm in arms:
        indicator = (np.asarray(data.treatment, dtype=float) == float(arm)).astype(float)
        production = _roles(
            nuisance,
            arm,
            scaled=scaled,
            indicator=indicator,
            g_bounds=g_bounds,
            reduction=reduction,
        )
        training = (
            None
            if inner is None
            else [
                _roles(
                    nuisance,
                    arm,
                    scaled=scaled,
                    indicator=indicator,
                    g_bounds=g_bounds,
                    reduction=reduction,
                    inner=inner,
                    fold=fold,
                )
                for fold in range(inner.n_folds)
            ]
        )

        # Qr: the outcome residual on the estimated mechanism, fitted on the rows that
        # *took* this arm.  The mask is the whole of the `| A = a` in the definition --
        # it is what makes the pooled value weighted by P(W) g_0(a | W) rather than by
        # P(W) -- and it is invisible wherever the design takes a distinct value in every
        # covariate cell, since each group is then a singleton and the weight cancels.
        # `tests/unit/test_reduced_regressions.py` pins it structurally for that reason.
        #
        # gr1: P(A = a | Qbar-hat).  Every row, including one whose outcome is missing:
        # A and W are recorded whatever happens to Y, which is the same reason `delta=`
        # leaves an incremental fit's dm/dg term untouched.  A probability, so clipped.
        #
        # gr2: the mechanism residual in inverse-probability form, on Qbar-hat.  Signed,
        # so no clip; and its target is the one quotient formed at fit time.
        specs = tuple(
            spec
            for spec in REDUCED_FAMILY_SPECS
            if spec.name in names and not (spec.name == "gr2" and reduction == "bivariate")
        )
        elsewhere = (
            None
            if companion is None
            else [
                reduced_designs(
                    companion.propensity[fold], companion.outcome[fold], arm, reduction=reduction
                )
                for fold in range(companion.n_folds)
            ]
        )
        for spec in specs:
            name = spec.name
            learner = regression_learner if spec.learner == "outcome" else classification_learner
            design, target = production[name]
            fit_mask = (
                (indicator == 1.0) & np.asarray(data.observed, dtype=bool)
                if spec.observed_arm_only
                else None
            )
            values, at_companion = _reduced_column(
                learner,
                design=design,
                target=target,
                training=None if training is None else [each[name] for each in training],
                companion=None if elsewhere is None else [each[name] for each in elsewhere],
                fit_mask=fit_mask,
                data=data,
                nuisance=nuisance,
                task=spec.task,
                clip=spec.clip,
                n_jobs=n_jobs,
                diagnostics=diagnostics[name],
            )
            columns[name].append(values)
            if at_companion is not None:
                companion_columns[name].append(at_companion)

    at_folds: tuple[ReducedSet, ...] = ()
    if companion is not None:
        companions: list[ReducedSet] = []
        for fold in range(companion.n_folds):
            fresh = {
                name: np.column_stack([column[fold] for column in companion_columns[name]])
                for name in names
            }
            if partial:
                companions.append(_replace_families(companion.reduced[fold], fresh))
            else:
                gr1 = fresh["gr1"]
                companions.append(
                    ReducedSet(
                        qr=fresh["qr"],
                        gr1=gr1,
                        gr2=(
                            fresh["gr2"] if reduction == "univariate" else np.full_like(gr1, np.nan)
                        ),
                        arms=arms,
                        g_bounds=(float(g_bounds[0]), float(g_bounds[1])),
                        reduction=reduction,
                    )
                )
        at_folds = tuple(companions)

    fresh = {name: np.column_stack(columns[name]) for name in names}
    if partial:
        assert base is not None  # validated above; narrows the type for static checking
        evaluated = _replace_families(base, fresh)
    else:
        gr1 = fresh["gr1"]
        evaluated = ReducedSet(
            qr=fresh["qr"],
            gr1=gr1,
            gr2=fresh["gr2"] if reduction == "univariate" else np.full_like(gr1, np.nan),
            arms=arms,
            g_bounds=(float(g_bounds[0]), float(g_bounds[1])),
            reduction=reduction,
        )

    return evaluated, diagnostics, at_folds


def _roles(
    nuisance: NuisanceEstimates,
    arm: float,
    *,
    scaled: FloatArray,
    indicator: FloatArray,
    g_bounds: tuple[float, float],
    reduction: str,
    inner: InnerDesigns | None = None,
    fold: int = 0,
) -> dict[str, tuple[FloatArray, FloatArray]]:
    r"""The ``(design, target)`` of each reduced regression, off one pair of primary arrays.

    Written once and called twice -- at the production arrays, and at outer fold ``k``'s
    fold-free copies -- because **both halves are generated regressors**, where the obvious
    reading is that only the design is. :math:`Q_r`'s
    target is a residual of :math:`\hat{\bar Q}` and :math:`g_{r,2}`'s is a quotient by
    :math:`\hat g`, so a nested construction that replaced the designs and left the targets
    alone would have removed half of the dependence and reported itself as having removed
    it all.  Only :math:`g_{r,1}`'s target -- the arm indicator -- is data and not an
    estimate.
    """
    if inner is None:
        mechanism_fit = nuisance.propensity
        regression_fit = nuisance.outcome
    else:
        mechanism_fit, regression_fit = inner.propensity[fold], inner.outcome[fold]
    designs = reduced_designs(mechanism_fit, regression_fit, arm, reduction=reduction)
    regression = regression_fit.arms[arm]
    truncated = mechanism_fit.bounded(g_bounds)[:, mechanism_fit.column_for(arm)]
    roles = {
        "qr": (designs["qr"], scaled - regression),
        "gr1": (designs["gr1"], indicator),
    }
    if reduction == "univariate":
        roles["gr2"] = (designs["gr2"], (indicator - truncated) / truncated)
    return roles


def reduced_designs(
    propensity: Propensity,
    outcome: InitialFit,
    arm: float,
    *,
    reduction: str = "univariate",
) -> dict[str, FloatArray]:
    r"""Which primary array each reduced regression is a regression **on**, at one arm.

    :math:`Q_r` conditions on the mechanism and the two reduced mechanisms condition on the
    outcome regression -- the crossing that is the whole construction, and the one the R
    source names the other way round from the paper.  Stated once here because it is read
    in two places: :func:`_roles`, which pairs each design with its target at the fitting
    rows, and the companion, which needs the design alone at rows that have no target.
    A second statement of it is how a companion comes to answer for a different regression
    from the one it accompanies.
    """
    if reduction not in REDUCTIONS:
        raise ValueError(f"reduction must be one of {list(REDUCTIONS)}; got {reduction!r}")
    designs = {
        "qr": propensity.arm(arm),
        "gr1": (
            outcome.arms[arm]
            if reduction == "univariate"
            else np.column_stack([outcome.arms[arm], propensity.arm(arm)])
        ),
    }
    if reduction == "univariate":
        designs["gr2"] = outcome.arms[arm]
    return designs


def _reduced_column(
    learner: Learner,
    *,
    design: FloatArray,
    target: FloatArray,
    training: list[tuple[FloatArray, FloatArray]] | None,
    companion: list[FloatArray] | None,
    fit_mask: BoolArray | None,
    data: CausalData,
    nuisance: NuisanceEstimates,
    task: Task,
    clip: tuple[float, float] | None,
    n_jobs: int,
    diagnostics: list[SuperLearnerDiagnostics],
) -> tuple[FloatArray, FloatArray | None]:
    """One reduced regression, out of fold, on a one- or two-column design.

    The design is a nuisance prediction rather than a covariate, which is the whole of
    what makes these *reduced*: the regression uses one column for the univariate reduction
    and two for the bivariate reduction, however many covariates the fit adjusted for.

    ``training`` is the nested construction: one ``(design, target)`` per outer fold, taken
    from primary models that left that fold out as well, used for the rows a fold **trains**
    on while the row it **predicts** keeps the production design.  ``None`` is the pooled
    construction and goes through :func:`~cleverly.estimators._nuisance.cross_fit_companion`
    unchanged, which is what makes a pooled fit bit for bit what it was.

    ``companion`` is one design per outer fold at the evaluation rows, predicted at by that
    fold's model and returned as a ``(K, m)`` slab beside the ``(n,)`` production column.
    """
    matrix = _as_reduced_design(design)
    elsewhere = None if companion is None else [_as_reduced_design(each) for each in companion]
    if training is not None:
        return _nested_column(
            learner,
            matrix=matrix,
            training=training,
            companion=elsewhere,
            fit_mask=fit_mask,
            data=data,
            nuisance=nuisance,
            task=task,
            clip=clip,
            n_jobs=n_jobs,
            diagnostics=diagnostics,
        )
    predictions, at_companion, fitted = cross_fit_companion(
        learner,
        matrix,
        np.asarray(target, dtype=float),
        data.weights,
        nuisance.folds,
        task=task,
        predict_designs={"values": matrix},
        companion_designs={} if elsewhere is None else {"values": elsewhere},
        fit_mask=fit_mask,
        groups=data.cluster,
        clip=clip,
        n_jobs=n_jobs,
    )
    diagnostics.extend(fitted)
    return predictions["values"], (None if elsewhere is None else at_companion["values"])


def _nested_column(
    learner: Learner,
    *,
    matrix: FloatArray,
    training: list[tuple[FloatArray, FloatArray]],
    companion: list[FloatArray] | None,
    fit_mask: BoolArray | None,
    data: CausalData,
    nuisance: NuisanceEstimates,
    task: Task,
    clip: tuple[float, float] | None,
    n_jobs: int,
    diagnostics: list[SuperLearnerDiagnostics],
) -> tuple[FloatArray, FloatArray | None]:
    """One reduced regression whose training rows never saw the fold it predicts.

    A sibling of :func:`cross_fit_predictions`'s fold loop rather than a call into it, and
    the reason is one line of it: that function fits and predicts on *one* design, and the
    whole of this construction is that the two differ.  Fold ``k`` trains on
    ``training[k]`` -- designs and targets from models that left fold ``k`` out -- and
    predicts at ``matrix``, the production design, so the evaluation half is what the pooled
    construction evaluates and only the estimation half moves.

    Predicting at the *inner* design instead would be a different estimator and a silent
    one: every array stays in range, and the fit would be answering for a mechanism no row
    was assigned under.  ``tests/unit/test_drtmle_crossfit.py`` pins the call site against
    exactly that, and the longhand beside it against reading the wrong fold's copy.
    """
    folds = nuisance.folds
    n = matrix.shape[0]
    mask = np.ones(n, dtype=bool) if fit_mask is None else np.asarray(fit_mask, dtype=bool)
    weights = data.weights
    groups = data.cluster

    def clipped(values: FloatArray) -> FloatArray:
        return values if clip is None else np.clip(values, clip[0], clip[1])

    def run_fold(
        index: int, train: IntArray, test: IntArray
    ) -> tuple[int, IntArray, FloatArray, FloatArray | None, Any]:
        rows = train[mask[train]]
        if rows.size == 0:
            raise ValueError(
                "a cross-fitting fold has no trainable rows for a reduced regression; "
                "reduce n_folds or use reduced_crossfit='pooled'"
            )
        design, target = training[index]
        model = fit_on_rows(
            learner,
            _as_reduced_design(design),
            np.asarray(target, dtype=float),
            weights,
            rows,
            task,
            groups,
        )
        values = clipped(np.asarray(predict_mean(model, matrix[test], task), dtype=float))
        elsewhere = (
            None
            if companion is None
            else clipped(np.asarray(predict_mean(model, companion[index], task), dtype=float))
        )
        return index, test, values, elsewhere, getattr(model, "diagnostics_", None)

    jobs = [(index, train, test) for index, (train, test) in enumerate(folds)]
    out = np.empty(n, dtype=float)
    slabs: dict[int, FloatArray] = {}
    for index, test, values, elsewhere, fitted in map_parallel(run_fold, jobs, n_jobs=n_jobs):
        out[test] = values
        if elsewhere is not None:
            slabs[index] = elsewhere
        if fitted is not None:
            diagnostics.append(fitted)
    if companion is None:
        return out, None
    return out, np.stack([slabs[index] for index in range(folds.n_folds)])


def _as_reduced_design(values: FloatArray) -> FloatArray:
    """Normalise a reduced design without flattening the bivariate construction."""
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    if array.ndim == 2 and array.shape[1] in (1, 2):
        return array
    raise ValueError(f"a reduced design must have one or two columns; got {array.shape}")
