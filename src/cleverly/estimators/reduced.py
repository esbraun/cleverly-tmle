r"""Reduced-dimension regressions for doubly-robust nonparametric inference.

Every interval this package reports is valid when the second-order remainder is
negligible, which needs *both* nuisances converging fast enough.  ``drtmle``
(van der Laan 2014; Benkeser, Carone, van der Laan & Gilbert 2017; Benkeser & Hejazi
2023) buys an interval that stays valid when only one of them is consistent, by
estimating the first-order part of that remainder with regressions of each nuisance's
residual on the *other* nuisance, and solving their score equations too.

This module fits those regressions.  Nothing here solves anything: the extra score
equations, the alternation they need and the influence curve they change are the commits
after this one, and until they land a fit that carried a :class:`ReducedSet` and reported
the ordinary estimates would be handing a reader a plain TMLE's number under a
doubly-robust name.  So no estimator reaches this yet.

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

**Why a residual regression is not degenerate here.**  :math:`Q_r` and :math:`g_{r,2}` are
identically zero when the nuisance they are residuals of is right -- row by row, not merely
on average.  So under an exact law with a saturated learner all of this vanishes and the
estimator reproduces ``TMLE`` exactly, which is why no ``test_influence_gateaux*`` module
can see whether it is built correctly and why the tests for this module hand it nuisances
that are wrong on purpose.  :math:`g_{r,1}` is the exception and the trap: it is a
probability, it does not vanish, and it sits in a denominator whose numerator does.  The R
source names it ``grn2`` where the paper calls it ``gr1``, and ``grn1`` where the paper
calls it ``gr2`` -- a formula transcribed from one and checked against the other is
inverted and still plausible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .._typing import BoolArray, FloatArray, Learner
from ..data.causal_data import CausalData
from ..learners._fitting import Task
from ..learners.super_learner import SuperLearnerDiagnostics
from ..utils.bounds import bound
from ._nuisance import NuisanceEstimates, cross_fit_predictions

__all__ = ["ReducedSet", "fit_reduced", "refuse_unsupported"]

#: The reductions the sources derive.  ``"univariate"`` is Benkeser et al.'s three
#: univariate regressions and ``drtmle``'s own default; ``"bivariate"`` is van der Laan
#: (2014)'s original single :math:`g_r(a|w) = P(A = a \mid \hat{\bar Q}, \hat g)` in place
#: of the pair.  Both are in scope for the variant; only the first is written.
REDUCTIONS = ("univariate", "bivariate")


def refuse_unsupported(kind: str, detail: str = "") -> None:
    """Raise for a reduction or a fit this module will not fake.

    Kept here rather than at the caller, and worded to say what the estimator would
    *need*: "not implemented" invites the reader to assume the gap is effort, and for one
    of these two it is not.
    """
    if kind == "bivariate":
        raise NotImplementedError(
            "reduction='bivariate' is not written yet. It replaces the two univariate "
            "reduced mechanisms with van der Laan (2014)'s single bivariate "
            "gr(a|w) = P(A = a | Qbar-hat(a, W), g-hat(a|W)), which is a two-column design "
            "and a different extra score equation rather than a wider loop over the one "
            "here. The derivation is settled and the work is transcription. Use "
            "reduction='univariate', which is Benkeser et al.'s replacement for it and "
            "drtmle's own default."
        )
    if kind == "multi_arm":
        raise NotImplementedError(
            f"the reduced-dimension regressions are derived for a binary treatment; {detail} "
            "van der Laan (2014) states its problem for a binary treatment and no theorem "
            "read here covers K arms, so an implementation that accepted one would be "
            "reporting a parameter nothing has derived -- and the per-arm mechanism tilts "
            "do not renormalise, so the targeted g*(.|W) at K arms is not a distribution "
            "over the arms, which is exactly the sort of thing a theorem would settle and "
            "an example would not. Fit a plain TMLE, which is derived at K arms."
        )
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
    r"""The three reduced-dimension regressions, evaluated at every row and every arm.

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
        it into ``[0, 1]`` would floor every negative value at zero and return a
        perfectly plausible array.
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


def fit_reduced(
    data: CausalData,
    nuisance: NuisanceEstimates,
    *,
    regression_learner: Learner,
    classification_learner: Learner,
    g_bounds: tuple[float, float],
    reduction: str = "univariate",
    n_jobs: int = 1,
) -> tuple[ReducedSet, dict[str, list[SuperLearnerDiagnostics]]]:
    r"""Fit the three reduced-dimension regressions out of fold, one set per arm.

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

    Returns
    -------
    The evaluated set, and the Super Learner diagnostics keyed ``"qr"``, ``"gr1"`` and
    ``"gr2"``.

    Notes
    -----
    **Which folds.**  ``nuisance.folds``, the same split the primary nuisances used, and
    the argument is not the one the roadmap first wrote down.  A reduced regression's
    *design* is an out-of-fold prediction, so with the split reused, fold ``k``'s reduced
    regression trains on rows ``j`` outside ``k`` whose design :math:`\hat g(W_j)` came
    from a model that *did* see fold ``k`` -- and row ``i``'s own data therefore reaches
    its own prediction through the other rows' design values.  That is
    :mod:`tests.unit.test_crossfit_leakage`'s dependence, arriving through the design
    matrix rather than through the target.

    Drawing an independent split for these regressions removes **none** of it: the
    contamination is in the design values, not in which rows are trained on, so a second
    split changes nothing and loses the alignment with the fits it is a reduction of.
    The one construction that does remove it is per-fold designs -- predict
    :math:`\hat g^{(-k)}` at *every* row, so fold ``k``'s reduced regression only ever
    sees designs from the model that excluded fold ``k``.  It needs no extra fits, since
    :func:`~cleverly.estimators._nuisance.cross_fit_predictions` already builds that
    model and keeps only its test-fold slice.  What it costs is worse than what it buys:
    the training designs would be that model's *in-sample* predictions and the test
    design its out-of-sample one, and a reduced regression is a regression **of** the
    design -- so it trades a second-order dependence for a first-order covariate shift.
    So the split is reused, which is also what ``drtmle`` does, and ``groups`` is
    forwarded so that the claim ``test_crossfit_leakage`` actually states -- a model must
    not train on rows standing in for the ones it predicts -- holds at the level it is
    stated.

    **One bound is chosen here rather than at targeting time**, and it is the only one in
    this package that is.  :math:`g_{r,2}`'s *target* is a quotient by the mechanism, so
    it cannot be left raw and re-truncated later the way
    :meth:`~cleverly.estimators._nuisance.NuisanceEstimates.bounded_propensity` and
    :meth:`~cleverly.estimators._nuisance.NuisanceEstimates.bounded_missingness` are: the
    array *is* a regression of that quotient.  The fit's own declared ``g_bounds`` is
    what it divides by, and it is recorded on :attr:`ReducedSet.g_bounds`.  The
    consequence has to be said where a reader meets it rather than discovered:
    :meth:`~cleverly.sensitivity.SensitivityAnalysis.truncation_curve` moves the clever
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
    if reduction != "univariate":
        refuse_unsupported(reduction)
    if data.is_continuous_treatment:
        refuse_unsupported("continuous")
    if data.n_arms != 2:
        # Read off the data rather than off ``nuisance.arms``, so the message can name the
        # levels the caller passed rather than the internal codes -- which is what every
        # error message here reports in.
        refuse_unsupported(
            "multi_arm",
            f"{data.treatment_name} has {data.n_arms} levels {list(data.treatment_levels)}.",
        )
    arms = nuisance.arms

    scaled = nuisance.scaler.scale(data.outcome)
    truncated = nuisance.bounded_propensity(g_bounds)
    diagnostics: dict[str, list[SuperLearnerDiagnostics]] = {"qr": [], "gr1": [], "gr2": []}
    columns: dict[str, list[FloatArray]] = {"qr": [], "gr1": [], "gr2": []}

    for arm in arms:
        column = nuisance.propensity.column_for(arm)
        mechanism = nuisance.propensity.arm(arm)
        regression = nuisance.outcome.arms[arm]
        indicator = (np.asarray(data.treatment, dtype=float) == float(arm)).astype(float)

        # Qr: the outcome residual on the estimated mechanism, fitted on the rows that
        # *took* this arm.  The mask is the whole of the `| A = a` in the definition --
        # it is what makes the pooled value weighted by P(W) g_0(a | W) rather than by
        # P(W) -- and it is invisible wherever the design takes a distinct value in every
        # covariate cell, since each group is then a singleton and the weight cancels.
        # `tests/unit/test_reduced_regressions.py` pins it structurally for that reason.
        qr = _reduced_column(
            regression_learner,
            design=mechanism,
            target=scaled - regression,
            fit_mask=(indicator == 1.0) & np.asarray(data.observed, dtype=bool),
            data=data,
            nuisance=nuisance,
            task="regression",
            clip=None,
            n_jobs=n_jobs,
            diagnostics=diagnostics["qr"],
        )

        # gr1: P(A = a | Qbar-hat).  Every row, including one whose outcome is missing:
        # A and W are recorded whatever happens to Y, which is the same reason `delta=`
        # leaves an incremental fit's dm/dg term untouched.  A probability, so clipped.
        gr1 = _reduced_column(
            classification_learner,
            design=regression,
            target=indicator,
            fit_mask=None,
            data=data,
            nuisance=nuisance,
            task="classification",
            clip=(0.0, 1.0),
            n_jobs=n_jobs,
            diagnostics=diagnostics["gr1"],
        )

        # gr2: the mechanism residual in inverse-probability form, on Qbar-hat.  Signed,
        # so no clip; and its target is the one quotient formed at fit time.
        gr2 = _reduced_column(
            regression_learner,
            design=regression,
            target=(indicator - truncated[:, column]) / truncated[:, column],
            fit_mask=None,
            data=data,
            nuisance=nuisance,
            task="regression",
            clip=None,
            n_jobs=n_jobs,
            diagnostics=diagnostics["gr2"],
        )

        for name, values in (("qr", qr), ("gr1", gr1), ("gr2", gr2)):
            columns[name].append(values)

    return (
        ReducedSet(
            qr=np.column_stack(columns["qr"]),
            gr1=np.column_stack(columns["gr1"]),
            gr2=np.column_stack(columns["gr2"]),
            arms=arms,
            g_bounds=(float(g_bounds[0]), float(g_bounds[1])),
            reduction=reduction,
        ),
        diagnostics,
    )


def _reduced_column(
    learner: Learner,
    *,
    design: FloatArray,
    target: FloatArray,
    fit_mask: BoolArray | None,
    data: CausalData,
    nuisance: NuisanceEstimates,
    task: Task,
    clip: tuple[float, float] | None,
    n_jobs: int,
    diagnostics: list[SuperLearnerDiagnostics],
) -> FloatArray:
    """One reduced regression, out of fold, on a one-column design.

    The design is a nuisance prediction rather than a covariate, which is the whole of
    what makes these *reduced*: the regression is univariate however many covariates the
    fit adjusted for.
    """
    matrix = np.asarray(design, dtype=float).reshape(-1, 1)
    predictions, fitted = cross_fit_predictions(
        learner,
        matrix,
        np.asarray(target, dtype=float),
        data.weights,
        nuisance.folds,
        task=task,
        predict_designs={"values": matrix},
        fit_mask=fit_mask,
        groups=data.cluster,
        clip=clip,
        n_jobs=n_jobs,
    )
    diagnostics.extend(fitted)
    return predictions["values"]
