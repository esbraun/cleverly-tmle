r"""Item 13: the remainder Theorem 1 assumes negligible, computed rather than asserted.

``docs/roadmap.md``'s **item 13** is the theorem's condition beyond the three score
equations.  Solving equations (8), (9) and (10) is *necessary*; Theorem 1 separately assumes
that what is left over is :math:`o_p(n^{-1/2})`, and nothing in this repository has ever
measured it.  ``docs/drtmle/validation-plan.md`` §5 states the quantity:

.. math::

    R_{\text{remaining}} = \hat\psi - \psi_0 - (P_n - P_0)\hat D_{\text{DR}}

and refuses :math:`P_n\hat D` as a stand-in for :math:`P_0\hat D` **by name** -- that is the
quantity targeting drove to zero, so it answers a different question.  :math:`P_0\hat D`
needs the corrected curve as a *function* of :math:`(W, A, Y)`, evaluated where the fit did
not look, and ``DRTMLE(evaluation=...)`` is what supplies it: every fold's nuisances at an
independent draw, moved by the same targeting steps the fitted arrays took.  This module is
the arithmetic on top.

**The fold convention, which §5 requires be documented rather than discovered.**  A
cross-fitted fit has no single nuisance function -- it has :math:`K` of them, one per outer
fold -- so :math:`P_0\hat D` is the fold-conditional average

.. math::

    P_0\hat D = \sum_k \frac{n_k}{n}\; E_0\bigl[\hat D^{(k)}(O)\bigr],

with :math:`n_k` the rows fold :math:`k` holds out and the expectation taken over the
independent draw.  It is the estimator's **own** fold weighting, not a uniform one, and
:class:`~cleverly.estimators._nuisance.CompanionEstimates` carries the counts for exactly
that reason.  Without this stated, ``R_remaining`` can be an artefact of how fold-specific
fits were extrapolated rather than a property of the estimator.

**Three columns, and they are not the same kind of number.**

``R_remaining``
    The whole of what item 13 asks about, exact given the companion: no limit is
    approximated anywhere in it.

``R2``
    The *plain* second-order remainder
    :math:`P_0[(\hat g_a - g_{0,a})/\hat g_a\,(\hat Q_a - \bar Q_{0,a})]` at the **fitted**
    nuisances, which is the regime-entry column Tier 2 gets in place of Tier 1's quadrature
    over a prescribed sequence.  Also exact -- the DGP knows :math:`\bar Q_0` and
    :math:`g_0`, and the companion knows :math:`\hat Q` and :math:`\hat g` at the same rows.

``R_Q`` and ``R_g``
    The two appendix branches, **approximated**, and what is approximated in them is stated
    below rather than buried.  §5 asks for them apart because a total trending to zero can
    conceal cancellation between them, which is gate 1's clause 4.

**What the branch columns are and are not.**  The 2016 working paper's appendix A gives
:math:`R_{Q,n} = R_{3,n} + R_{4,n} + M_{1,n}` and appendix B gives
:math:`R_{g,n} = \tilde R_{5,n} + \tilde R_{6,n} + \tilde M_{2,n}`
(``docs/drtmle/theorem-concordance.md`` §5).  Two observations make the second-order halves
computable and one keeps the rest honest.

*The sums need fewer limits than the terms do.*  Writing them out,

.. math::

    R_{3,n} + R_{4,n} &= P_0\Bigl[\bigl\{\bar Q_{0n,r}/g_0 - \bar Q_{n,r}/g_n\bigr\}
                                  (g_0 - g_n)\Bigr] \\
    \tilde R_{5,n} + \tilde R_{6,n} &= P_0\Bigl[\bigl\{(1_a/g_{1,0n,r})g_{2,0n,r}
                          - (1_a/g_{1,n,r})g_{2,n,r}\bigr\}(Y - \bar Q_n)\Bigr]

-- the univariate limits :math:`\bar Q_{0,r}`, :math:`g_{1,0,r}` and :math:`g_{2,0,r}`
**cancel** out of both.  What is left is the fitted reductions, which the companion has
exactly, and the ``0n`` limits.

*A ``0n`` limit is a quadrature and not a fit.*  :math:`\bar Q_{0n,r}` is the *population*
conditional mean of a computable quantity given two computable scalars, so it is estimated
here by a binned conditional average over the evaluation draw -- accuracy controlled by the
draw size and the bin count rather than by a model choice.  Both are reported: every branch
is recomputed at a second bin count and the difference travels beside it as the column's own
error.  Where that error exceeds the branch, the branch is reported as **not resolvable at
this DGP**, which is §5's *"where the DGP permits"* said out loud.

*The empirical-process terms are refused by name.*  :math:`M_{1,n}` and
:math:`\tilde M_{2,n}` are :math:`(P_n - P_0)` of a difference of estimated curves, and
under the fold convention above :math:`P_n` and :math:`P_0` are taken at *different*
renderings of the nuisances -- out of fold on the fitting sample, fold-conditional on the
evaluation draw.  There is no single-sample expression that is both, so rather than pick one
and call it the theorem's term, this module reports the second-order halves and says so.
They are the halves clause 4 is about: an empirical-process term is
:math:`o_p(n^{-1/2})` under the Donsker and :math:`L_2` conditions §5 lists and carries no
product of nuisance errors to cancel against.

**Two rules for the evaluation draw, and the second is why item 13's column can be read at
all.**  ``docs/roadmap.md``'s **E1** exists because C3c's ``sqrt(n) R_remaining`` was flat to
within its own error: a 9--13% decline over a fourfold :math:`n` against Monte Carlo errors of
7--11%, which is a plateau and a slow decline telling the same story.  The error was the
draw's: :math:`P_0\hat D` was a plain mean over :math:`m` i.i.d. rows, so it carried
:math:`\mathrm{sd}(D)/\sqrt m` -- measured at ``0.026`` for :math:`m = 1{,}500` against a
remainder of order ``0.007`` -- and that error lands *directly* in the remainder and is then
multiplied by :math:`\sqrt n`.

:func:`quadrature_frame` is the answer and it is a **reduction of the problem, not a bigger
draw**: the curve is affine in :math:`Y` given :math:`(A, W)` and reads :math:`A` only through
the indicator, so both of those coordinates integrate in closed form and what is left is a
quadrature in :math:`W` -- taken on the Sobol rule :meth:`~cleverly.datasets.DGP.quadrature`
supplies, so :math:`\psi_0` and :math:`P_0\hat D` come off one grid.  The derivation is in that
function's docstring.  :func:`evaluation_frame` stays beside it and is not legacy: it is the
independent route the quasi-random one is checked against.

**The rule is randomised, and E1b is why that is a correction rather than a refinement.**  E1
took one fixed scramble at every replicate, which makes the rule's error a **bias** no
replicate count removes, and offered a nested convergence ladder as the thing that bounds it.
A successive difference between two rungs bounds nothing without a convergence result that
applies, and a piecewise-smooth integrand has none -- so that design traded a measured noise
for an unmeasured bias and reported the trade as settled.  An **independent scramble per
replicate** removes the problem instead of bounding it: a randomised quasi-Monte Carlo rule is
unbiased at every point count, so the error is mean-zero noise again -- far smaller than the
draw's, and averaging down over a study exactly as the draw's did.  ``scramble=`` on
:func:`quadrature_frame` and :func:`truth_at` is that, :class:`CompanionStack` is how several
replicates of a rule reach one fit, and ``benchmarks/drtmle_companion_grid.py`` is what
measures each rule's own error from replication rather than from refinement.

**What this instrument cannot see**, named here rather than left for a reader to discover,
because an instrument whose blind spots are unlisted is how lesson 9 happens again:

*The law itself.*  Every column integrates ``dgp.propensity`` and ``dgp.outcome_mean`` against
predictions of them, so an error *in* those functions is invisible -- the same objects define
the truth the remainder is measured against.  ``tests/unit/test_drtmle_remainder_study.py``'s
oracle control is silent on this too.  What covers it is the tier modules' own quadratures,
which compute a declared coefficient from the same law by different arithmetic.

*A fold weighting that is wrong symmetrically.*  :func:`_fold_average` takes the estimator's
own fold weights, and a study whose folds are equal-sized -- which every cell of the coverage
study has -- cannot distinguish them from uniform ones.  ``test_the_fold_weighting_is_the
_estimators`` pins the wiring; nothing here pins the choice.

*A branch weighting, at the oracle.*  At the ``_HUGE`` control both nuisances are correct, so
:math:`Q_r` and :math:`g_{r,2}` vanish row by row and every branch is zero whatever measure it
was integrated against.  A ``row_weights`` bug in :func:`branch_products` is therefore
invisible in the control and needs a drifted cell -- which is the same degeneracy
``docs/roadmap.md``'s item 21 turned on, in a third place.

**Nothing here asserts.**  It is an instrument, like ``benchmarks/bench_drtmle.py``: it
returns numbers a table prints and a human reads against the rules frozen in §5.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import narwhals as nw
import numpy as np

from cleverly.datasets import DGP
from cleverly.inference.influence import reduced_correction_parts
from cleverly.utils.frames import frame_from_dict

__all__ = [
    "BIN_COUNTS",
    "Block",
    "CompanionStack",
    "RemainderRow",
    "Window",
    "branch_products",
    "conditional_mean",
    "corrected_curve",
    "corrected_remainder",
    "evaluation_frame",
    "plain_remainder",
    "quadrature_frame",
    "remainder_rows",
    "stacked_companion",
    "targeted_remainder",
    "truth_at",
]

#: The two bin counts every branch is computed at.  Not a tuning knob: the pair is what says
#: whether the binned limits had **settled**, so a branch smaller than the gap between them
#: is one this design was still visibly moving on.  A power of two apart, so the coarse grid
#: is a strict coarsening of the fine one and the difference is discretisation rather than
#: reshuffling.
#:
#: **The pair is not the reported *error* of the binned limits, and it was described as one
#: here.**  A successive difference between two rungs says a sequence settled and not where
#: -- ``docs/roadmap.md``'s E1b is the same retraction for the quadrature ladder, and
#: :attr:`RemainderRow.branch_movement` carries the reasoning and the one direction of the
#: inference this pair does support.
BIN_COUNTS = (12, 24)

#: The arm each per-arm column is reported at, and the contrast taken between them.
ARMS = (1.0, 0.0)


@dataclass(frozen=True)
class Window:
    """A contiguous run of companion rows, and the unit every integral here is taken over.

    It replaces the ``limit`` a coarser grid used to be read by, and the reason is that a
    companion now holds **more than one replicate of a rule**: several independent Sobol
    scrambles, several independent i.i.d. draws, or both.  A prefix could address the first
    of those and nothing else.

    Two things follow and both are contracts rather than conveniences.  A window is
    contiguous because the rows it selects have to be *a rule's own sample* -- under
    :func:`quadrature_frame`'s interleaving the first :math:`2k` rows of a block are that
    block's grid at :math:`k` points, so a rung is a shorter window with the same ``start``
    and a scramble is a different ``start`` entirely.  And every integrator slices with one
    of these **before** it does anything else, which matters most in
    :func:`branch_products`: the binned limits take quantiles of the rows they are given, so
    a window that arrived after the binning would report a limit conditioned on rows the
    block does not contain.
    """

    start: int
    stop: int

    @classmethod
    def prefix(cls, rows: int) -> Window:
        """The first ``rows`` rows -- what ``limit=`` meant, kept for the ladder's coarse rungs."""
        return cls(0, rows)

    @property
    def rows(self) -> int:
        return self.stop - self.start

    def head(self, rows: int) -> Window:
        """The first ``rows`` rows *of this window*, which is a coarser rung of the same block."""
        return Window(self.start, self.start + rows)


@dataclass(frozen=True)
class Block:
    """One replicate of one rule inside a stacked companion.

    ``rule`` is ``"sobol"`` or ``"draw"``; ``seed`` is the scramble or the draw seed, so a
    row can say which randomisation produced it; ``points`` is the Sobol point count and is
    ``0`` on a draw, because an i.i.d. sample has rows and no grid and writing the row count
    in both fields would make a table look like a ladder it is not on.
    """

    rule: str
    seed: int
    points: int
    window: Window


@dataclass(frozen=True)
class CompanionStack:
    """Several replicates of one or both rules, in one frame, for **one** fit.

    This is what makes a conditional measurement affordable.  Estimating an evaluation
    rule's own error needs the *same fitted curve* integrated by several independent
    replicates of that rule, and the naive way to get it is a refit per replicate -- which
    would cost a fit each and, worse, would have to be argued bit-identical before the
    spread across them could be called the rule's.

    Neither is necessary.  The companion contributes to no fit, no fold and no score
    (``tests/unit/test_drtmle_companion.py`` pins that bit for bit) and every companion
    prediction is taken row by row from a model fitted on the fitting rows -- so stacking
    the replicates into one frame and reading a :class:`Window` per replicate gives exactly
    what a refit would, off one fit.  ``tests/unit/test_drtmle_remainder_study.py`` pins the
    block against a companion fitted alone, which is the stacking analogue of the prefix's
    own pin and the assertion the whole design rests on.
    """

    frame: Any
    weights: np.ndarray
    blocks: tuple[Block, ...]


@dataclass(frozen=True)
class RemainderRow:
    """One estimand's remainder columns at one fit.

    Attributes
    ----------
    estimand:
        ``"ey1"``, ``"ey0"`` or ``"ate"``.
    psi, truth:
        The fit's estimate and the law's value, both on the outcome's own scale.
    p0_curve, pn_curve:
        :math:`P_0\\hat D` under the fold convention, and :math:`P_n\\hat D` off the
        reported curve.  The second is what targeting drove to zero and is here so that a
        reader can see it did -- **not** so that it can stand in for the first.
    remaining:
        :math:`\\hat\\psi - \\psi_0 - (P_n - P_0)\\hat D`.
    root_n_remaining:
        :math:`\\sqrt n` times it, which is the quantity item 13 asks to vanish.
    r2:
        The plain second-order remainder at the fitted nuisances, at the **initial**
        regression -- the plug-in one, which says the nuisances are what the design says.
    r2_targeted:
        The same expression at the **targeted** regression, which is what the fit's bias is
        and what §5's targeted-coefficient clause requires the regime be read off.  The pair
        is the point: C3a's pilot had only the first and read it as the second.
    branch_q, branch_g:
        Appendix A's and appendix B's second-order halves, or ``nan`` where the binned
        limits had not **settled** between the two bin counts.
    branch_movement:
        The larger of the two branches' movements between :data:`BIN_COUNTS`, which is what
        "had not settled" is decided on.

        **It is not an error bound, and this field was called ``branch_error`` until it
        was.**  It is a successive difference between two rungs of a refinement, which is
        the statistic ``docs/roadmap.md``'s E1b withdrew for the quadrature ladder --
        measured there at four times *below* the true error at the finest rung and three
        orders *above* it two rungs earlier.  Nothing about the binned limits makes their
        version of it better behaved: :func:`conditional_mean` is a regressogram in a
        two-dimensional design, and its bias is a smoothing bias with no monotonicity in
        the bin count to lean on.

        **What it can carry is one direction of the inference, and the suppression above
        uses only that direction.**  A branch that moves more between 12 and 24 bins than
        its own magnitude is not a number to read -- the instrument is visibly still
        moving.  A branch that moves less may still be wrong by any amount, because a
        smoothing bias can be stable across two resolutions and large at both.  So
        settling is **necessary and not sufficient**, and a settled branch's actual error
        is *unestablished* rather than small.  What would establish it is a reference whose
        own fidelity is measured against something other than its own refinement, which is
        ``docs/roadmap.md``'s E2.
    companion_se, companion_replicate_se:
        The evaluation rule's **own** error, on the :math:`\\sqrt n` scale the column above
        is read at -- two numbers because they are arrived at two ways and one field with
        two meanings is how a column comes to be misread.

        ``companion_se`` is :math:`\\sqrt n\\,\\mathrm{sd}(\\hat D)/\\sqrt m`, the i.i.d.
        draw's error from a *formula*.  It is right for the rule it was derived under and an
        enormous overstatement of a quasi-random rule's, and it needs no replication.

        ``companion_replicate_se`` is the standard error of the mean **across the
        replicates of the rule that this row averages**, at a fixed fit: the spread of
        ``root_n_remaining`` over the windows of :class:`CompanionStack`, divided by
        :math:`\\sqrt R`.  It assumes no convergence rate and no model of a witness -- it is
        what a standard error is -- and it is ``nan`` when a row was read at a single window,
        because one replicate has no spread and a number there would be an invention.

        **What this field replaced, and why, is worth carrying here.**  It was the movement
        of :math:`P_0\\hat D` when half the rows are dropped.  That is a fair reading of a
        *noise* inflated by :math:`\\sqrt 2` and is not a bound on a deterministic grid's
        error at all: a successive difference says a sequence settled, not where.  Measured
        on this ladder's own geometry with a piecewise-smooth integrand, it ran four times
        below the true error at the finest rung and three orders above it two rungs earlier.
        ``docs/roadmap.md``'s E1b is the retraction.
    """

    estimand: str
    psi: float
    truth: float
    p0_curve: float
    pn_curve: float
    remaining: float
    root_n_remaining: float
    r2: float
    r2_targeted: float
    branch_q: float
    branch_g: float
    branch_movement: float
    companion_se: float = float("nan")
    companion_replicate_se: float = float("nan")
    replicates: int = 1


def evaluation_frame(dgp: DGP, n: int, seed: int) -> Any:
    """An independent draw from the same law, for the companion to be evaluated at.

    Drawn from a seed stream disjoint from the study's, so that raising ``--evaluation-n``
    cannot change which rows a replicate was *fitted* on -- the same prefix-stability rule
    ``benchmarks/drtmle_coverage.py`` applies to its own two streams.

    This is the **i.i.d. rule**, and it is kept beside :func:`quadrature_frame` rather than
    replaced by it: it is what the deterministic rule is read against.  Two routes to one
    population integral sharing no arithmetic is the strongest check available here, and it
    is the same argument ``plain_remainder`` is checked against Tier 1's quadrature under.
    """
    return dgp.sample(n, seed=seed)[0]


def quadrature_frame(
    dgp: DGP, points: int, *, scramble: int | None = None
) -> tuple[Any, np.ndarray]:
    r"""The quasi-random rule: the law's own Sobol grid, and the weight each row carries.

    Returns ``(frame, weights)``.  The frame is ``2 * points`` rows -- every Sobol point of
    :meth:`~cleverly.datasets.DGP.quadrature` at **both** arms -- with ``A`` set to the arm
    and ``Y`` set to :math:`\bar Q_0(a, W)`; ``weights`` is :math:`g_0(a \mid W)`.  Hand both
    to :func:`corrected_remainder` and its siblings and the integral they return is the same
    population quantity the i.i.d. rule estimates, with the Monte Carlo taken out of two of
    its three coordinates.

    The weights are **unnormalised**, and deliberately: every average in this module divides
    by their sum, so the scale is free -- and dividing by ``points`` here would make a
    coarser grid's weights differ from a prefix of a finer one's by exactly that factor,
    breaking the nesting the paragraph below is about for no gain.

    **The rows are interleaved, and that is a contract rather than a layout.**  Row
    :math:`2j` is :math:`(W_j, A = 1)` and row :math:`2j + 1` is :math:`(W_j, A = 0)`, so
    the first :math:`2k` rows are *exactly* the grid at :math:`k` points -- and because
    :meth:`~cleverly.datasets.DGP.quadrature`'s grids nest within a scramble, the first
    :math:`2 \cdot 2^i` rows are the whole ladder of coarser grids, in one companion.  A
    ladder is therefore a :class:`Window` on one fit rather than a fit per grid, which is
    what makes the movement between two rungs *the quadrature* rather than a difference
    between two fits that would have to be argued bit-identical.  Stacking the arms in two
    blocks instead would leave every prefix reading one arm only.

    **``scramble`` selects the randomisation, and it is not a refinement knob.**  ``None``
    takes the law's default -- the scramble :meth:`~cleverly.datasets.DGP.truth` is
    integrated at, which is what keeps a lone call reproducible.  Any other value is an
    *independent* randomisation of the same rule, unbiased for the same integral at the same
    point count, so several of them at a fixed fit estimate the rule's own error directly.
    :func:`truth_at` **must be given the same one**: the cancellation it documents holds only
    if :math:`\psi_0` and :math:`P_0\hat D` are one integral, and that now means one
    randomisation as well as one point count.

    **Why that is exact rather than a variance reduction with a bias.**  Every quantity this
    module integrates is *affine in* :math:`Y` *given* :math:`(A, W)` and reads :math:`A`
    **only through the indicator** :math:`1\{A = a\}`.  Reading it off the two expressions
    the number is built from: the corrected curve's own terms are
    :math:`h(A, W)\{Y - \bar Q^*(A, W)\}` and :math:`\bar Q^*(a, W) - \psi`, and
    :func:`~cleverly.inference.influence.reduced_correction_parts` builds
    :math:`D^*_g = (Q_r/g)(1_a - g)`, which carries no :math:`Y` at all, and
    :math:`D^*_Q = 1_a \cdot (g_{r,2}/g_{r,1})(Y - \bar Q^*)`, which is affine in it.  So

    .. math::

        P_0\hat D = E_W\Bigl[\sum_{a \in \{0, 1\}} g_0(a \mid W)\,
                             \hat D\bigl(W, a, \bar Q_0(a, W)\bigr)\Bigr]

    is an identity and not an approximation: the sum over :math:`A` is finite and the
    integral over :math:`Y` is closed form, because :math:`E_0[Y \mid A, W]` is
    :math:`\bar Q_0` and nothing here is nonlinear in :math:`Y`.  What is left is a
    quadrature in :math:`W` alone, and the rule taken for it is
    :meth:`~cleverly.datasets.DGP.quadrature`'s at this ``scramble`` -- so :math:`\psi_0` and
    :math:`P_0\hat D` are read off one grid and a disagreement between them stays
    attributable, which is the whole reason
    :meth:`~cleverly.datasets.DGP.expectation` exists.

    **A law with a nonlinear outcome link would break the** :math:`Y` **half of this**, and
    a treatment with more than two arms would not break it at all -- the sum would just have
    more terms.  Both cells of the study are drawn from ``linear_dgp``, whose outcome is
    additive-error gaussian, and the estimator is binary throughout; a caller who changes
    either has to redo this paragraph rather than reuse it.
    """
    _refuse_unsupported(dgp)
    latent = _grid(dgp, points, scramble)
    truth_g = np.clip(np.asarray(dgp.propensity(latent), dtype=float), 1e-9, 1.0 - 1e-9)
    stacked = np.repeat(latent, len(ARMS), axis=0)
    payload: dict[str, np.ndarray] = {
        "Y": _interleave(
            [np.asarray(dgp.outcome_mean(latent, arm, None), dtype=float) for arm in ARMS]
        ),
        "A": _interleave([np.full(points, arm, dtype=float) for arm in ARMS]),
    }
    for index, name in enumerate(dgp.covariate_names):
        payload[name] = stacked[:, index]
    weights = _interleave([_arm_probability(truth_g, arm) for arm in ARMS])
    # `frame_from_dict` is what `DGP.sample` emits through, so the deterministic frame is
    # the same object in the same backend as a drawn one -- the companion cannot tell them
    # apart, which is what keeps `DRTMLE(evaluation=...)` unaware that a rule was swapped.
    return frame_from_dict(payload), weights


def truth_at(dgp: DGP, points: int, *, scramble: int | None = None) -> dict[str, float]:
    r""":math:`\psi_0` on the **companion's own grid**, which is not a refinement detail.

    :meth:`~cleverly.datasets.DGP.truth` integrates at :math:`2^{18}` points and this
    integrates the same three functionals at ``points``.  Using it is what makes the
    deterministic rule worth having, and the reason is a cancellation rather than a taste.
    Write the remainder out with the curve's centring substituted, :math:`P_0\hat D =
    E_0[\hat D^{u}] - \hat\psi`:

    .. math::

        R_{\text{remaining}} = E_0[\hat D^{u}] - \psi_0 - P_n \hat D
                             = E_0\bigl[\hat D^{u} - \bar Q_0\bigr] - P_n \hat D .

    The second equality holds **only if both expectations are the same integral**, and it is
    what turns an :math:`O(1)` integrand into one whose every term is a product of two
    nuisance errors -- so the quadrature's relative error acts on something of order
    :math:`n^{-\alpha}` rather than on :math:`\psi_0` itself.  Take :math:`\psi_0` from a
    finer grid and the two rules are differenced instead, and the :math:`O(1)` part of the
    error survives.  It is :meth:`~cleverly.datasets.DGP.expectation`'s own argument -- one
    rule, so a disagreement is attributable -- applied to a difference rather than to a
    comparison.

    ``scramble`` is the other half of "the same integral", and it is the newer half: under a
    randomised rule two scrambles are two unbiased estimates of :math:`\psi_0` differing by
    the rule's own error, so taking the truth from one and the curve from another differences
    two randomisations and puts back exactly the :math:`O(1)` term this function exists to
    cancel.  Pass :func:`quadrature_frame` the same value.

    The three keys are :meth:`~cleverly.datasets.DGP.truth`'s, and
    ``test_drtmle_remainder_study.py`` pins them equal at :math:`2^{18}` so the duplicated
    arithmetic cannot drift from the method it mirrors.
    """
    latent = _grid(dgp, points, scramble)
    q1 = np.asarray(dgp.outcome_mean(latent, 1.0, None), dtype=float)
    q0 = np.asarray(dgp.outcome_mean(latent, 0.0, None), dtype=float)
    return {"ey1": float(np.mean(q1)), "ey0": float(np.mean(q0)), "ate": float(np.mean(q1 - q0))}


def _grid(dgp: DGP, points: int, scramble: int | None) -> np.ndarray:
    """The latent matrix at this point count and this randomisation.

    ``None`` means the law's own default, which keeps every existing call bit for bit and
    keeps a lone invocation reproducible.  One helper rather than two call sites spelling the
    conditional out, since the two that read it -- the companion and its truth -- are the
    pair that must not disagree.
    """
    return dgp.quadrature(points) if scramble is None else dgp.quadrature(points, scramble=scramble)


def stacked_companion(
    dgp: DGP,
    *,
    points: int | Sequence[int] = 0,
    scrambles: Sequence[int] = (),
    draw_rows: int = 0,
    draw_seeds: Sequence[int] = (),
) -> CompanionStack:
    r"""Several replicates of the two rules, in one frame, for one fit.

    One :class:`Block` per replicate, in the order given -- every scramble of the quasi-random
    rule first, then every i.i.d. draw -- with a :class:`Window` addressing its rows.  Hand
    ``frame`` to ``DRTMLE(evaluation=…)`` and each block's window to :func:`remainder_rows`,
    and what comes back is what a refit per replicate would have produced.

    **The weight vector covers both rules and that is not a compromise.**  A Sobol block
    carries :math:`g_0(a \mid W)` and a draw block carries ones, and ones is exactly what
    ``row_weights=None`` resolves to in :func:`_row_mass` -- so a draw block read through this
    vector is the plain average the i.i.d. rule has always taken, to the bit.  What the shared
    vector buys is that a caller cannot pair a window with the wrong rule's measure, which is
    the stale-weights mistake ``_row_mass``'s row-count check exists to catch and which a
    second vector would reintroduce.

    **Why replicates rather than a finer grid.**  Refinement estimates nothing: it says a
    sequence settled and not where, and ``docs/roadmap.md``'s E1b is the two claims that cost.
    Independent replicates of a *randomised* rule are unbiased for the same integral, so their
    spread is a standard error under no assumption about a rate -- which is what the error of
    a piecewise-smooth integrand needs, since it has no rate to appeal to.

    **``points`` may be one count or one per scramble**, and the second form is E2's rather
    than E1b's.  A study measuring an evaluation *rule* wants every block at the same
    resolution, so the blocks are replicates of one thing; a study that also has to **score** a
    candidate on rows it did not see wants that block finer than the one it was fitted on,
    which is a different resolution in the same companion.  One count is broadcast, so every
    existing call is unchanged.
    """
    counts = (
        [int(points)] * len(scrambles)
        if isinstance(points, int)
        else [int(each) for each in points]
    )
    if len(counts) != len(scrambles):
        raise ValueError(
            f"{len(scrambles)} scramble(s) and {len(counts)} point count(s); a per-block "
            "resolution has to name one count per block or one count for all of them"
        )
    if any(counts) and not scrambles:
        raise ValueError("a Sobol block needs a scramble; pass scrambles= or leave points at 0")
    if draw_rows and not draw_seeds:
        raise ValueError("a draw block needs a seed; pass draw_seeds= or leave draw_rows at 0")

    frames: list[Any] = []
    masses: list[np.ndarray] = []
    blocks: list[Block] = []
    start = 0
    for scramble, count in zip(scrambles, counts, strict=True):
        frame, weights = quadrature_frame(dgp, count, scramble=scramble)
        frames.append(frame)
        masses.append(weights)
        blocks.append(Block("sobol", scramble, count, Window(start, start + weights.size)))
        start += weights.size
    for seed in draw_seeds:
        frames.append(evaluation_frame(dgp, draw_rows, seed))
        masses.append(np.ones(draw_rows, dtype=float))
        blocks.append(Block("draw", seed, 0, Window(start, start + draw_rows)))
        start += draw_rows
    if not frames:
        raise ValueError("a companion stack needs at least one block")
    return CompanionStack(_concatenate(frames), np.concatenate(masses), tuple(blocks))


def _concatenate(frames: Sequence[Any]) -> Any:
    """One frame from several, through the backend-neutral route the rest of the module uses.

    Rebuilt column by column with :func:`~cleverly.utils.frames.frame_from_dict` rather than
    with a narwhals concat, because that is what both companion builders already emit through
    and the stack has to be indistinguishable from either -- the fit must not be able to tell
    that a rule was stacked, exactly as it cannot tell that one was swapped.
    """
    if len(frames) == 1:
        return frames[0]
    tables = [nw.from_native(frame, eager_only=True) for frame in frames]
    names = tables[0].columns
    return frame_from_dict(
        {
            name: np.concatenate([table[name].to_numpy().astype(float) for table in tables])
            for name in names
        }
    )


def _interleave(blocks: list[np.ndarray]) -> np.ndarray:
    """One row per ``(point, arm)`` pair, point-major -- :func:`quadrature_frame`'s contract."""
    return np.stack(blocks, axis=1).reshape(-1)


def _refuse_unsupported(dgp: DGP) -> None:
    """The laws a deterministic companion cannot be built for, each refused by name.

    All four are refusals rather than gaps, and each names what the derivation would need.
    The coverage study's cells are drawn from ``linear_dgp``, which is none of them.
    """
    if len(dgp.covariate_names) != dgp.n_latent:
        raise ValueError(
            f"{dgp.name} has {dgp.n_latent} latent variable(s) and "
            f"{len(dgp.covariate_names)} covariate(s); a deterministic companion puts the "
            "whole grid in the frame, which a process with hidden variables cannot do"
        )
    if dgp.family != "gaussian":
        raise ValueError(
            f"{dgp.name} has a {dgp.family} outcome; setting Y to its conditional mean is a "
            "valid quadrature only where the mean is a value the outcome can take, and a "
            "binomial draw at Q-bar is not a row of this law"
        )
    for name, value in (
        ("missingness", dgp.missingness),
        ("an intermediate", dgp.intermediate),
        ("clustering", dgp.cluster_size),
    ):
        if value is not None:
            raise ValueError(
                f"{dgp.name} has {name}; the deterministic companion integrates A and Y in "
                "closed form and a further node needs its own sum, which nothing here writes"
            )


def conditional_mean(
    target: np.ndarray,
    *designs: np.ndarray,
    mask: np.ndarray | None = None,
    bins: int = BIN_COUNTS[0],
    weights: np.ndarray | None = None,
) -> np.ndarray:
    r"""``E_0[target | designs]`` on the draw itself, by an equal-count binned average.

    The reduced regressions' **limits** are population conditional expectations given one or
    two scalars that are themselves computable functions of :math:`W`, so estimating them is
    a quadrature over the evaluation draw rather than a second modelling choice.  Equal-count
    bins rather than equal-width ones because a fitted mechanism piles up: an equal-width
    grid puts most of the draw in two cells and leaves the tails at one row each.

    ``mask`` restricts which rows the average is taken over -- the ``| A = a`` of
    :math:`\bar Q_{0,r}`'s definition -- while every row still receives a value, since the
    branch integrals are taken over the whole draw.  A cell with no eligible row falls back
    to the masked mean, which is the coarsest conditioning available rather than a ``nan``
    that would propagate through an integral silently.

    ``weights`` is the rule's own weight per row and is ``None`` for an i.i.d. draw, where
    every row carries :math:`1/m`.  Under :func:`quadrature_frame` it is
    :math:`g_0(a \mid W)/\text{points}`, and the cell average it produces is the conditional
    expectation **exactly** rather than an estimate of one: every design here is a function
    of :math:`(W, a)`, so the two rows a Sobol point contributes fall in the same cell and
    their weights are the law's own conditional probabilities.  ``gr1``'s limit
    :math:`E[1\{A = a\} \mid \hat Q, \bar Q_0]` becomes :math:`E[g_0(a \mid W) \mid \text{cell}]`,
    which is that object's definition rather than a sample proxy for it.

    The **bin edges are unweighted** and deliberately so: the edges only say which rows share
    a cell, and equal-count-in-rows is the same partition of the grid whichever measure the
    average inside it is taken under.  Weighting them would change the cells rather than the
    conditioning, and the discretisation error the ``BIN_COUNTS`` pair measures is about the
    cells.
    """
    values = np.asarray(target, dtype=float).reshape(-1)
    eligible = np.ones(values.size, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    mass = (
        np.ones(values.size)
        if weights is None
        else np.asarray(weights, dtype=float).reshape(-1).copy()
    )
    codes = np.zeros(values.size, dtype=np.int64)
    width = 1
    for design in designs:
        column = np.asarray(design, dtype=float).reshape(-1)
        edges = np.quantile(column, np.linspace(0.0, 1.0, bins + 1)[1:-1])
        codes = codes * bins + np.searchsorted(edges, column, side="right")
        width *= bins
    totals = np.bincount(codes[eligible], weights=(values * mass)[eligible], minlength=width)
    counts = np.bincount(codes[eligible], weights=mass[eligible], minlength=width)
    fallback = _average(values, mass, eligible)
    cells = np.where(counts > 0.0, totals / np.where(counts > 0.0, counts, 1.0), fallback)
    return cells[codes]


def _average(values: np.ndarray, mass: np.ndarray, eligible: np.ndarray | None = None) -> float:
    """``sum(w v) / sum(w)`` over the eligible rows, and ``0.0`` where there are none.

    One helper rather than an ``np.average`` at each site, because ``np.average`` raises on a
    zero total weight and the two callers that can hit one -- an empty mask, and a bin no row
    is eligible for -- both want the same coarsest-conditioning fallback the unweighted
    version already took.
    """
    keep = np.ones(values.size, dtype=bool) if eligible is None else eligible
    total = float(mass[keep].sum())
    if total <= 0.0:
        return 0.0
    return float(np.dot(values[keep], mass[keep]) / total)


def _fold_average(per_fold: list[float], weights: np.ndarray) -> float:
    """Section 5's fold-conditional average, at the estimator's own fold weights."""
    return float(np.dot(np.asarray(per_fold, dtype=float), weights))


def _slice(values: np.ndarray, window: Window | None) -> np.ndarray:
    """One block's rows, or all of them.

    Under :func:`quadrature_frame`'s interleaving a window starting at a block's own start is
    a complete coarser grid of that block, so this is how a ladder is read off **one** fit.
    Under the i.i.d. rule it is a smaller draw, which is the same thing statistically and not
    the same thing exactly.
    """
    return values if window is None else values[window.start : window.stop]


def _row_mass(row_weights: np.ndarray | None, rows: int) -> np.ndarray:
    """The companion rule's weight per row, defaulting to the i.i.d. rule's uniform one.

    Checked against the row count rather than trusted, because a weight vector that is one
    grid stale is the one mistake here that produces a plausible number: it would integrate
    the fitted curve against the wrong measure and report the answer to five decimals.
    """
    if row_weights is None:
        return np.ones(rows, dtype=float)
    mass = np.asarray(row_weights, dtype=float).reshape(-1)
    if mass.size != rows:
        raise ValueError(
            f"the companion holds {rows} rows and the rule supplied {mass.size} weight(s); "
            "quadrature_frame returns the two together for exactly this reason"
        )
    return mass


def plain_remainder(
    result: Any,
    dgp: DGP,
    bounds: tuple[float, float],
    row_weights: np.ndarray | None = None,
    window: Window | None = None,
) -> dict[str, float]:
    r"""``R_2`` per arm at the **fitted** nuisances, fold-weighted over the companion draw.

    .. math::

        R_{2,a} = P_0\Bigl[\frac{\hat g_a - g_{0,a}}{\hat g_a}
                           \bigl(\hat Q(a, W) - \bar Q_0(a, W)\bigr)\Bigr]

    at the **initial** regression and mechanism, which is what the plug-in remainder is a
    remainder of.  The mechanism is the **truncated** one, because that is what the clever
    covariate divides by and so what the expansion is taken at; on a cell whose bound never
    binds the two are the same array.

    Tier 1 gets this exactly by quadrature over a prescribed sequence
    (``benchmarks/drtmle_injection.exact_remainder``); Tier 2 cannot, because its nuisances
    are fitted -- so this is the same quantity read off the companion instead, and the two
    are deliberately different code for the same definition at different tiers.

    ``row_weights`` is the companion rule's weight per row -- ``None`` for the i.i.d. draw,
    :func:`quadrature_frame`'s :math:`g_0(a \mid W)/\text{points}` for the deterministic one.
    This integrand is a function of :math:`W` and the arm alone, so under the deterministic
    rule the arm duplication is redundant and the weights merely undo it: the two rows a
    Sobol point contributes carry the same value and weights summing to :math:`1/\text{points}`.

    ``limit`` reads the first rows only, which under :func:`quadrature_frame`'s interleaving
    is the same integral at a coarser grid.  It does **not** touch the fold weights: those
    are fitting-sample counts and have nothing to do with how many companion rows are read.
    """
    companion = result.nuisance.companion
    if companion is None:
        raise ValueError("R_2 at fitted nuisances needs a companion; fit with evaluation=")
    scaler = result.nuisance.scaler
    latent = _latent(companion.data, dgp)
    weights = companion.fold_weights
    mass = _slice(_row_mass(row_weights, latent.shape[0]), window)

    out: dict[str, float] = {}
    for arm in ARMS:
        per_fold = []
        truth_q = np.asarray(dgp.outcome_mean(latent, arm, None), dtype=float)
        truth_g = _arm_probability(np.asarray(dgp.propensity(latent), dtype=float), arm)
        for fold in range(companion.n_folds):
            estimated_g = companion.propensity[fold].bounded(bounds)[
                :, companion.propensity[fold].column_for(arm)
            ]
            estimated_q = scaler.unscale_levels(companion.outcome[fold].arms[arm])
            per_fold.append(
                _average(
                    _slice((estimated_g - truth_g) / estimated_g * (estimated_q - truth_q), window),
                    mass,
                )
            )
        out[f"r2_{int(arm)}"] = _fold_average(per_fold, weights)
    out["r2_ate"] = out["r2_1"] - out["r2_0"]
    return out


def targeted_remainder(result: Any, dgp: DGP, bounds: tuple[float, float]) -> dict[str, float]:
    r"""``R_2`` per arm at the **targeted** regression: what the fit's bias actually is.

    .. math::

        R_{2,a}(\bar Q^*) = P_0\Bigl[\frac{\hat g_a - g_{0,a}}{\hat g_a}
                                     \bigl(\bar Q^*(a, W) - \bar Q_0(a, W)\bigr)\Bigr]

    -- :func:`plain_remainder`'s expression at :math:`\bar Q^*` in place of :math:`\hat Q`,
    which is the pair ``docs/drtmle/validation-plan.md`` §5's targeted-coefficient clause
    requires be reported together.  C3a's pilot had only the first and read it as the second.

    **Taken over the fit's own rows rather than over the companion**, and that departure from
    every other column in this module is deliberate.  The companion holds each fold's
    *initial* arrays; the targeted ones live on ``ReductionFluctuation.evaluation``, which
    exists only on the ``DRTMLE`` path -- and the quantity wanted here is the **plain
    ``TMLE``'s** bias, since that is the estimator whose interval a shortfall is claimed
    against.  So :math:`P_0` is approximated by the sample mean over the fitting rows, which
    carries an :math:`O(n^{-1/2})` quadrature error that averages down over the study's
    replicates rather than biasing any of them.  ``benchmarks/drtmle_tier1_bias.py`` takes it
    the same way, and at Tier 1 the two can be checked against a quadrature that does not
    (``drtmle_injection.exact_targeted_remainder``).

    :math:`\bar Q_0` and :math:`g_0` come from the law, so this needs no companion at all and
    is available on a fit that declared no ``evaluation=``.
    """
    fluctuation = result.repeats[0].fluctuations["mean"]
    scaler = result.nuisance.scaler
    latent = _latent(result.data, dgp)
    bounded = result.nuisance.propensity.bounded(bounds)

    out: dict[str, float] = {}
    for arm in ARMS:
        estimated_g = bounded[:, result.nuisance.propensity.column_for(arm)]
        truth_g = _arm_probability(np.asarray(dgp.propensity(latent), dtype=float), arm)
        targeted_q = scaler.unscale_levels(fluctuation.targeted.arms[arm])
        truth_q = np.asarray(dgp.outcome_mean(latent, arm, None), dtype=float)
        out[f"r2_{int(arm)}"] = float(
            np.mean((estimated_g - truth_g) / estimated_g * (targeted_q - truth_q))
        )
    out["r2_ate"] = out["r2_1"] - out["r2_0"]
    return out


def corrected_curve(
    result: Any, *, outcome: np.ndarray | None = None
) -> dict[float, list[np.ndarray]]:
    r""":math:`\hat D` **row by row** at the companion: one ``(m,)`` array per fold, per arm.

    The curve is built by :func:`~cleverly.inference.influence.reduced_correction_parts` and
    the same three-term expression :func:`~cleverly.inference.influence.counterfactual_means`
    uses -- **the same functions the reported curve comes through**, at the companion's
    arrays instead of the fit's.  A second copy of that expression written for this module
    is exactly how a remainder comes to describe a curve nobody reported, which is the class
    of defect ``docs/roadmap.md``'s item 20 was.

    The centring is the *fit's* :math:`\hat\psi`, not the companion's own mean: what is
    wanted is :math:`E_0[\hat D]` at the estimator that was reported, and re-centring at the
    evaluation draw would drive it to zero by construction.  Values are on the **scaled**
    outcome; :func:`corrected_remainder` is what unscales them.

    ``outcome`` substitutes the companion's own :math:`Y`, already scaled, and it exists for
    exactly one purpose: :func:`quadrature_frame`'s whole justification is that this curve is
    **affine in** :math:`Y`, and the only instrument that can watch that premise break is one
    that evaluates it at three outcome columns and checks the second difference is zero.  A
    clip, a squared residual or a robust loss anywhere in the curve would leave every other
    check in this module passing and make the deterministic rule quietly wrong.
    """
    record = result.repeats[0].fluctuations["mean"].reduction
    evaluation = None if record is None else record.evaluation
    if evaluation is None:
        raise ValueError("P_0 D-hat needs a companion; fit with evaluation=")

    scaler = result.nuisance.scaler
    data = evaluation.data
    scaled = scaler.scale(data.outcome) if outcome is None else np.asarray(outcome, dtype=float)
    bounds = record.bounds
    guard = tuple(record.guard)

    per_arm: dict[float, list[np.ndarray]] = {arm: [] for arm in ARMS}
    for fold in range(evaluation.n_folds):
        mechanism = evaluation.propensity[fold].arm(ARMS[0])
        targeted = evaluation.outcome[fold]
        corrections = reduced_correction_parts(
            scaled,
            targeted,
            data.treatment,
            evaluation.reduced[fold],
            mechanism,
            bounds=bounds,
            observed=data.observed,
            guard=guard,
        ).total()
        truncated = evaluation.propensity[fold].bounded(bounds)
        residual = scaled - targeted.observed
        for arm in ARMS:
            column = evaluation.propensity[fold].column_for(arm)
            indicator = (np.asarray(data.treatment, dtype=float) == arm).astype(float)
            covariate = indicator / truncated[:, column]
            psi_scaled = (result.estimates[_name(arm)].psi - scaler.lower) / scaler.range
            per_arm[arm].append(
                covariate * residual
                + targeted.arms[arm]
                - psi_scaled
                - np.asarray(corrections[arm], dtype=float)
            )
    return per_arm


def corrected_remainder(
    result: Any,
    dgp: DGP,
    row_weights: np.ndarray | None = None,
    window: Window | None = None,
) -> dict[str, float]:
    r""":math:`P_0\hat D` per estimand, fold-weighted, on the outcome's own scale.

    :func:`corrected_curve` is the curve and this is the two averages over it -- the rule's
    own weighted mean within a fold, and the estimator's fold weights across them.

    **This is the column** ``row_weights`` **exists for.**  Under the i.i.d. rule it is
    ``None`` and the average is a plain one, whose error is :math:`\mathrm{sd}(D)/\sqrt m` and
    lands directly in :attr:`RemainderRow.remaining`; under :func:`quadrature_frame` it is
    :math:`g_0(a \mid W)/\text{points}` and the identity in that function's docstring makes
    the :math:`A` and :math:`Y` coordinates exact, leaving a Sobol quadrature in :math:`W`.

    ``dgp`` is unread and is kept for the call signature every column in this module shares.
    """
    del dgp
    curve = corrected_curve(result)
    scaler = result.nuisance.scaler
    evaluation = result.repeats[0].fluctuations["mean"].reduction.evaluation
    weights = evaluation.fold_weights
    mass = _slice(_row_mass(row_weights, curve[ARMS[0]][0].size), window)

    means = {
        arm: _fold_average([_average(_slice(row, window), mass) for row in rows], weights)
        for arm, rows in curve.items()
    }
    return {
        "ey1": float(scaler.unscale_difference(means[1.0])),
        "ey0": float(scaler.unscale_difference(means[0.0])),
        "ate": float(scaler.unscale_difference(means[1.0] - means[0.0])),
    }


def branch_products(
    result: Any,
    dgp: DGP,
    *,
    bins: int,
    row_weights: np.ndarray | None = None,
    window: Window | None = None,
) -> dict[str, float]:
    r"""Appendix A's and appendix B's **second-order halves**, at one bin count.

    ``R_3 + R_4`` and ``R̃_5 + R̃_6`` of
    ``docs/drtmle/theorem-concordance.md`` §5, with the univariate limits cancelled out and
    the two ``0n`` limits estimated by :func:`conditional_mean` on the evaluation draw.  The
    ``M`` terms are not here and are refused in this module's docstring rather than
    approximated.

    Per arm and summed into the ATE's contrast, on the outcome's own scale.  The reductions
    are read at each fold's own slab and averaged with the fold weights, exactly as
    :func:`corrected_remainder` is: the branch is a property of the same fold-conditional
    estimator the curve is.

    ``row_weights`` reaches **both** the outer integrals and the binned limits inside them,
    and it has to: a ``0n`` limit is a conditional expectation under :math:`P_0` and a cell
    average taken at the wrong measure is a different object, not a noisier reading of the
    same one.  :func:`conditional_mean` says why the weighted cell average is exact and why
    the bin edges stay unweighted.

    ``limit`` slices **before** the binning rather than after it, and it has to: the bins are
    quantiles of the rows read, so a coarser grid gets its own cells.  Averaging a finer
    grid's cell values over a prefix would report a limit conditioned on rows the grid does
    not contain.
    """
    record = result.repeats[0].fluctuations["mean"].reduction
    evaluation = None if record is None else record.evaluation
    if evaluation is None:
        raise ValueError("the appendix branches need a companion; fit with evaluation=")

    scaler = result.nuisance.scaler
    data = evaluation.data
    full = _latent(data, dgp)
    scaled = _slice(scaler.scale(data.outcome), window)
    treatment = _slice(np.asarray(data.treatment, dtype=float), window)
    latent = _slice(full, window)
    weights = evaluation.fold_weights
    bounds = record.bounds
    mass = _slice(_row_mass(row_weights, full.shape[0]), window)

    truth_g_one = np.asarray(dgp.propensity(latent), dtype=float)
    per_arm: dict[str, dict[float, list[float]]] = {"q": {}, "g": {}}
    for arm in ARMS:
        indicator = (treatment == arm).astype(float)
        truth_g = _arm_probability(truth_g_one, arm)
        truth_q = scaler.scale(np.asarray(dgp.outcome_mean(latent, arm, None), dtype=float))
        branch_q, branch_g = [], []
        for fold in range(evaluation.n_folds):
            column = evaluation.propensity[fold].column_for(arm)
            estimated_g = _slice(evaluation.propensity[fold].bounded(bounds)[:, column], window)
            estimated_q = _slice(evaluation.outcome[fold].arms[arm], window)
            reduced = evaluation.reduced[fold]
            qr = _slice(reduced.qr[:, reduced.column_for(arm)], window)
            gr1 = _slice(reduced.bounded_gr1(bounds)[:, reduced.column_for(arm)], window)
            gr2 = _slice(reduced.gr2[:, reduced.column_for(arm)], window)

            # Appendix A. `Q_{0n,r}` conditions on the estimated mechanism *and* the true
            # one, which is what makes R_3 an approximation error rather than a fitted one;
            # `Q_{n,r}` is the fitted reduction, which the companion holds exactly.
            qr_limit = conditional_mean(
                scaled - estimated_q,
                estimated_g,
                truth_g,
                mask=indicator == 1.0,
                bins=bins,
                weights=mass,
            )
            branch_q.append(
                _average((qr_limit / truth_g - qr / estimated_g) * (truth_g - estimated_g), mass)
            )

            # Appendix B. Both reduced mechanisms' `0n` limits condition on the estimated
            # outcome regression and the true one, for the same reason and the other way up.
            gr1_limit = conditional_mean(indicator, estimated_q, truth_q, bins=bins, weights=mass)
            gr2_limit = conditional_mean(
                (indicator - estimated_g) / estimated_g,
                estimated_q,
                truth_q,
                bins=bins,
                weights=mass,
            )
            floor, ceiling = bounds
            gr1_limit = np.clip(gr1_limit, floor, ceiling)
            branch_g.append(
                _average(
                    (indicator * gr2_limit / gr1_limit - indicator * gr2 / gr1)
                    * (scaled - estimated_q),
                    mass,
                )
            )
        per_arm["q"][arm] = branch_q
        per_arm["g"][arm] = branch_g

    out: dict[str, float] = {}
    for key in ("q", "g"):
        one = _fold_average(per_arm[key][1.0], weights)
        zero = _fold_average(per_arm[key][0.0], weights)
        out[f"branch_{key}_ey1"] = float(scaler.unscale_difference(one))
        out[f"branch_{key}_ey0"] = float(scaler.unscale_difference(zero))
        out[f"branch_{key}_ate"] = float(scaler.unscale_difference(one - zero))
    return out


def remainder_rows(
    result: Any,
    dgp: DGP,
    *,
    n: int,
    bounds: tuple[float, float],
    row_weights: np.ndarray | None = None,
    window: Window | None = None,
    truth: dict[str, float] | None = None,
    windows: Sequence[Window] | None = None,
    truths: Sequence[dict[str, float]] | None = None,
) -> list[RemainderRow]:
    """Every remainder column for one fit, one row per estimand.

    ``n`` is the **fitting** sample size, which is what the root-``n`` scaling is in: the
    evaluation draw is a quadrature rule and its size is an accuracy knob, not a sample size
    the estimator's rate is stated in.  Reading it off the companion is the mistake this
    argument exists to prevent.

    ``row_weights`` is that rule's weight per companion row and travels to every column the
    companion feeds -- ``None`` for an i.i.d. draw, :func:`quadrature_frame`'s second return
    value for the quasi-random one.  It does **not** reach :func:`targeted_remainder`, which
    is taken over the fitting rows and has no companion in it at all.  ``window`` reads one
    block of the companion, which under :func:`quadrature_frame` is the same integral at one
    randomisation and, at a shorter window, at a coarser grid.

    ``truth`` is :math:`\\psi_0`, defaulting to :meth:`~cleverly.datasets.DGP.truth`.  **On
    the quasi-random path pass** :func:`truth_at` **at the same grid and the same scramble**:
    the cancellation it documents is what keeps the quadrature acting on a product of nuisance
    errors rather than on :math:`\\psi_0` itself, and it is most of what the rule buys.

    **``windows`` and ``truths`` are how a row averages several replicates of one rule**, and
    every column below is then the mean over them, at a fixed fit.  That is the estimate a
    randomised rule wants -- :math:`R` independent randomisations average to the same integral
    with :math:`R` times less variance -- and it is what puts an honest
    :attr:`RemainderRow.companion_replicate_se` on the row, since a standard error needs
    replication and cannot be had from one block by refining it.  Each window is paired with
    its own :math:`\\psi_0`, because the cancellation is *within* a replicate: averaging the
    remainders is right and averaging :math:`P_0\\hat D` against one shared truth is not.
    """
    if windows is not None and window is not None:
        raise ValueError("pass window= for one block or windows= for several, not both")
    selected = list(windows) if windows is not None else [window]  # type: ignore[list-item]
    if truths is not None and len(truths) != len(selected):
        raise ValueError(
            f"{len(selected)} window(s) and {len(truths)} truth(s); each replicate's psi_0 is "
            "read on its own grid, so the two travel together"
        )
    psi_0s = (
        list(truths)
        if truths is not None
        else [dgp.truth() if truth is None else truth] * len(selected)
    )

    targeted = targeted_remainder(result, dgp, bounds)
    per_window = [
        _one_window(result, dgp, n=n, bounds=bounds, row_weights=row_weights, window=each)
        for each in selected
    ]

    rows = []
    for name in ("ate", "ey1", "ey0"):
        estimate = result.estimates[name]
        pn = float(np.mean(estimate.influence_curve))
        remaining = np.array(
            [
                float(estimate.psi) - psi[name] - (pn - block["p0"][name])
                for block, psi in zip(per_window, psi_0s, strict=True)
            ],
            dtype=float,
        )
        root_n = float(np.sqrt(n)) * remaining
        movement = float(np.mean([block["movement"][name] for block in per_window]))
        branch_q = float(np.mean([block["branch_q"][name] for block in per_window]))
        branch_g = float(np.mean([block["branch_g"][name] for block in per_window]))
        # One direction of the inference only -- see `RemainderRow.branch_movement`. A
        # branch moving more between the two bin counts than its own magnitude is an
        # instrument still visibly moving and is suppressed; a branch that has settled is
        # *not* thereby resolved, and nothing here claims it is.
        settled = movement <= max(abs(branch_q), abs(branch_g))
        rows.append(
            RemainderRow(
                estimand=name,
                psi=float(estimate.psi),
                truth=float(np.mean([psi[name] for psi in psi_0s])),
                p0_curve=float(np.mean([block["p0"][name] for block in per_window])),
                pn_curve=pn,
                remaining=float(remaining.mean()),
                root_n_remaining=float(root_n.mean()),
                r2=float(np.mean([block["r2"][_KEYS[name]] for block in per_window])),
                r2_targeted=targeted[_KEYS[name]],
                branch_q=branch_q if settled else float("nan"),
                branch_g=branch_g if settled else float("nan"),
                branch_movement=movement,
                companion_se=float(np.mean([block["se"][name] for block in per_window])),
                companion_replicate_se=_replicate_se(root_n),
                replicates=len(selected),
            )
        )
    return rows


def _replicate_se(values: np.ndarray) -> float:
    """The standard error of the mean across a rule's replicates, or ``nan`` below two.

    ``nan`` rather than zero, and the distinction is the point: one replicate has no spread,
    and a zero there would read as a rule with no error rather than as an error nobody
    measured.  That is the shape of mistake E1b exists to correct.
    """
    return float(np.std(values, ddof=1) / np.sqrt(values.size)) if values.size > 1 else float("nan")


def _one_window(
    result: Any,
    dgp: DGP,
    *,
    n: int,
    bounds: tuple[float, float],
    row_weights: np.ndarray | None,
    window: Window | None,
) -> dict[str, Any]:
    """Every companion-fed column at one block, before the averaging across blocks."""
    branches = {
        bins: branch_products(result, dgp, bins=bins, row_weights=row_weights, window=window)
        for bins in BIN_COUNTS
    }
    coarse, fine = branches[BIN_COUNTS[0]], branches[BIN_COUNTS[1]]
    return {
        "p0": corrected_remainder(result, dgp, row_weights, window),
        "r2": plain_remainder(result, dgp, bounds, row_weights, window),
        "se": _companion_witness(result, n, row_weights, window),
        "branch_q": {name: fine[f"branch_q_{name}"] for name in ("ate", "ey1", "ey0")},
        "branch_g": {name: fine[f"branch_g_{name}"] for name in ("ate", "ey1", "ey0")},
        # A *movement* between two bin counts and not an error: see
        # `RemainderRow.branch_movement` for why the difference matters and which half of
        # the inference this column can carry.
        "movement": {
            name: max(
                abs(fine[f"branch_q_{name}"] - coarse[f"branch_q_{name}"]),
                abs(fine[f"branch_g_{name}"] - coarse[f"branch_g_{name}"]),
            )
            for name in ("ate", "ey1", "ey0")
        },
    }


def _companion_witness(
    result: Any,
    n: int,
    row_weights: np.ndarray | None,
    window: Window | None,
) -> dict[str, float]:
    r"""The i.i.d. rule's error per estimand from a formula, on the :math:`\sqrt n` scale.

    :math:`\sqrt n\,\mathrm{sd}(\hat D)/\sqrt m`, which is about the *rule* and not about the
    estimator -- the whole of what E1 set out to separate, since a reader who cannot see the
    quadrature's contribution cannot tell a flat column from a noisy one.

    **It is the i.i.d. rule's error and nothing else**, and it is an enormous overstatement of
    a quasi-random rule's.  What that rule's error is comes from replication, which is a
    property of several windows rather than of one, so it lives in :func:`remainder_rows`
    where the windows are.  The halving witness this once returned beside it is gone: a
    successive difference is a stability diagnostic and was read as a bound, which is E1b's
    first retraction.
    """
    curve = corrected_curve(result)
    scaler = result.nuisance.scaler
    evaluation = result.repeats[0].fluctuations["mean"].reduction.evaluation
    folds = evaluation.fold_weights
    total = curve[ARMS[0]][0].size
    rows = total if window is None else window.rows
    mass = _slice(_row_mass(row_weights, total), window)

    spread: dict[float, float] = {}
    for arm, per_fold in curve.items():
        # The fold-weighted standard deviation of the curve itself.  Squared and combined
        # across folds rather than pooled, because the folds hold different functions and a
        # pooled spread would charge their disagreement to the draw.
        variances = [
            _average(_slice(row, window) ** 2, mass) - _average(_slice(row, window), mass) ** 2
            for row in per_fold
        ]
        spread[arm] = float(np.sqrt(max(_fold_average(variances, folds), 0.0)))

    root_n = float(np.sqrt(n))
    scale = float(scaler.range)
    # `ate`'s spread is bounded by the sum rather than computed from a per-row contrast: the
    # two arms' curves are the same rows, so a difference of standard deviations would
    # understate and their sum is the honest conservative reading of a witness column.
    spread["ate"] = spread[1.0] + spread[0.0]  # type: ignore[index]
    out: dict[str, float] = {}
    for name in ("ate", "ey1", "ey0"):
        key: Any = "ate" if name == "ate" else (1.0 if name == "ey1" else 0.0)
        out[name] = root_n * scale * spread[key] / float(np.sqrt(rows))
    return out


#: Which remainder key an estimand reads.  One mapping rather than one per call site, since
#: the two remainder columns are indexed the same way and a slip between them would put a
#: contrast's number under an arm's.
_KEYS = {"ate": "r2_ate", "ey1": "r2_1", "ey0": "r2_0"}


def _name(arm: float) -> str:
    return "ey1" if arm == 1.0 else "ey0"


def _arm_probability(one: np.ndarray, arm: float) -> np.ndarray:
    """``P(A = arm | W)`` from the arm-1 column, by complement -- the binary path's rule."""
    return one if arm == 1.0 else 1.0 - one


def _latent(data: Any, dgp: DGP) -> np.ndarray:
    """The latent matrix ``dgp.propensity`` and ``dgp.outcome_mean`` are defined on.

    The observed covariates **are** the first columns of it, and a process with hidden
    columns has no way to hand them back -- so such a law is refused here rather than
    silently evaluated at zeros, which would return a plausible truth for a different
    process.  Every cell of the coverage study is drawn from a fully observed law.
    """
    covariates = np.asarray(data.covariates, dtype=float)
    if covariates.shape[1] != dgp.n_latent:
        raise ValueError(
            f"{dgp.name} has {dgp.n_latent} latent variable(s) and the draw carries "
            f"{covariates.shape[1]} covariate(s); the remainder columns evaluate the law's "
            "own nuisances at the evaluation rows, which a process with hidden variables "
            "cannot supply"
        )
    return covariates
