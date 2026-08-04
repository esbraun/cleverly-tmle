"""How does the doubly-robust alternation actually exit, and what does its curve rest on?

Run with ``python benchmarks/bench_drtmle.py``.  This is a **characterisation**, not a
benchmark and not a test: it asserts nothing, and its output is a distribution rather than
a verdict.

``docs/roadmap.md``'s open items 4, 6 and 7 all rested on the same evidence -- six seeded
fits at ``n = 800`` on one process -- and item 7 wanted the loop's exit criterion changed
but was explicit that the change had to wait: *"a threshold changed after seeing a failure
needs the failure characterised first"*.  This script is that characterisation, and it
exists so the numbers stop living as prose in a docstring.  It is the same reasoning
``bench_tmle.py`` states for keeping its rows: a comparison nobody can rerun becomes
folklore.

**It has been run twice on the same 96 draws, and both tables are in
``docs/drtmle/investigation-log.md``** -- the first under *How the alternation exits*, the
second under *What the B2b dispatch measured*.  The first was taken before the exit criterion
item 7 replaced and before piece B1b; the second is the live one, and every column moved:

* **the exit distribution inverted**, from 2 tolerance / 86 stall / 8 cap to **87 / 8 / 1**,
  at a median of 4 to 9 rounds against 12 to 24 and a seventh of the wall clock.  Nothing
  about the iteration changed -- the exit test reads a different ruler;
* **item 6 now holds universally**, 96 of 96 against 94 of 96, which is the one place a
  bounded mechanism convention made a limitation slightly worse and is where that prediction
  stopped being a guess;
* **item 7's disagreement is 96 of 96**, against 68, which is that same ruler seen from the
  other side;
* **``weak-overlap``'s score check fails on 0 of 24**, against 23 of 24 -- items 11 and 20,
  closed by B1b, measured at scale rather than on four fixtures.  Its overlap columns are
  unchanged, which is what says the failure was the convention and not the draws.

So **rerun this to see whether those hold, not to reproduce the first table**, and read the
second before quoting any of it.

The questions, and the columns that answer them:

* **item 4 -- how often does equation (10)'s near-singular solve bite?**  ``ill>0`` and the
  ``exit`` split.  :math:`g_{r,2}` vanishes exactly where the mechanism is right, so the
  covariate is worst conditioned on the fits anybody wants; ``linear`` is in the sweep
  because it is the process a ``glm`` mechanism gets *right*.
* **item 6 -- does the closing pass's mechanism stage ever stop on its tolerance?**
  ``closing capped``.
* **item 7 -- how far apart are the relative and absolute criteria?**  ``rel eq10`` against
  ``worst |score|`` as a share of ``se/sqrt(n)``.
* **item 12 -- stopping and validity are two questions.**  ``std score`` is the standardised
  score :math:`|P_n S_j| / \\hat{sd}(S_j)`, reported **beside** the stopping rule rather than
  folded into it, which is §4's instruction and the thing the loop's ``_NEGLIGIBLE / n`` bar
  conflates.
* **item 22's numerical half -- do the two update orders reach the same fixed point?**
  ``--order paper`` fits each draw a second time under the working paper's own recursion and
  reports :math:`\\Delta\\psi` in units of ``se``, and the ratio of the two standard errors.
  Both orders run *here*, against the same nuisances, which is what
  ``docs/drtmle/theorem-concordance.md`` §6 asks for; **do not** compare fluctuation
  coefficients across them, since the submodels a round passes through differ.
* **the five places weak overlap enters** (§4).  ``clip``, ``min g``, ``ess``, and the high
  quantiles of all three clever covariates and of the reductions they are built from -- so
  that a failure can be attributed to one of them rather than to "poor overlap".
* **item 25 -- which estimator is each cell evidence about?**  ``bound-active``, a count of
  the cell's draws on the far side of
  `the supported contract <../docs/roadmap.md>`_, off the same
  :func:`~cleverly.validation.correction_check` witness the coverage study reads.  It is a
  scope column and not a failure count: those draws' identities hold at roundoff and their
  scores are negligible.

**This will not run in the Claude Code cloud sandbox**, and ``CLAUDE.md`` explains why: the
defaults here are ~96 fits of tens of seconds each, and ``--order paper`` doubles that at a
higher cost per fit (the paper's route took 22 rounds against 8 on the draw both were first
measured on).  Dispatch ``.github/workflows/drtmle-convergence.yml`` and read the tables out
of the job log, or run it on a machine with cores to spare.  Interrupt with ``Ctrl-C`` and
never ``kill -9``.

Usage::

    python benchmarks/bench_drtmle.py                                   # the full sweep
    python benchmarks/bench_drtmle.py --processes nonlinear --sizes 400 --seeds 2 --jobs 1
    python benchmarks/bench_drtmle.py --order paper                     # item 22, numerically
    python benchmarks/bench_drtmle.py --reduced-learner fast            # a richer reduction
    python benchmarks/bench_drtmle.py --truncation 0.01 0.025           # psi and se by bound
    python benchmarks/bench_drtmle.py --rows                            # every fit, not cells
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from cleverly import DRTMLE
from cleverly.datasets import DGP, linear_dgp, nonlinear_dgp, weak_overlap_dgp
from cleverly.estimators.base import format_table
from cleverly.estimators.tmle import correction_parts
from cleverly.utils.parallel import map_parallel

#: Two sizes rather than the one the existing measurement used.  The alternation's cost is
#: round count times folds and barely depends on ``n`` (``tests/unit/test_drtmle_fit.py``
#: measures 400 rows as *slower* than 600), so a second size is nearly free and is the only
#: way to see whether the conditioning is a property of the sample size at all.
DEFAULT_SIZES = (600, 1200)

#: Fits per cell.  Twelve is enough draws to say whether an exit is the norm or the
#: exception, which is the only claim items 4 and 6 make.
#:
#: It used to say "nothing here needs a Monte Carlo standard error", and the ``--order`` arm
#: is where that stopped being true: comparing two update orders is a claim about a
#: *distribution* of paired differences, not about whether an exit is common.  Twelve is
#: still what it is, and the response is the instrument rather than the count -- the
#: route-against-reseed table reports a distribution-free **count** of pairs, which is honest
#: at twelve, and a mean with ``sd/sqrt(M)`` beside the median rather than a Monte Carlo
#: interval on a median, which this repository has no estimator for.  Raise it for that arm
#: rather than reading its median as though it were a coverage number.
DEFAULT_SEEDS = 12


def _off_diagonal() -> DGP:
    """Linear outcome, nonlinear treatment: ``glm`` is right for one nuisance and not the other.

    The cell the variant is *for*, built the way ``tests/e2e/test_coverage_slow.py`` builds
    it so the two are talking about the same process.
    """
    linear, hard = linear_dgp(), nonlinear_dgp()
    return DGP(
        name="linear outcome, nonlinear treatment",
        n_latent=4,
        covariate_names=("W1", "W2", "W3", "W4"),
        propensity=hard.propensity,
        outcome_mean=linear.outcome_mean,
    )


#: ``linear`` earns its place by being the *easy* process.  Equation (10)'s covariate is
#: ``gr2/gr1`` and ``gr2 = E[(1_a - g-hat)/g-hat | Qbar]`` vanishes where the mechanism is
#: right, so a process a ``glm`` gets right is where that covariate is nearest to zero and
#: its Newton solve nearest to singular.  A sweep over hard processes alone would measure
#: the pathology where it is least likely to appear.
PROCESSES: dict[str, Callable[[], DGP]] = {
    "linear": linear_dgp,
    "nonlinear": nonlinear_dgp,
    "weak-overlap": weak_overlap_dgp,
    "off-diagonal": _off_diagonal,
}

#: ``tests/conftest.py``'s ``FAST_KWARGS``, restated because ``conftest`` lives under
#: ``tests/`` and this does not -- ``bench_tmle.py`` restates its equivalent for the same
#: reason.  ``estimands`` is spelled out because the default binary report asks for
#: ``att``/``atc``, which ``DRTMLE`` refuses.
SETTINGS = {
    "outcome_learner": "glm",
    "treatment_learner": "glm",
    "n_folds": 5,
    "learner_folds": 3,
    "simultaneous": False,
    "estimands": ("ate", "ey1", "ey0"),
}

#: The quantile the covariate and reduction columns are read at.  A maximum is one row and
#: says whatever that row did; the 99th percentile is the tail the score's variance actually
#: comes from, and is what ``sensitivity/positivity.py`` reports for the same reason.
TAIL = 0.99


@dataclass(frozen=True)
class Overlap:
    """Where weak overlap enters this estimator, one column per place (§4 of the plan).

    ``1/g`` in equation (8) is one of five, and locating a failure means being able to say
    which.  Every field is read off the fit and none is derived from another.

    Attributes
    ----------
    clip_share:
        The share of ``(row, arm)`` pairs whose **initial** mechanism lies outside the
        truncation.  The initial one and not the targeted one, deliberately: since B1b the
        alternation carries the truncated tilt forward, so a converged fit clips nothing at
        the exit however hard the draw was, and a count taken there would be zero on every
        row of this table -- ``docs/roadmap.md``'s stop-ship 14, a column that could not
        disagree.  This is a property of the draw.

        **Read off the fit rather than recomputed here.**  It was three lines of numpy in
        this module until item 25's witness put the same quantity on
        :class:`~cleverly.validation.CorrectionCheck`, and two implementations of "which side
        of the supported contract is this cell on" is one more than the question can stand:
        the study in ``benchmarks/drtmle_coverage.py`` reads the fit's, and a sweep answering
        with its own arithmetic could disagree with it about a cell.  The number is
        unchanged, which was checked before the duplicate went.
    contract:
        ``"theorem"`` or ``"bound-active"``, off the same witness: which estimator this fit's
        numbers are evidence about (item 25).  Not a verdict -- a bound-active fit here has
        every identity at roundoff and every score negligible, which is exactly what the
        ``weak-overlap`` rows of the B2b dispatch say.
    margin:
        How close the targeted mechanism comes to either bound, as a fraction of the
        interval between them.  The witness B1b replaced the clipped count with: a
        constrained root sits *against* the boundary of the feasible set, so a draw whose
        tilt wanted to leave the bounds comes back pressed to one.
    min_g:
        The smallest initial mechanism value over rows and arms, untruncated.
    ess:
        The smallest per-arm effective sample size, as a share of ``n``:
        ``(sum h)^2 / (n * sum h^2)`` for ``h = 1_a / g^b(a|W)``.  Kish's, on the clever
        covariate rather than on a weight, which is what the covariate *is* here.
    h8, h9, h10:
        The 99th percentile of each equation's clever covariate in absolute value --
        ``1_a/g*`` for (8), ``Q_r/g*`` for (9) and ``g_{r,2}/g_{r,1}`` for (10).  Three
        columns rather than one because the three go bad for different reasons, which is
        the whole point of §4's widened diagnosis.
    qr, gr1, gr2:
        The 99th percentile of :math:`|Q_r|`, the *smallest* :math:`g_{r,1}` -- it is the
        denominator, so its tail is at the bottom -- and the 99th percentile of
        :math:`|g_{r,2}|`.
    """

    clip_share: float = float("nan")
    contract: str = ""
    margin: float = float("nan")
    min_g: float = float("nan")
    ess: float = float("nan")
    h8: float = float("nan")
    h9: float = float("nan")
    h10: float = float("nan")
    qr: float = float("nan")
    gr1: float = float("nan")
    gr2: float = float("nan")


@dataclass(frozen=True)
class Curve:
    """What the reported curve rests on: the identity, the scores, and where they come from.

    Attributes
    ----------
    identity:
        The worst :math:`|\\Delta|` over arms and equations -- the score the targeting step
        recorded minus the mean of the term the reported curve subtracts, from
        :func:`~cleverly.validation.drtmle.correction_check`.  Zero to roundoff since B1b,
        and on this table so that a regression is a number rather than a silence.
    clip_bias:
        The worst :math:`|B_{clip}|`, which is zero for the same reason and is the column
        that says *why* the identity holds rather than that it does.
    standardised:
        :math:`\\max_j |P_n S_j| / \\hat{sd}(S_j)` over the three equations, which is §4's
        separate diagnostic: it says whether the fit that came out is entitled to a Wald
        interval, where the loop's own bar says when to stop iterating.  Multiply by
        ``sqrt(n)`` for the t-scale a reader compares against 1.
    top1, top5, top10:
        The share of the worst-behaved score's absolute mass carried by its top 1%, 5% and
        10% of rows.  A score driven to zero by cancellation between a handful of large
        rows and everything else is a different object from one that is small rowwise, and
        only the second is evidence that the equation was solved in a way an interval can
        rest on.
    hessian:
        The reported outcome fluctuation's Hessian condition number.  It describes the
        **closing pass's joint solve** over equations (8) and (10) together, which is what
        the reported fit came out of.  Equation (9)'s has no counterpart since B1b: the
        bounded solve is a root find rather than a Newton step, so there is no Hessian to
        report and the column would be ``nan`` on every row -- ``ill`` on the table above is
        what carries equation (10)'s conditioning instead.
    """

    identity: float = float("nan")
    clip_bias: float = float("nan")
    standardised: float = float("nan")
    top1: float = float("nan")
    top5: float = float("nan")
    top10: float = float("nan")
    hessian: float = float("nan")


@dataclass
class Exit:
    """One fit's record.  Every field is read off the fit, none is derived."""

    process: str
    n: int
    data_seed: int
    fold_seed: int
    #: Which arm of the sweep this fit is: ``"base"``, ``"paper"``, ``"reseed"``,
    #: ``"reduced"``, ``"nested"`` or ``"trunc=<lower>"``.  The comparison tables group on
    #: everything *but* this.
    variant: str
    seconds: float
    exit_reason: str
    rounds: int
    converged: bool
    failure: str
    ill_conditioned: int
    closing: int
    closing_capped: bool
    #: Equation (10)'s relative score at the round the **loop** exited on -- the quantity the
    #: exit test compared against ``spec.tol``, and so the one item 7 is about.  Not the same
    #: as the closing row below, which is what the reported curve rests on: the loop's last
    #: round is solved at reductions the curve never reads.
    loop_reduced: float
    loop_mechanism: float
    closing_reduced: float
    scale_reduced: float
    epsilon_max: float
    score_ok: bool
    worst_share: float
    psi: float
    se: float
    overlap: Overlap = field(default_factory=Overlap)
    curve: Curve = field(default_factory=Curve)
    error: str = ""

    @property
    def draw(self) -> tuple[str, int, int]:
        """The draw this fit is of, which is what a comparison across arms pairs on."""
        return (self.process, self.n, self.data_seed)

    def row(self) -> list[str]:
        return [
            self.process,
            f"{self.n:,}",
            str(self.data_seed),
            self.variant,
            self.error or self.exit_reason,
            str(self.rounds),
            str(self.ill_conditioned),
            f"{self.closing}{'*' if self.closing_capped else ''}",
            f"{self.loop_reduced:.1e}",
            f"{self.closing_reduced:.1e}",
            f"{self.scale_reduced:.1e}",
            f"{self.worst_share:.1e}",
            "yes" if self.score_ok else "NO",
            self.failure,
        ]


def _failed(payload: Payload, error: str) -> Exit:
    """A draw the estimator raised on, recorded rather than dropped.

    A sweep that swallowed these would report the exits of the fits that happened to
    survive and call that the distribution.
    """
    return Exit(
        process=payload.process,
        n=payload.n,
        data_seed=payload.data_seed,
        fold_seed=payload.fold_seed,
        variant=payload.variant,
        seconds=float("nan"),
        exit_reason="",
        rounds=0,
        converged=False,
        failure="",
        ill_conditioned=0,
        closing=0,
        closing_capped=False,
        loop_reduced=float("nan"),
        loop_mechanism=float("nan"),
        closing_reduced=float("nan"),
        scale_reduced=float("nan"),
        epsilon_max=float("nan"),
        score_ok=False,
        worst_share=float("nan"),
        psi=float("nan"),
        se=float("nan"),
        error=error,
    )


@dataclass(frozen=True)
class Payload:
    """One fit to run: which draw, and which arm of the sweep it belongs to."""

    process: str
    n: int
    data_seed: int
    fold_seed: int
    variant: str = "base"
    #: ``update_order=``, ``reduced_*_learner=`` and ``g_bounds=`` for this arm, applied on
    #: top of :data:`SETTINGS`.  Carried here rather than derived from ``variant`` inside
    #: the worker so that the arm's *name* and the settings that produce it cannot drift.
    settings: tuple[tuple[str, object], ...] = ()

    @property
    def key(self) -> tuple[str, int, int]:
        """The draw, which is what a comparison across arms groups on."""
        return (self.process, self.n, self.data_seed)


def _quantile(values: np.ndarray, q: float = TAIL) -> float:
    finite = np.asarray(values, dtype=float).reshape(-1)
    finite = finite[np.isfinite(finite)]
    return float(np.quantile(np.abs(finite), q)) if finite.size else float("nan")


def _concentration(rows: np.ndarray, share: float) -> float:
    """What fraction of a score's absolute mass its largest ``share`` of rows carries.

    Uniform contributions give back ``share`` itself, so a column reading 0.5 at
    ``share = 0.01`` says half the score lives on one row in a hundred.
    """
    values = np.sort(np.abs(np.asarray(rows, dtype=float).reshape(-1)))[::-1]
    total = values.sum()
    if not values.size or total <= 0:
        return float("nan")
    take = max(1, round(share * values.size))
    return float(values[:take].sum() / total)


def _standardised(rows: np.ndarray) -> float:
    """:math:`|P_n S| / \\hat{sd}(S)`, §4's separate diagnostic."""
    values = np.asarray(rows, dtype=float).reshape(-1)
    spread = float(np.std(values))
    return float(abs(np.mean(values)) / spread) if spread > 0 else float("nan")


def _targeted_propensity(fit) -> np.ndarray:
    """The mechanism the reported covariates divide by, ``(n, K)`` and truncated.

    Arm 1 is the tilted one and arm 0 is its **complement**, which is how
    :meth:`~cleverly.estimators._nuisance.Propensity.bounded` forms it on the binary path --
    it clips ``g1`` and complements it rather than clipping both columns.  A diagnostic that
    formed the pair differently would describe covariates no fit here used, so this is one
    function rather than the same three lines in two callers.
    """
    repeat = fit.repeats[0]
    fluctuation = repeat.fluctuations["mean"]
    reduction = fluctuation.reduction
    upper_arm = reduction.reduced.arms[-1]
    tilted = (
        fluctuation.mechanism.propensity
        if fluctuation.mechanism is not None
        else repeat.nuisance.propensity.arm(upper_arm)
    )
    return np.clip(np.column_stack([1.0 - tilted, tilted]), *reduction.bounds)


def _indicators(fit, arms: tuple[float, ...]) -> np.ndarray:
    """``1{A = a}`` as an ``(n, K)`` design in the reductions' own arm order."""
    treatment = np.asarray(fit.data.treatment, dtype=float).reshape(-1)
    return np.column_stack([(treatment == arm).astype(float) for arm in arms])


def _overlap(fit, bounds: tuple[float, float], correction: Any) -> Overlap:
    """The five places, read off one fit.

    ``correction`` is the fit's own :func:`~cleverly.validation.correction_check`, which is
    where three of these columns now come from rather than from arithmetic repeated here --
    see :attr:`Overlap.clip_share`.
    """
    repeat = fit.repeats[0]
    nuisance, fluctuation = repeat.nuisance, repeat.fluctuations["mean"]
    reduction = fluctuation.reduction
    arms = reduction.reduced.arms

    raw = np.column_stack([nuisance.propensity.arm(arm) for arm in arms])
    starred = _targeted_propensity(fit)
    h8 = _indicators(fit, arms) / starred
    ess = min(
        float(column.sum() ** 2 / (len(column) * np.square(column).sum()))
        for column in h8.T
        if np.square(column).sum() > 0
    )
    reduced = reduction.reduced
    return Overlap(
        clip_share=correction.initial_clip_share,
        contract=correction.contract,
        margin=correction.margin,
        min_g=float(np.min(raw)),
        ess=ess,
        h8=_quantile(h8),
        h9=_quantile(np.asarray(reduced.qr) / starred),
        h10=_quantile(np.asarray(reduced.gr2) / reduced.bounded_gr1(bounds)),
        qr=_quantile(reduced.qr),
        gr1=float(np.min(reduced.gr1)),
        gr2=_quantile(reduced.gr2),
    )


def _curve(fit) -> Curve:
    """The identity, the standardised scores and where each score's mass sits."""
    repeat = fit.repeats[0]
    nuisance, fluctuation = repeat.nuisance, repeat.fluctuations["mean"]
    data = fit.data
    scaled = nuisance.scaler.scale(data.outcome)
    weights = np.asarray(data.weights, dtype=float).reshape(-1)
    parts = correction_parts(data, nuisance, fluctuation, fluctuation.targeted, scaled)
    check = fit.validation.correction_check()

    # Equation (8)'s rowwise contribution per arm, and the two corrections' from the very
    # arrays the reported curve subtracts -- `correction_parts` is what the curve and
    # `correction_check` both come through, so a third expression here would be a second
    # implementation of the formula rather than a reading of it.
    arms = fluctuation.reduction.reduced.arms
    residual = np.asarray(scaled) - np.asarray(fluctuation.targeted.observed)
    starred = _targeted_propensity(fit)
    indicators = _indicators(fit, arms)

    contributions = [
        weights * indicators[:, j] / starred[:, j] * residual for j in range(len(arms))
    ]
    if parts is not None:
        for arm in arms:
            if "Q" in parts.guard:
                contributions.append(weights * np.asarray(parts.d_g[arm]))
            if "g" in parts.guard:
                contributions.append(weights * np.asarray(parts.d_q[arm]))

    standardised = [_standardised(rows) for rows in contributions]
    worst = contributions[int(np.nanargmax(standardised))] if standardised else np.zeros(0)
    identity = [abs(row.residual) for row in check.rows if np.isfinite(row.residual)]
    clip_bias = [abs(row.clip_bias) for row in check.rows if np.isfinite(row.clip_bias)]
    return Curve(
        identity=max(identity, default=float("nan")),
        clip_bias=max(clip_bias, default=float("nan")),
        standardised=float(np.nanmax(standardised)) if standardised else float("nan"),
        top1=_concentration(worst, 0.01),
        top5=_concentration(worst, 0.05),
        top10=_concentration(worst, 0.10),
        hessian=float(fluctuation.hessian_condition),
    )


def one_fit(payload: Payload) -> Exit:
    """Fit once and read the alternation's record and its diagnostics off it."""
    frame, _ = PROCESSES[payload.process]().sample(payload.n, seed=payload.data_seed)
    estimator = DRTMLE(
        **SETTINGS,  # type: ignore[arg-type]
        **dict(payload.settings),  # type: ignore[arg-type]
        random_state=payload.fold_seed,
    )
    start = time.perf_counter()
    try:
        fit = estimator.fit(frame, outcome="Y", treatment="A").single()
    # Recorded and reported rather than swallowed: `main` prints how many raised and what
    # they raised, so a sweep cannot quietly report the exits of the survivors.
    except Exception as exc:
        return _failed(payload, type(exc).__name__)
    seconds = time.perf_counter() - start

    reduction = fit.repeats[0].fluctuations["mean"].reduction
    # Two rows, and the distinction is the point. The last is the closing pass's, which is
    # where the reported curve is built; the one before it is the round the loop exited on,
    # whose relative score is what the exit test was applied to. Item 7 is a question about
    # the second, and reading the first would answer a different one. The trace always has
    # both: at least one refitting round, plus the closing row.
    _, _, loop_reduced, loop_mechanism, _ = reduction.trace[-2]
    _, _, closing_reduced, _, _ = reduction.trace[-1]
    check = fit.validation.score_check()
    correction = fit.validation.correction_check()
    estimate = fit.estimates["ate"]
    reference = estimate.std_error / np.sqrt(payload.n)
    worst = max(abs(row.score) for row in check.rows)

    return Exit(
        process=payload.process,
        n=payload.n,
        data_seed=payload.data_seed,
        fold_seed=payload.fold_seed,
        variant=payload.variant,
        seconds=seconds,
        exit_reason=reduction.exit_reason,
        rounds=reduction.n_outer,
        converged=reduction.converged,
        failure=reduction.failure or "",
        ill_conditioned=reduction.ill_conditioned,
        closing=reduction.closing,
        closing_capped=reduction.closing_capped,
        loop_reduced=float(loop_reduced),
        loop_mechanism=float(loop_mechanism),
        closing_reduced=float(closing_reduced),
        # The *smallest* column scale, not the mean of them: `relative_score` is a max over
        # columns of `|score|/scale`, so the smallest denominator is the one that inflates
        # the ratio the loop exits on.  Exactly zero is not a missing value -- it is a
        # column of the covariate that vanished identically, which is the pathology item 4
        # describes taken to its limit.
        scale_reduced=float(np.min(reduction.score_scale)) if reduction.score_scale.size else 0.0,
        epsilon_max=float(np.max(np.abs(reduction.epsilon))) if reduction.epsilon.size else 0.0,
        score_ok=check.passed,
        worst_share=float(worst / reference) if reference > 0 else float("nan"),
        psi=float(estimate.psi),
        se=float(estimate.std_error),
        overlap=_overlap(fit, reduction.bounds, correction),
        curve=_curve(fit),
    )


def _share(values: list[bool]) -> str:
    return f"{sum(values)}/{len(values)}"


def _median(values: Sequence[float]) -> float:
    finite = [value for value in values if np.isfinite(value)]
    return statistics.median(finite) if finite else float("nan")


def _cells(results: list[Exit], variant: str) -> list[tuple[str, int, list[Exit]]]:
    """One entry per ``(process, n)`` cell of a single arm, in a stable order."""
    chosen = [r for r in results if r.variant == variant and not r.error]
    keys = sorted({(r.process, r.n) for r in chosen}, key=lambda c: (c[0], c[1]))
    return [
        (process, n, [r for r in chosen if r.process == process and r.n == n])
        for process, n in keys
    ]


def summarise(results: list[Exit]) -> list[list[str]]:
    """One row per (process, n) cell: what the twelve draws did, not what one of them did."""
    rows: list[list[str]] = []
    cells = sorted({(r.process, r.n) for r in results}, key=lambda c: (c[0], c[1]))
    for process, n in cells:
        cell = [r for r in results if r.process == process and r.n == n and r.variant == "base"]
        if not cell:
            continue
        ok = [r for r in cell if not r.error]
        if not ok:
            rows.append([process, f"{n:,}", f"0/{len(cell)}", "", "", "", "", "", "", "", ""])
            continue
        rounds = [r.rounds for r in ok]
        exits = [r.exit_reason for r in ok]
        rows.append(
            [
                process,
                f"{n:,}",
                f"{len(ok)}/{len(cell)}",
                f"{statistics.median(rounds):.0f} [{min(rounds)}-{max(rounds)}]",
                "/".join(str(exits.count(kind)) for kind in ("tolerance", "stall", "cap")),
                _share([r.ill_conditioned > 0 for r in ok]),
                _share([r.closing_capped for r in ok]),
                f"{_median([r.loop_reduced for r in ok]):.1e}",
                f"{_median([r.scale_reduced for r in ok]):.1e}",
                f"{_median([r.worst_share for r in ok]):.1e}",
                _share([not r.score_ok for r in ok]),
            ]
        )
    return rows


def overlap_rows(results: list[Exit]) -> list[list[str]]:
    """The five places weak overlap enters, per cell, as medians over the draws.

    ``bound-active`` is a **count** and not a median, because item 25's contract is a
    statement about a cell rather than about its typical draw: a cell with one bound-active
    fit in twelve is a cell whose coverage number is evidence about two estimators, and a
    median would report it as though it were about one.
    """
    rows = []
    for process, n, cell in _cells(results, "base"):
        rows.append(
            [
                process,
                f"{n:,}",
                f"{_median([r.overlap.clip_share for r in cell]):.3f}",
                _share([r.overlap.contract == "bound-active" for r in cell]),
                f"{_median([r.overlap.margin for r in cell]):.1e}",
                f"{_median([r.overlap.min_g for r in cell]):.3f}",
                f"{_median([r.overlap.ess for r in cell]):.2f}",
                f"{_median([r.overlap.h8 for r in cell]):.1f}",
                f"{_median([r.overlap.h9 for r in cell]):.2e}",
                f"{_median([r.overlap.h10 for r in cell]):.2e}",
                f"{_median([r.overlap.qr for r in cell]):.2e}",
                f"{_median([r.overlap.gr1 for r in cell]):.3f}",
                f"{_median([r.overlap.gr2 for r in cell]):.2e}",
            ]
        )
    return rows


def curve_rows(results: list[Exit]) -> list[list[str]]:
    """What the reported curve rests on, per cell."""
    rows = []
    for process, n, cell in _cells(results, "base"):
        rows.append(
            [
                process,
                f"{n:,}",
                f"{max(r.curve.identity for r in cell):.1e}",
                f"{max(r.curve.clip_bias for r in cell):.1e}",
                f"{_median([r.curve.standardised for r in cell]):.2e}",
                f"{max(r.curve.standardised for r in cell):.2e}",
                f"{_median([r.curve.top1 for r in cell]):.2f}",
                f"{_median([r.curve.top5 for r in cell]):.2f}",
                f"{_median([r.curve.top10 for r in cell]):.2f}",
                f"{_median([r.curve.hessian for r in cell]):.1e}",
            ]
        )
    return rows


def comparison_rows(results: list[Exit], variant: str) -> list[list[str]]:
    """One arm against ``base``, draw by draw, summarised per cell.

    Paired on the draw rather than compared cell-mean to cell-mean: the two arms fit the
    *same* data with the *same* fold seed, so the difference is the arm's and pairing is
    what removes the draw-to-draw variation that would otherwise swamp it.

    ``worst identity`` is here because :func:`curve_rows` is base-only, and the update-order
    rule frozen in ``docs/drtmle/validation-plan.md`` §4 asks for the state identity in
    **either** arm rather than in the one that happens to be the reference.  Every
    :class:`Exit` already carries a populated :class:`Curve` -- :func:`one_fit` computes it
    whatever the arm -- so the number existed and only the table dropped it.  The column
    makes that clause measurable; it does not move the rule, and reading it as a change to
    one would be the thing §4 was frozen to prevent.
    """
    base = {
        (r.process, r.n, r.data_seed): r for r in results if r.variant == "base" and not r.error
    }
    rows = []
    for process, n, cell in _cells(results, variant):
        paired = [(base[r.draw], r) for r in cell if r.draw in base]
        if not paired:
            continue
        shifts = [abs(other.psi - first.psi) / first.se for first, other in paired]
        ratios = [other.se / first.se for first, other in paired]
        rows.append(
            [
                process,
                f"{n:,}",
                str(len(paired)),
                f"{_median(shifts):.2e}",
                f"{max(shifts):.2e}",
                f"{_median(ratios):.4f}",
                f"{min(ratios):.4f} - {max(ratios):.4f}",
                _share([not other.score_ok for _, other in paired]),
                # A max rather than a median, as `curve_rows` takes it: the identity's right
                # value is zero, so the cell's verdict is its worst row and an average would
                # let one broken fit hide behind eleven sound ones.
                f"{max(other.curve.identity for _, other in paired):.1e}",
                f"{_median([other.rounds for _, other in paired]):.0f}",
            ]
        )
    return rows


def _shifts(results: list[Exit], variant: str) -> dict[tuple[str, int], list[tuple[Exit, float]]]:
    """``|dpsi|/se`` against ``base`` for one arm, per cell, keyed so two arms can be paired."""
    base = {
        (r.process, r.n, r.data_seed): r for r in results if r.variant == "base" and not r.error
    }
    out: dict[tuple[str, int], list[tuple[Exit, float]]] = {}
    for process, n, cell in _cells(results, variant):
        out[(process, n)] = [
            (r, abs(r.psi - base[r.draw].psi) / base[r.draw].se) for r in cell if r.draw in base
        ]
    return out


def route_rows(results: list[Exit], variant: str = "paper") -> list[list[str]]:
    r"""One arm's difference from base, against the yardstick of a different fold split.

    Item 22's numerical half asks whether the two update orders reach the same fixed point.
    On its own ``|dpsi|/se`` cannot answer that, because it has no scale: ``0.22`` was
    measured on one draw, and until something says what a *different split of the same
    order* moves, that number is equally consistent with "the routes disagree" and with
    "this is what any refit does".  The ``reseed`` arm is that something, and this table is
    where the two are read together.

    ``variant`` is what makes it serve item 15 as well, and the generalisation is not a
    convenience: the cross-fitting construction is the *same* question with a different arm
    in it -- is the difference this change makes larger than the difference a redrawn split
    makes -- so it wants this yardstick rather than a second one built beside it.  What the
    two questions do **not** share is which answer supports which conclusion, and
    ``docs/drtmle/validation-plan.md`` §7 is where that is written down: for the update
    order agreement was the expected finding, and for the construction a difference that
    *shrinks* is.

    Three columns carry the reading.  The two medians are the paired quantity per arm.  The
    **count** is distribution-free and is the one to trust at twelve draws: in how many of
    them the route moved ``psi`` further than the reseed did.  Around half is the null --
    the routes are doing what a split does -- and a count near the pair count is a route
    difference that is *not* split noise, whichever way the medians happen to fall.

    The mean and ``sd/sqrt(M)`` are reported beside the median for continuity with
    :attr:`~cleverly.validation.simulation.EstimandSummary.bias_se`, which is the form this
    repository already uses for "is this real?".  There is no median-based Monte Carlo
    standard error here or anywhere in the package, which is why the count exists.
    """
    route, noise = _shifts(results, variant), _shifts(results, "reseed")
    rows = []
    for key in sorted(set(route) & set(noise)):
        by_draw = {fit.draw: value for fit, value in noise[key]}
        paired = [(value, by_draw[fit.draw]) for fit, value in route[key] if fit.draw in by_draw]
        if not paired:
            continue
        ours = [value for value, _ in paired]
        theirs = [value for _, value in paired]
        spread = float(np.std(ours, ddof=1)) if len(ours) > 1 else float("nan")
        rows.append(
            [
                key[0],
                f"{key[1]:,}",
                str(len(paired)),
                f"{_median(ours):.2e}",
                f"{_median(theirs):.2e}",
                f"{sum(1 for a, b in paired if a > b)}/{len(paired)}",
                f"{float(np.mean(ours)):.2e} +/- {spread / np.sqrt(len(ours)):.1e}",
            ]
        )
    return rows


def _payloads(args: argparse.Namespace, seeds: list[tuple[int, int, int]]) -> list[Payload]:
    """Every fit the requested arms ask for, base first so a failure is visible early.

    An arm is ``(variant, settings, reseed)``.  ``reseed`` is what makes the control arm
    possible: it changes the **fold** seed and nothing else, where every other arm changes a
    setting and holds the fold seed fixed.  Both kinds pair against ``base`` on
    ``(process, n, data_seed)``, so one comparison machinery serves both.
    """
    payloads = []
    arms: list[tuple[str, tuple[tuple[str, object], ...], bool]] = [("base", (), False)]
    if args.order:
        arms.append(("paper", (("update_order", args.order),), False))
    if args.order_control:
        # Same estimator, same data, one different fold split. Without it `|dpsi|/se`
        # between the two update orders has no yardstick: a route difference and a split
        # difference are the same number until something says which is which.
        arms.append(("reseed", (), True))
    if args.reduced_learner:
        arms.append(
            (
                "reduced",
                (
                    ("reduced_outcome_learner", args.reduced_learner),
                    ("reduced_treatment_learner", args.reduced_learner),
                ),
                False,
            )
        )
    if args.reduced_crossfit:
        # A plain `DRTMLE` keyword, so it needs no subclass the way the oracle reductions
        # did -- the construction rides in the settings tuple like any other arm, and every
        # comparison table picks it up unchanged.
        arms.append(("nested", (("reduced_crossfit", args.reduced_crossfit),), False))
    for lower in args.truncation:
        arms.append((f"trunc={lower:g}", (("g_bounds", (lower, 1.0 - lower)),), False))
    for variant, settings, reseed in arms:
        for process in args.processes:
            for n in args.sizes:
                for data_seed, fold_seed, control_seed in seeds:
                    payloads.append(
                        Payload(
                            process,
                            n,
                            data_seed,
                            control_seed if reseed else fold_seed,
                            variant,
                            settings,
                        )
                    )
    return payloads


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processes", nargs="+", default=list(PROCESSES), choices=list(PROCESSES))
    parser.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES))
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20250801, help="the sweep's own seed")
    parser.add_argument("--rows", action="store_true", help="print every fit, not just the cells")
    parser.add_argument(
        "--order",
        choices=("paper",),
        default=None,
        help="also fit every draw under the working paper's update order (item 22)",
    )
    parser.add_argument(
        "--order-control",
        action="store_true",
        help="also fit every draw at a different fold seed, as the yardstick the update-order "
        "comparison is read against",
    )
    parser.add_argument(
        "--reduced-learner",
        default=None,
        help="also fit every draw with this learner for the reduced regressions",
    )
    parser.add_argument(
        "--reduced-crossfit",
        choices=("nested",),
        default=None,
        help="also fit every draw with the reduced regressions nested inside each outer "
        "fold's complement (item 15); read against --order-control, which is the yardstick "
        "that says whether a construction difference is larger than a split's",
    )
    parser.add_argument(
        "--truncation",
        type=float,
        nargs="*",
        default=[],
        help="also refit every draw at these lower g_bounds; a refit rather than a retarget, "
        "so gr2's target moves with the bound and the curve is the valid one",
    )
    args = parser.parse_args()

    # Three streams, and the first two are what they always were: `generate_state` is
    # prefix-stable, so drawing a third block leaves every earlier run's data and fold seeds
    # bit for bit unchanged and the tables stay comparable. The second stream varies because
    # the one pathological fit on record was "a fit whose fold split was drawn unseeded", so
    # a sweep holding `random_state` at FAST_KWARGS's 0 would sweep straight past the thing
    # it is measuring; the third is the control arm's, and is only read when it is asked for.
    #
    # **That stability is across a third *stream*, not across `--seeds`, and the difference
    # bites.** The blocks are `[:s]`, `[s:2s]` and `[2s:]`, so raising `s` leaves the data
    # seeds' prefix alone and moves the fold and control blocks wholesale: a 36-seed run
    # shares its first twelve *datasets* with a 12-seed one and not one of their fold splits.
    # Two such runs are therefore not nested, and neither supersedes the other -- read them
    # as separate samples that happen to share some draws. Measured: raising `--seeds` from
    # 12 to 36 moved `weak-overlap`'s median route shift at `n = 600` from `1.6e-01` to
    # `5.0e-01`, most of which is this rather than the extra draws.
    drawn = np.random.SeedSequence(args.seed).generate_state(3 * args.seeds)
    seeds = [
        (int(data), int(fold), int(control))
        for data, fold, control in zip(
            drawn[: args.seeds],
            drawn[args.seeds : 2 * args.seeds],
            drawn[2 * args.seeds :],
            strict=True,
        )
    ]

    payloads = _payloads(args, seeds)
    arms = sorted({payload.variant for payload in payloads})
    print(
        f"{len(payloads)} fits over {len(args.processes)} processes, arms {arms}, jobs={args.jobs}"
    )
    started = time.perf_counter()
    results = map_parallel(one_fit, [(payload,) for payload in payloads], n_jobs=args.jobs)
    elapsed = time.perf_counter() - started

    if args.rows:
        print("\nEvery fit")
        print("=" * 9)
        print(
            format_table(
                [
                    "process",
                    "n",
                    "seed",
                    "arm",
                    "exit",
                    "rounds",
                    "ill",
                    "closing",
                    "eq10 at exit",
                    "eq10 at close",
                    "min mean|h|",
                    "|score| / (se/sqrt n)",
                    "check",
                    "failure",
                ],
                [r.row() for r in results],
            )
        )

    title = "How the alternation exited"
    print(f"\n{title}")
    print("=" * len(title))
    print(
        format_table(
            [
                "process",
                "n",
                "fits",
                "rounds med [range]",
                "tol/stall/cap",
                "ill>0",
                "closing capped",
                "med eq10 at exit",
                "med min mean|h|",
                "med |score| / (se/sqrt n)",
                "check fails",
            ],
            summarise(results),
        )
    )

    title = "Where weak overlap enters"
    print(f"\n{title}")
    print("=" * len(title))
    print(
        format_table(
            [
                "process",
                "n",
                "clip share",
                "bound-active",
                "margin",
                "min g",
                "ess/n",
                "q99 h(8)",
                "q99 h(9)",
                "q99 h(10)",
                "q99 |Qr|",
                "min gr1",
                "q99 |gr2|",
            ],
            overlap_rows(results),
        )
    )

    title = "What the reported curve rests on"
    print(f"\n{title}")
    print("=" * len(title))
    print(
        format_table(
            [
                "process",
                "n",
                "worst identity",
                "worst B_clip",
                "med std score",
                "max std score",
                "top 1%",
                "top 5%",
                "top 10%",
                "med hessian cond",
            ],
            curve_rows(results),
        )
    )

    for variant in arms:
        if variant == "base":
            continue
        title = f"{variant} against base, paired on the draw"
        print(f"\n{title}")
        print("=" * len(title))
        print(
            format_table(
                [
                    "process",
                    "n",
                    "pairs",
                    "med |dpsi|/se",
                    "max |dpsi|/se",
                    "med se ratio",
                    "se ratio range",
                    "check fails",
                    "worst identity",
                    "med rounds",
                ],
                comparison_rows(results, variant),
            )
        )

    if {"paper", "reseed"} <= set(arms):
        title = "The update-order difference, against what a different fold split moves"
        print(f"\n{title}")
        print("=" * len(title))
        print(
            format_table(
                [
                    "process",
                    "n",
                    "pairs",
                    "med route |dpsi|/se",
                    "med reseed |dpsi|/se",
                    "route > reseed",
                    "mean route +/- se",
                ],
                route_rows(results, "paper"),
            )
        )
        print(
            "\n  Read the count first and the medians second. Around half the pairs is the null:\n"
            "  the two routes move psi about as much as a different split of one route does, so\n"
            "  what looked like a route difference is what any refit does. A count near the pair\n"
            "  count is a route difference the split cannot account for -- and the rule it is\n"
            "  judged against is in docs/drtmle/validation-plan.md section 4, predeclared."
        )

    if {"nested", "reseed"} <= set(arms):
        title = "The cross-fitting construction, against what a different fold split moves"
        print(f"\n{title}")
        print("=" * len(title))
        print(
            format_table(
                [
                    "process",
                    "n",
                    "pairs",
                    "med nested |dpsi|/se",
                    "med reseed |dpsi|/se",
                    "nested > reseed",
                    "mean nested +/- se",
                ],
                route_rows(results, "nested"),
            )
        )
        print(
            "\n  The same yardstick as the table above and the opposite reading, which is why the\n"
            "  rule is written down before the run. There, two routes provably solve the same\n"
            "  equations and agreement was the expected finding. Here the two constructions are\n"
            "  different estimators, and what the pooled argument needs is not that they agree\n"
            "  but that the difference *shrinks* with n -- a large but shrinking gap supports\n"
            "  pooled cross-fitting and a small but stable one does not. Read the trend across\n"
            "  sizes first; docs/drtmle/validation-plan.md section 7 is the predeclared rule."
        )

    print("\nReading the numbers")
    print("=" * 19)
    ok = [r for r in results if not r.error and r.variant == "base"]
    errored = [r for r in results if r.error]
    if errored:
        kinds = sorted({r.error for r in errored})
        print(f"{len(errored)} of {len(results)} fits raised ({', '.join(kinds)}) and are")
        print("excluded from every column above; they are not a convergence outcome.")
    if not ok:
        print("No fit completed, so there is nothing to characterise.")
        return

    capped = [r for r in ok if r.exit_reason == "cap"]
    stalled = [r for r in ok if r.exit_reason == "stall"]
    print(
        f"{len(capped)} of {len(ok)} fits ran out of rounds and {len(stalled)} stalled; "
        f"{len(ok) - len(capped) - len(stalled)} reached the tolerance. Read all three "
        "counts, and against both prior sweeps: 8/86/2 cap/stall/tolerance under the exit "
        "criterion item 7 replaced, and 1/8/87 under the one in force. Item 4 of the roadmap "
        "called a capped exit a minority behaviour of particular draws; it is that, and "
        "stalling is no longer what the loop mostly does."
    )
    closing_capped = [r for r in ok if r.closing_capped]
    print(
        f"The closing pass's mechanism stage stopped on its 20-step cap on {len(closing_capped)} "
        f"of {len(ok)} fits. Item 6 expects all of them: equation (9)'s covariate reads the "
        "mechanism it tilts, so iterating shrinks the residual without removing it."
    )
    # Item 7's question, put as one number: a fit the relative criterion calls unsolved
    # while the absolute one calls negligible is a fit the exit criterion is wrong about.
    disagree = [r for r in ok if r.loop_reduced > 1e-10 and r.worst_share < 1e-3]
    print(
        f"On {len(disagree)} of {len(ok)} fits equation (10)'s relative score is above the "
        f"loop's tolerance while the worst absolute score is under 1e-3 of se/sqrt(n) -- the "
        "relative criterion calling unsolved what the statistical one calls negligible. That "
        "is item 7, and it is the count a threshold change would have to be argued from."
    )
    # Items 11 and 20, re-measured. Before B1b this was the count that mattered most, and
    # it is here so that a regression is a number on the face of the sweep rather than
    # something a reader has to go looking for.
    broken = [r for r in ok if np.isfinite(r.curve.identity) and r.curve.identity > 1e-12]
    print(
        f"{len(broken)} of {len(ok)} fits report a state identity above 1e-12. That is items "
        "11 and 20, closed by piece B1b: the score is solved at the truncated tilt the curve "
        "reads, so anything but zero here is a regression rather than a fit that needs more "
        "rounds, and the wording `correction_check` uses says so."
    )
    print(
        f"\n{len(results)} fits in {elapsed:.0f}s wall clock at jobs={args.jobs} "
        f"({_median([r.seconds for r in results if not r.error]):.1f}s median per fit)."
    )


if __name__ == "__main__":
    main()
