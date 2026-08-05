r"""Is item 13's flat remainder a *learner* failure at all?  The gates, before the number.

``docs/roadmap.md``'s **E2**.  C3c read :math:`\sqrt n R_{\text{remaining}}` as flat with the
three reduced regressions fitted by ``glm``, a configuration whose own consistency the
concordance marks ``unverified``.  So the study tested Theorem 1's conclusion without
establishing its premise, and the cheapest way to tell a falsified *configuration* from a
falsified *estimator* is to refit both cells with the reductions at their population limits and
see whether the column moves.

**This module runs that comparison, and it runs the gates that have to pass first.**
``benchmarks/drtmle_reference.py`` builds the reference and says in its own docstring what it
does not establish: the result is a **numerical reference and not an oracle**, and what is left
after the construction is a smoothing bias and a finite point count.  A paired
reference-against-``glm`` number read before those are bounded is not evidence about the
reduction learner, and this module exists so that it cannot be read that way -- the gate table
prints first, and the comparison prints a difference with an interval rather than a verdict.

**The verdict is not here yet, and that is the commit boundary rather than an omission.**
``docs/roadmap.md``'s E2 requires the equivalence margin and the reference-uncertainty budget
to be *frozen in their own commit*, so what this module prints is readings: the gates' two
statistics and the paired difference, each with the draws behind it.  The rule those are read
against arrives next, before any dispatch, which is the ordering C3c's own value rests on.

**Why the gates cannot be a refinement difference, which is the trap here.**  The obvious
fidelity statistic is the movement of the reference between two knot counts.  That is the
statistic ``docs/roadmap.md`` withdrew for the quadrature ladder and then again for the binned
branches, and E2 inheriting it would rebuild the mistake it exists to repair.  Measured on this
repository's own ladder a successive difference ran four times *below* the true error at the
finest rung and three orders *above* it two rungs earlier; it is uninformative in both
directions, and it is undefined for the rung a ladder starts on.

**So the three gates are three different instruments, and none of them is that one.**

*A -- the exact-law control*, which is a test rather than a column and lives in
``tests/unit/test_reference_exact_law.py``: on a law whose conditioning index is discrete the
reduced regressions are finite sums, and the provider reproduces them array for array.  It
gates the **construction** -- the weights, the mask, the arms, the recomputation at the current
targeted pair -- and it is silent about the smoother, since no exact law has a continuum in it.

*B -- the held-out weighted risk*, here.  Each candidate is fitted on the reference block and
scored on an **independent, finer** block from a disjoint scramble stream.  The cross term
vanishes identically because a reference is a weighted :math:`L_2` projection
(:func:`~benchmarks.drtmle_reference.held_out_risk` carries the algebra), so a *difference* of
two candidates' risks estimates a difference of squared weighted errors.  It **orients**, which
is what a refinement difference cannot do: a near-interpolating reference has a smaller
movement and a larger risk, so the two point opposite ways and only one of them ranks.

*C -- the randomisation budget*, here.  Independent scrambles of the **reference** block, one
refit each.  This is what a fixed grid's error cannot be estimated from and what E1b's device
supplies: a randomised quasi-Monte Carlo rule is unbiased at every point count, so the spread
across scrambles is a standard error assuming no rate.  Unlike E1b's evaluation rule it is
**not free** -- the reference enters the fit, so a scramble is a fit -- which is why the budget
runs on a subset of draws and is a declared cost rather than a default.

**What C cannot see, and B is why it is not alone.**  Every scramble shares the knot count and
the basis, so the across-scramble spread is orthogonal to a bias in the basis.  Randomisation
gates the quadrature half and the held-out risk gates the smoothing half; neither substitutes
for the other, and the exact-law control gates neither.

**The comparison is paired at every level it can be.**  One draw, one fold seed, one companion:
the ``glm`` arm and the reference arm see the same rows, the same prescribed nuisance
functions, the same split, and the **same evaluation windows and truths** -- so the evaluation
rule's own error, which E1b measured as very nearly the whole of a one-replicate study's
across-draw spread at :math:`n = 2{,}400`, is largely common to the two and cancels in the
difference.  What is left is the reduction learner.

**What this module does not do**, each refused by name.

* **No coverage claim.**  The reduction learner's effect on an interval is not this question,
  and a module that also reported a coverage number would be answering something its inputs
  were not frozen for.
* **No learner comparison.**  Growing-basis against ``glm`` against ``boost`` is E2b's and
  fires only if this branches.  The ladder here is the *reference's own* resolution, read as a
  fidelity gate rather than as a contest between estimator families.
* **No rate.**  Item 13 is a rate and closes at E5.  Two sizes here size the comparison, they
  do not carry an exponent.
* **The word oracle.**  The reference is a numerical estimate of a population limit and every
  gate above exists because it is one.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cleverly import DRTMLE
from cleverly.estimators.base import format_table
from cleverly.utils.parallel import map_parallel

# The benchmarks package is a checkout rather than an installed distribution, so a plain
# `python benchmarks/drtmle_reference_study.py` has to find its siblings the way its
# neighbours do.
if __package__ in (None, ""):  # pragma: no cover - only on the direct-script path
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks import drtmle_injection, drtmle_remainder, drtmle_tier2
from benchmarks.drtmle_reference import (
    KNOT_LADDER,
    EqualCountBins,
    ReferenceReductionDRTMLE,
    SplineProjection,
    arm_truth,
    fit_mask,
    fold_targets,
    held_out_risk,
)

#: The two tiers, keyed as every harness on this page keys them so one interface serves both.
TIERS = {1: drtmle_injection, 2: drtmle_tier2}

#: Selected at import and replaced by ``main``; the tier is a module of designs rather than a
#: branch, exactly as it is in the coverage harness.
injection: Any = drtmle_injection

#: The learner the reference is compared **against**, and it is not a knob.  It is C3c's
#: configuration, which is the thing under test: the question is whether that configuration's
#: reductions are why the column is flat, and changing it would answer a different one.
REDUCED_LEARNER = "glm"

#: The reference itself -- the middle rung of ``benchmarks/drtmle_reference.KNOT_LADDER``,
#: named here rather than defaulted so a run's banner and the record cannot come apart.
REFERENCE = SplineProjection(KNOT_LADDER[1])

#: The rungs gate B ranks the reference among.  A ladder over an integer, so there is no
#: continuous constant to commit to, and it is read as a **risk ordering** and never as a
#: movement between two resolutions.
RUNGS = tuple(SplineProjection(knots) for knots in KNOT_LADDER)

#: The negative control gate B must reject.  Not a candidate: it is here so that "the gate
#: rejects a reference that is too coarse" is a measurement rather than a hope, and because a
#: deliberately coarse arm reaches a *close final estimate* while being a badly wrong function
#: -- which is the one shape of result that would otherwise tempt a reader to skip the gate.
NEGATIVE_CONTROL = EqualCountBins(8)

#: The three scramble streams, disjoint from each other and from the coverage study's.  Which
#: block a rule takes must not depend on which rows a draw was fitted on, and the reference's
#: stream must not meet the evaluation's for a stronger reason: the reference's error
#: propagates into the fit **deterministically**, so a shared scramble would make the fit and
#: the integral the same random variable with a covariance nobody can sign.
REFERENCE_SEED = 93_000_000
SCORING_SEED = 94_000_000
EVALUATION_SEED = 95_000_000

#: Sobol points per block.  The reference block is what the three regressions are fitted on and
#: is sized by the **points-per-parameter budget** the reference refuses to be fitted thinner
#: than: ``spline(32)`` has 35 parameters and so needs 2,240 rows, and ``qr``'s ``| A = a``
#: mask keeps about half of a block's ``2 * points``.  The scoring block is deliberately finer,
#: since a held-out risk carries its own Monte Carlo error and nothing pairs it away.
DEFAULT_REFERENCE_POINTS = 4_096
DEFAULT_SCORING_POINTS = 8_192
DEFAULT_EVALUATION_POINTS = 2_048

#: Independent evaluation scrambles averaged into one reading per fit.  Two rather than one
#: because :func:`~benchmarks.drtmle_remainder.remainder_rows` then reports a replicate
#: standard error rather than ``nan``, and because averaging a randomised rule halves its
#: variance -- both arms read the same ones, so this sharpens the *pair* and not one side.
DEFAULT_EVALUATION_SCRAMBLES = 2

#: Gate C's budget: independent reference scrambles, and the draws they are taken on.  A
#: refit each, unlike E1b's evaluation replicates, so this is a cost rather than a default and
#: the two are separate knobs.
DEFAULT_BUDGET_SCRAMBLES = 4
DEFAULT_BUDGET_DRAWS = 4

DEFAULT_SIZES = (600, 2_400)
DEFAULT_DRAWS = 32

#: Resamples behind every reported interval, and the percentiles they are read at.  A bootstrap
#: over **draws**, because the draw is the independent unit: a fit's gate rows and its
#: evaluation replicates are both conditional on it.
BOOTSTRAP = 2_000
INTERVAL = (2.5, 97.5)

#: Which reduced regression names are reported, in the order every table prints them.
REDUCTIONS = ("qr", "gr1", "gr2")


# ------------------------------------------------------------------------------- the records


@dataclass
class FitRow:
    """One estimand of one fit, and the grain the paired table is arithmetic on.

    Flat and JSON-serialisable, as every record on this page is: a nested one would need a
    schema before a reader could recompute anything from the artefact.

    Attributes
    ----------
    estimator:
        ``"glm"`` or ``"reference"``.  The pairing key is ``(cell, n, data_seed, estimand)``,
        and the two arms share a fold seed and a companion.
    scramble:
        Which reference block this fit's reductions were fitted on.  ``0`` on the ``glm`` arm,
        which has no reference block -- and not the scramble it *would* have used, since a
        column that reads like a randomisation on an arm that took none is how a budget comes
        to be computed over the wrong rows.
    root_n_remaining:
        Item 13's column, averaged over this fit's evaluation replicates.  The paired
        difference of the two arms' values is the whole of what E2 decides on.
    companion_replicate_se:
        The spread of that column across those replicates, which is the evaluation rule's own
        error at this fit and is **not** the estimator's.
    companion_rows:
        Rows of the whole stacked companion this fit carried -- the reference blocks, the
        scoring block and the evaluation blocks together.  What a fit is *billed* in, since
        every one of them is a prediction per fold per nuisance, and it differs between an
        ordinary draw and a gate-C budget draw.
    """

    cell: str
    n: int
    data_seed: int
    fold_seed: int
    estimator: str
    scramble: int
    estimand: str
    psi: float
    truth: float
    p0_curve: float
    pn_curve: float
    remaining: float
    root_n_remaining: float
    companion_replicate_se: float
    companion_rows: int
    rounds: int
    exit_reason: str
    valid: bool
    seconds: float
    error: str = ""


@dataclass
class RiskRow:
    """Gate B's grain: one candidate's held-out risk on one regression, arm and fold.

    Reported apart rather than averaged, because the three reductions have different targets on
    different scales -- ``qr``'s is a residual and ``gr1``'s is an indicator -- so a mean over
    them would be a number with no units and a gate on it would be a gate on ``gr1``.
    """

    cell: str
    n: int
    data_seed: int
    candidate: str
    reduction: str
    treatment_arm: float
    fold: int
    risk: float
    fitted_rows: int
    scored_rows: int
    error: str = ""


@dataclass(frozen=True)
class Payload:
    """One draw: both arms, its gate rows, and however many reference scrambles it carries."""

    cell: str
    n: int
    data_seed: int
    fold_seed: int
    reference_points: int
    scoring_points: int
    evaluation_points: int
    evaluation_scrambles: int
    reference_scrambles: int


@dataclass(frozen=True)
class Layout:
    """A draw's companion, sliced into the three roles its blocks play.

    One frame and one weight vector for all of them, which is
    :func:`~benchmarks.drtmle_remainder.stacked_companion`'s contract and is what stops a
    caller pairing a window with the wrong rule's measure.  What is new here is that the blocks
    are at **different resolutions**: the scoring block has to be finer than the block a
    candidate was fitted on, which is a property of the roles and not of the rule.
    """

    stack: Any
    reference: tuple[Any, ...]
    scoring: Any
    evaluation: tuple[Any, ...]


def layout(payload: Payload, dgp: Any) -> Layout:
    """Every block one draw needs, in one companion, from three disjoint scramble streams."""
    offset = payload.data_seed % 1_000_003
    reference_seeds = [REFERENCE_SEED + offset + i for i in range(payload.reference_scrambles)]
    evaluation_seeds = [EVALUATION_SEED + offset + i for i in range(payload.evaluation_scrambles)]
    scrambles = [*reference_seeds, SCORING_SEED + offset, *evaluation_seeds]
    points = [
        *[payload.reference_points] * len(reference_seeds),
        payload.scoring_points,
        *[payload.evaluation_points] * len(evaluation_seeds),
    ]
    stack = drtmle_remainder.stacked_companion(dgp, points=points, scrambles=scrambles)
    taken = len(reference_seeds)
    return Layout(
        stack=stack,
        reference=stack.blocks[:taken],
        scoring=stack.blocks[taken],
        evaluation=stack.blocks[taken + 1 :],
    )


class RecordingReferenceDRTMLE(ReferenceReductionDRTMLE):
    """Keeps the last nuisance state its provider was handed.

    Gate B has to score candidates **at a state the reference actually answered at**, and the
    alternation's last one is the one the reported reductions came from.  Nothing on a
    ``TMLEResult`` carries it: ``nuisance`` is the *initial* pair, deliberately, because
    ``NuisanceEstimates.outcome`` stays the initial regression exactly as it does on ``TMLE``.
    Recording it here is five lines and keeps the gate a statement about the fit rather than
    about its starting point.
    """

    state: Any = None
    bounds: tuple[float, float] | None = None

    def _reference_set(self, current: Any, g_bounds: tuple[float, float]) -> Any:
        self.state = current
        self.bounds = g_bounds
        return super()._reference_set(current, g_bounds)


# ------------------------------------------------------------------------------ one draw


def _fit(payload: Payload, estimator: Any) -> Any:
    import warnings

    dgp = injection.base_law()
    frame, _ = dgp.sample(payload.n, seed=payload.data_seed)
    with warnings.catch_warnings():
        # Positivity warnings are per-draw noise here, as they are in the coverage harness.
        warnings.simplefilter("ignore")
        return estimator.fit(frame, outcome="Y", treatment="A").single()


def _settings(payload: Payload, evaluation: Any) -> dict[str, Any]:
    return dict(
        injection.settings(payload.cell, payload.n),
        reduced_outcome_learner=REDUCED_LEARNER,
        reduced_treatment_learner=REDUCED_LEARNER,
        random_state=payload.fold_seed,
        evaluation=evaluation,
    )


def _alternation(fit: Any) -> dict[str, Any]:
    reduction = fit.repeats[0].fluctuations["mean"].reduction
    if reduction is None:  # pragma: no cover - a DRTMLE fit always carries one
        return {"rounds": 0, "exit_reason": ""}
    return {"rounds": int(reduction.rounds), "exit_reason": str(reduction.exit_reason)}


def _fit_rows(
    payload: Payload,
    fit: Any,
    place: Layout,
    *,
    estimator: str,
    scramble: int,
    seconds: float,
) -> list[FitRow]:
    """Every estimand's remainder columns at one fit, read on the evaluation blocks.

    Both arms are handed the *same* windows and the same per-window :math:`\\psi_0`, which is
    what makes the difference below a paired quantity rather than two independent readings of
    an integral.  Each window carries its own truth because the cancellation
    :func:`~benchmarks.drtmle_remainder.truth_at` documents is **within** a replicate.
    """
    dgp = injection.base_law()
    check = fit.validation.score_check()
    alternation = _alternation(fit)
    windows = [block.window for block in place.evaluation]
    truths = [
        drtmle_remainder.truth_at(dgp, block.points, scramble=block.seed)
        for block in place.evaluation
    ]
    measured = drtmle_remainder.remainder_rows(
        fit,
        dgp,
        n=payload.n,
        bounds=fit.config.g_bounds,
        row_weights=place.stack.weights,
        windows=windows,
        truths=truths,
    )
    return [
        FitRow(
            cell=payload.cell,
            n=payload.n,
            data_seed=payload.data_seed,
            fold_seed=payload.fold_seed,
            estimator=estimator,
            scramble=scramble,
            estimand=row.estimand,
            psi=row.psi,
            truth=row.truth,
            p0_curve=row.p0_curve,
            pn_curve=row.pn_curve,
            remaining=row.remaining,
            root_n_remaining=row.root_n_remaining,
            companion_replicate_se=row.companion_replicate_se,
            companion_rows=int(place.stack.weights.size),
            valid=bool(check.passed),
            seconds=seconds,
            **alternation,
        )
        for row in measured
    ]


def _failed_fit(
    payload: Payload, estimator: str, scramble: int, error: str, rows: int
) -> list[FitRow]:
    nan = float("nan")
    return [
        FitRow(
            cell=payload.cell,
            n=payload.n,
            data_seed=payload.data_seed,
            fold_seed=payload.fold_seed,
            estimator=estimator,
            scramble=scramble,
            estimand=name,
            psi=nan,
            truth=nan,
            p0_curve=nan,
            pn_curve=nan,
            remaining=nan,
            root_n_remaining=nan,
            companion_replicate_se=nan,
            companion_rows=rows,
            rounds=0,
            exit_reason="",
            valid=False,
            seconds=nan,
            error=error,
        )
        for name in ("ate", "ey1", "ey0")
    ]


def _risk_rows(payload: Payload, state: Any, bounds: Any, place: Layout) -> list[RiskRow]:
    r"""Gate B: every candidate's held-out weighted risk, per regression, arm and fold.

    Fitted on the reference block and scored on the scoring block, which is a **different
    scramble at a finer resolution** -- rows no candidate saw, from a stream disjoint from
    both the reference's and the evaluation's.  The mask travels with the regression: ``qr``'s
    ``| A = a`` restricts the rows it is scored on exactly as it restricts the rows it is
    fitted on, or the risk would be taken under a measure the regression is not defined at.

    A candidate whose points-per-parameter budget the block does not meet is recorded as a row
    carrying its refusal rather than dropped: a rung that could not be scored is a gap in the
    gate and has to look like one.
    """
    dgp = injection.base_law()
    mass = np.asarray(place.stack.weights, dtype=float)
    fitting = _block_mask(place.reference[0].window, mass.size)
    scoring = _block_mask(place.scoring.window, mass.size)
    candidates = (*RUNGS, NEGATIVE_CONTROL)

    rows: list[RiskRow] = []
    for arm in state.arms:
        truth = arm_truth(state, dgp=dgp, arm=arm)
        for fold in range(state.companion.n_folds):
            designs, targets = fold_targets(state, fold=fold, arm=arm, truth=truth, g_bounds=bounds)
            for name in REDUCTIONS:
                keep = fit_mask(name, truth.indicator)
                inside = fitting if keep is None else (fitting & keep)
                outside = scoring if keep is None else (scoring & keep)
                for candidate in candidates:
                    row = RiskRow(
                        cell=payload.cell,
                        n=payload.n,
                        data_seed=payload.data_seed,
                        candidate=candidate.label,
                        reduction=name,
                        treatment_arm=float(arm),
                        fold=fold,
                        risk=float("nan"),
                        fitted_rows=int(inside.sum()),
                        scored_rows=int(outside.sum()),
                    )
                    try:
                        fitted = candidate.fit(
                            designs[name][inside], targets[name][inside], mass[inside]
                        )
                        row.risk = held_out_risk(
                            fitted, designs[name][outside], targets[name][outside], mass[outside]
                        )
                    except Exception as exc:  # recorded, never swallowed
                        row.error = type(exc).__name__
                    rows.append(row)
    return rows


def _block_mask(window: Any, rows: int) -> np.ndarray:
    mask = np.zeros(rows, dtype=bool)
    mask[window.start : window.stop] = True
    return mask


def one_draw(payload: Payload) -> tuple[list[FitRow], list[RiskRow]]:
    """The pair, the gate rows, and one fit per reference scramble.

    The ``glm`` arm is fitted once whatever the budget is: it has no reference block, so a
    scramble does nothing to it, and refitting it per scramble would price a control that
    cannot move.  Gate B is read once per draw, off the first reference arm's recorded state.
    """
    dgp = injection.base_law()
    place = layout(payload, dgp)

    rows: list[FitRow] = []
    risks: list[RiskRow] = []

    started = time.perf_counter()
    try:
        plain = _fit(payload, DRTMLE(**_settings(payload, place.stack.frame)))
        rows.extend(
            _fit_rows(
                payload,
                plain,
                place,
                estimator="glm",
                scramble=0,
                seconds=time.perf_counter() - started,
            )
        )
    except Exception as exc:  # recorded, never swallowed
        rows.extend(_failed_fit(payload, "glm", 0, type(exc).__name__, place.stack.weights.size))

    for index, block in enumerate(place.reference):
        started = time.perf_counter()
        try:
            estimator = RecordingReferenceDRTMLE(
                dgp=dgp,
                reference=REFERENCE,
                window=block.window,
                row_weights=place.stack.weights,
                **_settings(payload, place.stack.frame),
            )
            fit = _fit(payload, estimator)
        except Exception as exc:  # recorded, never swallowed
            rows.extend(
                _failed_fit(
                    payload,
                    "reference",
                    block.seed,
                    type(exc).__name__,
                    place.stack.weights.size,
                )
            )
            continue
        rows.extend(
            _fit_rows(
                payload,
                fit,
                place,
                estimator="reference",
                scramble=block.seed,
                seconds=time.perf_counter() - started,
            )
        )
        if index == 0 and estimator.state is not None:
            try:
                risks.extend(_risk_rows(payload, estimator.state, estimator.bounds, place))
            except Exception as exc:  # pragma: no cover - recorded, never hidden
                print(f"gate B unavailable on {payload.cell} n={payload.n}: {exc!r}")
    return rows, risks


# ------------------------------------------------------------------------------- the tables


def _finite(values: Sequence[float]) -> np.ndarray:
    return np.array([v for v in values if np.isfinite(v)], dtype=float)


def _mean(values: Sequence[float]) -> float:
    finite = _finite(values)
    return float(finite.mean()) if finite.size else float("nan")


def _cells(rows: Sequence[Any]) -> list[tuple[str, int]]:
    return sorted({(row.cell, row.n) for row in rows})


def paired_differences(rows: Sequence[FitRow], cell: str, n: int, estimand: str) -> np.ndarray:
    r"""``root_n_remaining`` at the reference minus at ``glm``, one value per draw.

    Paired on ``data_seed``, and on the **first** reference scramble where a draw carries
    several: gate C's extra scrambles are a budget measurement and folding them into the
    comparison would give a budget draw several times the weight of an ordinary one.
    """

    def select(estimator: str) -> dict[int, float]:
        picked: dict[int, tuple[int, float]] = {}
        for row in rows:
            if (row.cell, row.n, row.estimand, row.estimator) != (cell, n, estimand, estimator):
                continue
            if row.error or not np.isfinite(row.root_n_remaining):
                continue
            seen = picked.get(row.data_seed)
            if seen is None or row.scramble < seen[0]:
                picked[row.data_seed] = (row.scramble, row.root_n_remaining)
        return {seed: value for seed, (_, value) in picked.items()}

    plain, reference = select("glm"), select("reference")
    shared = sorted(set(plain) & set(reference))
    return np.array([reference[seed] - plain[seed] for seed in shared], dtype=float)


def budget_spread(rows: Sequence[FitRow], cell: str, n: int, estimand: str) -> tuple[float, int]:
    r"""Gate C: the reference's own across-scramble sd of the column, and the draws behind it.

    The **within-draw** spread averaged over draws, which is the reference's error *at a fixed
    fit* -- the same decomposition E1b takes for the evaluation rule, and it is a standard
    error under no assumption about a convergence rate because a randomised quasi-Monte Carlo
    rule is unbiased at every point count.

    What it cannot see is the smoothing bias: every scramble shares the knot count and the
    basis, so this spread is orthogonal to a bias in the basis.  Gate B is that half.
    """
    grouped: dict[int, list[float]] = {}
    for row in rows:
        if (row.cell, row.n, row.estimand, row.estimator) != (cell, n, estimand, "reference"):
            continue
        if row.error or not np.isfinite(row.root_n_remaining):
            continue
        grouped.setdefault(row.data_seed, []).append(row.root_n_remaining)
    within = [float(np.var(np.array(v), ddof=1)) for v in grouped.values() if len(v) > 1]
    if not within:
        return (float("nan"), 0)
    return (float(np.sqrt(np.mean(within))), len(within))


def risk_gaps(rows: Sequence[RiskRow], cell: str, n: int, reduction: str) -> dict[str, np.ndarray]:
    """Each candidate's risk minus the reference's, one paired value per draw.

    Averaged over ``(arm, fold)`` **within** a draw before the difference is taken, because
    those are the same regression problem at different splits rather than replicates of one
    number -- and paired against the reference's own risk on the same rows, which is what makes
    the difference a difference of squared weighted errors rather than of two noisy risks.
    """
    by_candidate: dict[str, dict[int, list[float]]] = {}
    for row in rows:
        if (row.cell, row.n, row.reduction) != (cell, n, reduction) or row.error:
            continue
        if not np.isfinite(row.risk):
            continue
        by_candidate.setdefault(row.candidate, {}).setdefault(row.data_seed, []).append(row.risk)
    means = {
        label: {seed: float(np.mean(values)) for seed, values in draws.items()}
        for label, draws in by_candidate.items()
    }
    base = means.get(REFERENCE.label, {})
    return {
        label: np.array(
            [draws[seed] - base[seed] for seed in sorted(set(draws) & set(base))], dtype=float
        )
        for label, draws in means.items()
        if label != REFERENCE.label
    }


def interval(values: np.ndarray, seed: int = 20250801) -> tuple[float, float]:
    """A percentile interval for a mean, resampling **draws** -- the independent unit here."""
    if values.size < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, values.size, (BOOTSTRAP, values.size))
    means = values[picks].mean(axis=1)
    low, high = np.percentile(means, INTERVAL)
    return (float(low), float(high))


#: Headers for :func:`gate_rows`, declared beside it -- the same hazard every harness here
#: guards against, and pinned the same way.
GATE_HEADERS = ("gate", "cell", "n", "reading", "draws")


def gate_rows(fits: Sequence[FitRow], risks: Sequence[RiskRow]) -> list[list[str]]:
    """Gates B and C as readings, printed **before** any paired number.

    Gate A is not here: it is an exact-law control and a test, so it has already either passed
    or turned the suite red before a dispatch could start.

    Every reading is a *difference* with an interval, never a ratio.  A held-out risk carries
    the irreducible ``E_0[w(T - m)^2]`` of its own target, which is common to every candidate
    and can dominate both -- so a ratio of two risks is near one whatever the candidates are,
    and only the difference estimates a difference of squared weighted errors.
    """
    rows: list[list[str]] = []
    for cell, n in _cells(risks):
        for reduction in REDUCTIONS:
            gaps = risk_gaps(risks, cell, n, reduction)
            for label in sorted(gaps):
                values = gaps[label]
                low, high = interval(values)
                rows.append(
                    [
                        f"B. risk vs {label}",
                        cell,
                        f"{n:,}",
                        f"{_mean(values):+.3e} [{low:+.2e}, {high:+.2e}] on {reduction}",
                        str(values.size),
                    ]
                )
    for cell, n in _cells(fits):
        spread, draws = budget_spread(fits, cell, n, "ate")
        rows.append(
            [
                "C. reference sd",
                cell,
                f"{n:,}",
                f"{spread:.4f}" if np.isfinite(spread) else "not measured",
                str(draws),
            ]
        )
    return rows


#: Headers for :func:`comparison_rows`, declared beside it.
COMPARISON_HEADERS = (
    "cell",
    "n",
    "estimand",
    "draws",
    "glm",
    "reference",
    "paired d",
    "d 95%",
    "rule se",
)


def comparison_rows(rows: Sequence[FitRow]) -> list[list[str]]:
    r"""The paired comparison, and it is **not** a verdict.

    ``glm`` and ``reference`` are the mean of :math:`\sqrt n R_{\text{remaining}}` in each arm,
    and ``paired d`` is the mean of the per-draw difference with a bootstrap interval over
    draws.  Read the paired column and not the two levels: the evaluation rule's own error is
    very nearly the whole of a one-replicate study's across-draw spread at :math:`n = 2{,}400`
    (E1b), it is common to the two arms, and it cancels in the difference and not in either
    level.

    ``rule se`` is the evaluation rule's error at a fit, averaged over fits -- the column that
    says how much of the two levels is the instrument.  It is *not* the error of ``paired d``,
    which is the interval beside it.
    """
    out: list[list[str]] = []
    for cell, n in _cells(rows):
        for estimand in ("ate", "ey1", "ey0"):
            differences = paired_differences(rows, cell, n, estimand)
            low, high = interval(differences)
            here = [
                row
                for row in rows
                if (row.cell, row.n, row.estimand) == (cell, n, estimand) and not row.error
            ]
            levels = {
                arm: _mean([row.root_n_remaining for row in here if row.estimator == arm])
                for arm in ("glm", "reference")
            }
            out.append(
                [
                    cell,
                    f"{n:,}",
                    estimand,
                    str(differences.size),
                    f"{levels['glm']:+.4f}",
                    f"{levels['reference']:+.4f}",
                    f"{_mean(differences):+.4f}" if differences.size else "-",
                    f"[{low:+.4f}, {high:+.4f}]" if differences.size > 1 else "-",
                    f"{_mean([row.companion_replicate_se for row in here]):.4f}",
                ]
            )
    return out


#: Headers for :func:`cost_rows`, declared beside it.
COST_HEADERS = ("cell", "n", "companion rows", "fits", "secs/fit", "rounds", "invalid")


def cost_rows(rows: Sequence[FitRow]) -> list[list[str]]:
    """What the design costs, priced in **fits** rather than in draws.

    A draw buys one ``glm`` fit and one reference fit per reference scramble, so a budget draw
    is several times an ordinary one and a table that priced a draw would understate it.
    """
    out: list[list[str]] = []
    for cell, n in _cells(rows):
        here = [row for row in rows if (row.cell, row.n) == (cell, n) and row.estimand == "ate"]
        timed = [row for row in here if np.isfinite(row.seconds)]
        if not timed:
            continue
        out.append(
            [
                cell,
                f"{n:,}",
                f"{int(np.median([row.companion_rows for row in timed])):,}",
                str(len(timed)),
                f"{_mean([row.seconds for row in timed]):.2f}",
                f"{_mean([float(row.rounds) for row in timed]):.1f}",
                str(sum(1 for row in here if row.error or not row.valid)),
            ]
        )
    return out


def write_records(
    fits: Sequence[FitRow], risks: Sequence[RiskRow], directory: Path
) -> tuple[Path, Path]:
    """Two artefacts from one timestamp, so they join.

    The fit rows and the gate rows are different grains -- one per ``(fit, estimand)`` and one
    per ``(candidate, regression, arm, fold)`` -- and a single file at the coarser grain would
    make the gate table unrecomputable from the evidence, which is the whole reason the
    standing decision on manifested rows exists.
    """
    directory.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    paths = (directory / f"{stamp}.jsonl", directory / f"{stamp}-risks.jsonl")
    for path, records in zip(paths, (fits, risks), strict=True):
        with path.open("w") as handle:
            for record in records:
                handle.write(json.dumps(asdict(record)) + "\n")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cells",
        nargs="+",
        default=list(drtmle_injection.CELLS),
        choices=list(drtmle_injection.CELLS),
    )
    parser.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES))
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument(
        "--reference-points",
        type=int,
        default=DEFAULT_REFERENCE_POINTS,
        help="Sobol points the reduced regressions are fitted on. Sized by the reference's "
        "own points-per-parameter budget, which it refuses to be fitted thinner than",
    )
    parser.add_argument(
        "--scoring-points",
        type=int,
        default=DEFAULT_SCORING_POINTS,
        help="Sobol points gate B scores on, from a disjoint scramble stream. Finer than the "
        "fitting block on purpose: a held-out risk carries its own error and nothing pairs "
        "it away",
    )
    parser.add_argument("--evaluation-points", type=int, default=DEFAULT_EVALUATION_POINTS)
    parser.add_argument(
        "--evaluation-scrambles",
        type=int,
        default=DEFAULT_EVALUATION_SCRAMBLES,
        help="independent randomisations of the evaluation rule, averaged into one reading "
        "per fit and shared by both arms so the pairing removes them",
    )
    parser.add_argument(
        "--budget-scrambles",
        type=int,
        default=DEFAULT_BUDGET_SCRAMBLES,
        help="gate C: independent reference scrambles on a budget draw. Each is a REFIT, "
        "unlike an evaluation replicate, which is why it runs on a subset of draws",
    )
    parser.add_argument("--budget-draws", type=int, default=DEFAULT_BUDGET_DRAWS)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20250801)
    parser.add_argument("--tier", type=int, default=2, choices=(1, 2))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmarks/results/drtmle-reference"),
        help="where the per-fit and per-candidate JSONL go; git-ignored generated output",
    )
    args = parser.parse_args()

    global injection
    injection = TIERS[args.tier]

    drawn = np.random.SeedSequence(args.seed).generate_state(2 * args.draws)
    seeds = [
        (int(data), int(fold))
        for data, fold in zip(drawn[: args.draws], drawn[args.draws :], strict=True)
    ]
    payloads = [
        Payload(
            cell=cell,
            n=n,
            data_seed=data_seed,
            fold_seed=fold_seed,
            reference_points=args.reference_points,
            scoring_points=args.scoring_points,
            evaluation_points=args.evaluation_points,
            evaluation_scrambles=args.evaluation_scrambles,
            # The budget draws are the first few of the stream rather than a random subset, so
            # that raising `--draws` cannot change which draws carry it.
            reference_scrambles=args.budget_scrambles if index < args.budget_draws else 1,
        )
        for cell in args.cells
        for n in args.sizes
        for index, (data_seed, fold_seed) in enumerate(seeds)
    ]
    print(
        f"tier {args.tier}: {len(payloads)} draws over cells {list(args.cells)} and sizes "
        f"{list(args.sizes)}, reference {REFERENCE.label} on {args.reference_points:,} points "
        f"against {REDUCED_LEARNER}, scored on {args.scoring_points:,}, evaluated on "
        f"{args.evaluation_scrambles} x {args.evaluation_points:,}, budget "
        f"{args.budget_scrambles} scramble(s) on {args.budget_draws} draw(s), jobs={args.jobs}"
    )

    started = time.perf_counter()
    collected = map_parallel(one_draw, [(payload,) for payload in payloads], n_jobs=args.jobs)
    elapsed = time.perf_counter() - started
    fits = [row for batch, _ in collected for row in batch]
    risks = [row for _, batch in collected for row in batch]
    paths = write_records(fits, risks, args.out)

    def table(title: str, headers: Sequence[str], rows: list[list[str]]) -> None:
        print(f"\n{title}")
        print("=" * len(title))
        print(format_table(list(headers), rows))

    table("The fidelity gates, and read these first", GATE_HEADERS, gate_rows(fits, risks))
    table("The paired comparison", COMPARISON_HEADERS, comparison_rows(fits))
    table("What it cost", COST_HEADERS, cost_rows(fits))

    failures = [row for row in fits if row.error]
    if failures:
        print(f"\n{len(failures)} fit(s) failed: {sorted({row.error for row in failures})}")

    print("\nReading the numbers")
    print("=" * 19)
    print(
        "Read the gate table first, and read it as differences with intervals. `B. risk vs`\n"
        "is a candidate's held-out weighted risk MINUS the shipped reference's, on rows\n"
        "neither saw, per reduced regression. A positive value says the reference is the\n"
        "better estimate of that function; a negative one says a rung beats it. It is a\n"
        "difference and never a ratio, because a risk carries the irreducible variance of\n"
        "its own target and that part is common to every candidate.\n"
        "\n"
        "`C. reference sd` is the reference's own contribution to the column, measured by\n"
        "independent scrambles of the block it is fitted on -- one REFIT each. It is a\n"
        "standard error under no assumption about a rate, and it is blind to the smoothing\n"
        "bias, which every scramble shares. Gate B is that half and neither substitutes.\n"
        "\n"
        "`paired d` is the whole comparison: the reference's column minus `glm`'s, on the\n"
        "same draw, the same split and the same evaluation windows. Read it rather than the\n"
        "two levels beside it -- the evaluation rule's error is most of each level and\n"
        "cancels in the difference.\n"
        "\n"
        "Nothing here is a coverage claim, nothing here selects a learner, and nothing here\n"
        "reads a rate. Item 13 is a rate and closes at E5."
    )
    print(f"\n{len(fits)} fit row(s) and {len(risks)} gate row(s) in {elapsed:.0f}s wall clock.")
    for path in paths:
        print(f"Rows: {path}")


if __name__ == "__main__":
    main()
