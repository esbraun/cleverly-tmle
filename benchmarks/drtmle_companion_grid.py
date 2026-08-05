r"""How much of item 13's column is the instrument, measured by replication.

``docs/roadmap.md``'s **E1b**.  C3c read :math:`\sqrt n R_{\text{remaining}}` as **flat** --
``1.427 / 1.264 / 1.252`` in ``q-drift`` over a fourfold ``n`` -- and the honest statement of
that reading is that a 9--13% decline was measured against Monte Carlo errors of 7--11%.  A
plateau and a slow decline are the same picture at that precision, so item 13 does not close
either way until the two can be told apart.  Part of that error is the estimator's own
sampling spread, which only a replicate count reduces.  Part is the **evaluation rule's**,
which lands directly in every replicate's remainder and is then multiplied by :math:`\sqrt n`.
Nothing separated them, and this module is what does.

**What E1 got wrong here, because this module is the correction.**  Its first version
answered the same question with two statistics that do not answer it, and both are withdrawn
in ``docs/roadmap.md``:

*A successive difference is not an error bound.*  ``delta`` -- the paired movement of the
column between a rung and the next coarser one -- was read as bounding the grid's own error.
It bounds nothing without monotonicity or a convergence result that applies, and Tier 2's
integrand is only piecewise smooth (``_smooth_one``'s kernel cutoff is a jump of
:math:`3.4\times10^{-4}`) so no rate is guaranteed.  Measured on this ladder's own geometry
with such an integrand, the finest rung's ``delta`` ran **four times below** the true error --
and three orders *above* it two rungs earlier.  It is uninformative rather than conservative,
and it is kept below as a **stability** column with that said out loud.

*A ratio of two marginal variances is not a share.*  ``var removed = 1 - s²_grid/s²_draw`` was
read as the fraction of the column that was the evaluation draw.  Writing
:math:`R_r = X + e_r`, the pairing is exact -- the companion is inert to the fit, so both rules
are read through the **same fitted curve** and :math:`\mathrm{Var}(X)` cancels -- and
:math:`E[e_{\text{draw}} \mid \text{fit}] = 0` kills that rule's covariance.  But a *fixed*
grid's error is a deterministic function of the fitted curve, so
:math:`\mathrm{Cov}(X, e_{\text{grid}})` survives and the ratio is not identified.

**What this module does instead is one mechanism.**  The rule is **randomised**: an
independent scramble per replicate, several replicates per fit.  Two things follow, and the
first is why the second is possible.

*The error becomes mean-zero.*  A randomised quasi-Monte Carlo rule is unbiased at every point
count -- the randomisation is over the scramble, not over how many points are taken -- so
:math:`E[e_{\text{grid}} \mid \text{fit}] = 0` exactly as the draw's is, and the attribution
below is identified rather than bounded.

*The error becomes estimable from replication rather than from refinement.*  Several
independent replicates of a rule **at one fit** measure :math:`\mathrm{Var}(e_r \mid
\text{fit})` directly, assuming no rate and needing no model of a witness.  Averaging that
over draws is :math:`\mathrm{Var}(e_r)`, and the share it accounts for follows without any
differencing at all.

**It is still one fit per draw.**  The companion contributes to no fit, no fold and no score,
and every companion prediction is taken row by row -- so every replicate of both rules is a
row block of one stacked companion, addressed by a
:class:`~benchmarks.drtmle_remainder.Window`.  ``tests/unit/test_drtmle_remainder_study.py``
pins a block against the same companion fitted alone, bit for bit, which is the assertion this
design rests on.

**What it measures, and what it deliberately does not.**  It reports each rule's own error and
what share of a column's across-draw variance that error accounts for.  It reports no
coverage, no gate verdict and no learner comparison, and it selects nothing: E1b's scope is
*the precision half of the quadrature question* and a module that also reported a coverage
number would be answering a question its own inputs were not frozen for.

**What this cannot see.**  It is an instrument for the *instrument*: no refinement or
randomisation of a quadrature can detect a defect in the estimator, and a small integration
error says nothing whatever about the remainder.  It is also blind to the law, since every
column integrates ``dgp.propensity`` and ``dgp.outcome_mean`` against predictions of them.
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
# `python benchmarks/drtmle_companion_grid.py` has to find its siblings the way
# `drtmle_coverage.py` does.
if __package__ in (None, ""):  # pragma: no cover - only on the direct-script path
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks import drtmle_injection, drtmle_remainder, drtmle_tier2

#: The two tiers, keyed as ``drtmle_coverage.py`` keys them so one interface serves both.
TIERS = {1: drtmle_injection, 2: drtmle_tier2}

#: Selected at import and replaced by ``main``; the tier is a module of designs rather than a
#: branch, exactly as it is in the coverage harness.
injection: Any = drtmle_injection

#: The reduced-regression learner, and it is **not** a knob here.  E1b selects no learner: the
#: question is how precise the instrument is at the configuration C3c ran, and changing the
#: learner would answer a different one (which is E2's and E2b's).
REDUCED_LEARNER = "glm"

#: The i.i.d. control's seed stream, disjoint from the study's for the reason
#: ``drtmle_coverage.EVALUATION_SEED`` is.
CONTROL_SEED = 91_000_000

#: The scramble stream, disjoint from both of the above for the same reason.  Which scramble
#: a replicate takes must not depend on which rows a draw was fitted on, or the rule's error
#: and the estimator's would share a source.
SCRAMBLE_SEED = 92_000_000

DEFAULT_POINTS = (512, 1_024, 2_048)
DEFAULT_SIZES = (600, 2_400)
DEFAULT_DRAWS = 6

#: Replicates of each rule per fit, which is what makes the error conditional.  Both default
#: to 8: the standard deviation of 8 values carries about a quarter's relative error, which
#: resolves an order-of-magnitude comparison between two rules comfortably and does not
#: pretend to a third digit.
DEFAULT_SCRAMBLES = 8
DEFAULT_DRAW_REPLICATES = 8

#: What the coverage study's own companion held, so the control row is the rule C3c ran at the
#: size C3c ran it -- a comparison against the recorded configuration rather than against a
#: convenient one.
DEFAULT_CONTROL_N = 2_000

#: Resamples behind every reported interval, and the percentiles they are read at.  A
#: bootstrap over **draws**, because the draw is the independent unit here: a fit's replicates
#: are conditional on it and resampling them would report the conditional error as the whole.
BOOTSTRAP = 2_000
INTERVAL = (5.0, 95.0)


@dataclass
class GridRow:
    """One replicate of one rule at one refinement, on one draw.

    Flat and JSON-serialisable, as ``Replicate`` is, and it is the **grain the artefact is
    written at**: every table below is arithmetic on these rows, so a reader with the JSONL
    can recompute any of them.  E1's rows were one per rung and could not express a
    replicate, which is why its error columns had to be derived rather than measured.

    Attributes
    ----------
    rule:
        ``"sobol"`` or ``"draw"``.
    replicate:
        The scramble, or the i.i.d. companion's seed.  Which randomisation produced the row,
        so a spread across rows can be attributed to the rule rather than to anything else.
    points, rows:
        Sobol points and companion rows *in this replicate's block*.  ``points`` is ``0`` on a
        draw row: an i.i.d. sample has rows and no grid, and writing the row count in both
        columns would make a table look like a ladder it is not on.
    p0_curve, remaining, root_n_remaining:
        The three columns item 13 is read from, at this rule, refinement and replicate.
    companion_se:
        The i.i.d. rule's error from the formula :math:`\\sqrt n\\,\\mathrm{sd}(\\hat D)/\\sqrt m`.
        Carried on every row because it is free, and read only on the draw rows -- on a
        quasi-random rule it is an enormous overstatement, and the honest number for that rule
        is the spread across these rows.
    branch_q, branch_g, branch_movement:
        The appendix branches at this refinement, because the binned limits have a grid
        dependence of their own and it is coupled to this one: the cell count is fixed at
        ``BIN_COUNTS`` while the rows per cell grow with the rule.

        ``branch_movement`` is their movement between the two bin counts and is a
        **stability diagnostic rather than an error bound**, for the reason ``delta`` is
        one column to its left: a successive difference between two rungs says a sequence
        settled and not where.  It was called ``branch_error`` until this module's own
        argument was turned on it.
    """

    cell: str
    n: int
    data_seed: int
    fold_seed: int
    rule: str
    replicate: int
    points: int
    rows: int
    p0_curve: float
    remaining: float
    root_n_remaining: float
    companion_se: float
    branch_q: float
    branch_g: float
    branch_movement: float
    seconds: float
    error: str = ""


@dataclass(frozen=True)
class Payload:
    """One draw, read at every rung of the ladder in every replicate of both rules."""

    cell: str
    n: int
    data_seed: int
    fold_seed: int
    points: tuple[int, ...]
    scrambles: int
    control_n: int
    draw_replicates: int


def _fit(payload: Payload, evaluation: Any) -> Any:
    """One ``DRTMLE`` at this draw, against a companion the caller supplies."""
    import warnings

    dgp = injection.base_law()
    frame, _ = dgp.sample(payload.n, seed=payload.data_seed)
    with warnings.catch_warnings():
        # Positivity warnings are per-draw noise here, as they are in the coverage harness.
        warnings.simplefilter("ignore")
        return (
            DRTMLE(
                **injection.settings(payload.cell, payload.n),
                reduced_outcome_learner=REDUCED_LEARNER,
                reduced_treatment_learner=REDUCED_LEARNER,
                random_state=payload.fold_seed,
                evaluation=evaluation,
            )
            .fit(frame, outcome="Y", treatment="A")
            .single()
        )


def _stack(payload: Payload) -> Any:
    """Every replicate of both rules, in one companion.

    The scramble and draw streams are offset by the draw's own seed and are disjoint from
    each other and from the study's, so no replicate of either rule shares a source with the
    rows the fit was taken on.
    """
    offset = payload.data_seed % 1_000_003
    return drtmle_remainder.stacked_companion(
        injection.base_law(),
        points=max(payload.points) if payload.scrambles else 0,
        scrambles=tuple(SCRAMBLE_SEED + offset + i for i in range(payload.scrambles)),
        draw_rows=payload.control_n if payload.draw_replicates else 0,
        draw_seeds=tuple(CONTROL_SEED + offset + i for i in range(payload.draw_replicates)),
    )


def _row(
    payload: Payload,
    fit: Any,
    stack: Any,
    block: Any,
    *,
    points: int,
    seconds: float,
) -> GridRow:
    """One rung of one replicate.

    ``points`` is the rung, which on a Sobol block is a **shorter window with the same
    start** -- the block's own coarser grid -- and on a draw block is the whole block.  The
    truth travels with the window: :math:`\\psi_0` at this replicate's scramble and this
    rung's point count, since the cancellation :func:`truth_at` documents is within a
    replicate.
    """
    dgp = injection.base_law()
    sobol = block.rule == "sobol"
    window = block.window.head(2 * points) if sobol else block.window
    truth = drtmle_remainder.truth_at(dgp, points, scramble=block.seed) if sobol else None
    (measured,) = [
        row
        for row in drtmle_remainder.remainder_rows(
            fit,
            dgp,
            n=payload.n,
            bounds=fit.config.g_bounds,
            row_weights=stack.weights,
            window=window,
            truth=truth,
        )
        if row.estimand == "ate"
    ]
    return GridRow(
        cell=payload.cell,
        n=payload.n,
        data_seed=payload.data_seed,
        fold_seed=payload.fold_seed,
        rule=block.rule,
        replicate=block.seed,
        points=points if sobol else 0,
        rows=window.rows,
        p0_curve=measured.p0_curve,
        remaining=measured.remaining,
        root_n_remaining=measured.root_n_remaining,
        companion_se=measured.companion_se,
        branch_q=measured.branch_q,
        branch_g=measured.branch_g,
        branch_movement=measured.branch_movement,
        seconds=seconds,
    )


def one_draw(payload: Payload) -> list[GridRow]:
    """Every rung of every replicate of both rules, on one draw, off **one fit**.

    The fit count is the design: a refit per replicate would cost one fit each *and* would
    have to be argued bit-identical before a spread across them could be called the rule's.
    An exception is recorded as a row with its type in ``error`` rather than dropped -- a
    replicate that could not be computed is a gap in the evidence and has to look like one.
    """
    stack = _stack(payload)
    started = time.perf_counter()
    try:
        fit = _fit(payload, stack.frame)
    except Exception as exc:  # pragma: no cover - recorded, never swallowed
        return [
            _failed(payload, block, max(payload.points), type(exc).__name__)
            for block in stack.blocks
        ]
    seconds = time.perf_counter() - started

    out: list[GridRow] = []
    for block in stack.blocks:
        rungs = sorted(payload.points) if block.rule == "sobol" else [0]
        for points in rungs:
            try:
                out.append(_row(payload, fit, stack, block, points=points, seconds=seconds))
            except Exception as exc:  # pragma: no cover - recorded, never swallowed
                out.append(_failed(payload, block, points, type(exc).__name__))
    return out


def _failed(payload: Payload, block: Any, points: int, error: str) -> GridRow:
    nan = float("nan")
    return GridRow(
        cell=payload.cell,
        n=payload.n,
        data_seed=payload.data_seed,
        fold_seed=payload.fold_seed,
        rule=block.rule,
        replicate=block.seed,
        points=points if block.rule == "sobol" else 0,
        rows=block.window.rows,
        p0_curve=nan,
        remaining=nan,
        root_n_remaining=nan,
        companion_se=nan,
        branch_q=nan,
        branch_g=nan,
        branch_movement=nan,
        seconds=nan,
        error=error,
    )


# ------------------------------------------------------------------ the decomposition


@dataclass(frozen=True)
class Decomposition:
    r"""One rule's error and the share of a column's variance it accounts for.

    The whole of E1b's arithmetic, and it is deliberately three lines rather than a ratio of
    two spreads.  With :math:`R` replicates of a rule at each of :math:`D` fits, and
    :math:`v_{d,i}` the column at replicate :math:`i` of fit :math:`d`:

    .. math::

        \widehat{\mathrm{Var}}(e) &= \operatorname{mean}_d
                 \operatorname{Var}_i (v_{d,i}) \\
        \widehat{\mathrm{Var}}(X) &= \operatorname{Var}_d
                 (\bar v_{d}) - \widehat{\mathrm{Var}}(e) / R \\
        \text{share} &= \widehat{\mathrm{Var}}(e) /
                 \bigl(\widehat{\mathrm{Var}}(X) + \widehat{\mathrm{Var}}(e)\bigr).

    The first line is what replication buys and is why nothing here is differenced: it is the
    rule's error *at a fixed fit*, so no assumption about how the rule's error relates to the
    estimator's is needed.  The second is the estimator's own variance, and because it is
    computed for **both** rules from data that share it, the two readings of it are a free
    consistency check -- they estimate one quantity and are printed so a reader can see them
    agree.  The third is the share of a *one-replicate* study's across-draw variance that the
    rule accounts for, which is what E5 sizes a replicate count against.

    ``estimator`` may come out slightly negative when a rule's error dominates and :math:`D`
    is small; it is reported as measured rather than clipped, because a negative variance
    estimate is a count too small and clipping it would hide that.
    """

    rule_variance: float
    estimator_variance: float
    replicates: int
    draws: int

    @property
    def spread(self) -> float:
        """The across-draw sd a **one-replicate** study of this rule would see."""
        return float(np.sqrt(max(self.estimator_variance + self.rule_variance, 0.0)))

    @property
    def rule_sd(self) -> float:
        return float(np.sqrt(max(self.rule_variance, 0.0)))

    @property
    def share(self) -> float:
        total = self.estimator_variance + self.rule_variance
        return float(self.rule_variance / total) if total > 0 else float("nan")


def decompose(values: Sequence[Sequence[float]]) -> Decomposition:
    """The variance decomposition, from one list of replicate values per draw.

    Draws contributing fewer than two finite replicates are dropped from the within-fit term
    and kept in the between-fit one, which is the honest reading of a partial draw: its mean
    is still an estimate of that fit's column and its spread is not estimable.
    """
    per_draw = [np.array([v for v in row if np.isfinite(v)], dtype=float) for row in values]
    usable = [row for row in per_draw if row.size]
    if len(usable) < 2:
        return Decomposition(float("nan"), float("nan"), 0, len(usable))
    within = [float(np.var(row, ddof=1)) for row in usable if row.size > 1]
    replicates = int(np.median([row.size for row in usable]))
    rule = float(np.mean(within)) if within else float("nan")
    between = float(np.var([float(row.mean()) for row in usable], ddof=1))
    estimator = between - (rule / replicates if np.isfinite(rule) and replicates else 0.0)
    return Decomposition(rule, estimator, replicates, len(usable))


def bootstrap_share(values: Sequence[Sequence[float]], seed: int = 20250801) -> tuple[float, float]:
    """A percentile interval for :attr:`Decomposition.share`, resampling **draws**.

    The draw is the independent unit: a fit's replicates are conditional on it, so resampling
    them would treat the conditional error as the whole and report an interval far too
    narrow.  Reported on every share, because a share printed without one is what E1 printed.
    """
    rng = np.random.default_rng(seed)
    rows = list(values)
    if len(rows) < 2:
        return (float("nan"), float("nan"))
    shares = []
    for _ in range(BOOTSTRAP):
        picked = rng.integers(0, len(rows), len(rows))
        share = decompose([rows[i] for i in picked]).share
        if np.isfinite(share):
            shares.append(share)
    if not shares:
        return (float("nan"), float("nan"))
    low, high = np.percentile(shares, INTERVAL)
    return (float(low), float(high))


# ------------------------------------------------------------------------------ the tables

#: Headers for :func:`grid_rows`, declared beside it -- the same hazard
#: ``drtmle_coverage.py`` guards against, and pinned the same way.
GRID_HEADERS = (
    "cell",
    "n",
    "rule",
    "points",
    "rows",
    "draws",
    "reps",
    "P0 D-hat",
    "delta",
    "sqrt(n) R_rem",
    "spread",
    "rule sd",
    "share",
    "share 90%",
    "est sd",
    "R_Q",
    "R_g",
    "branch move",
)


def _mean(values: Sequence[float]) -> float:
    finite = np.array([v for v in values if np.isfinite(v)], dtype=float)
    return float(finite.mean()) if finite.size else float("nan")


def _replicates(records: Sequence[GridRow], rule: str, points: int) -> list[list[float]]:
    """``root_n_remaining`` per replicate, grouped by draw -- what :func:`decompose` reads."""
    grouped: dict[int, list[float]] = {}
    for row in records:
        if row.rule == rule and row.points == points and not row.error:
            grouped.setdefault(row.data_seed, []).append(row.root_n_remaining)
    return [grouped[seed] for seed in sorted(grouped)]


def grid_rows(records: Sequence[GridRow]) -> list[list[str]]:
    r"""One row per rung, with the control beneath each cell's rungs.

    **``share`` is the measurement and it is conditional.**  The rule's own error is the
    spread of the column across that rule's independent replicates **at a fixed fit**, which
    is a standard deviation of a randomised rule and needs no model of a witness and no
    assumption about a convergence rate.  ``share`` is what fraction of a one-replicate
    study's across-draw variance that error accounts for -- for the draw row, the number E5
    sizes a replicate count against.  :class:`Decomposition` is the arithmetic and
    ``share 90%`` is a bootstrap over draws, because a share printed without an interval is
    what this module printed before.

    ``est sd`` is the estimator's own across-draw spread implied by the same decomposition.
    Both rules estimate it from the same fits, so **the two rows should agree**, and a
    disagreement is a count too small rather than a finding -- it is printed for exactly that
    reason.

    ``delta`` is the mean absolute movement of :math:`\sqrt n P_0\hat D` between this rung and
    the next coarser one, paired within a draw **and within a scramble**.  It is a
    **stability diagnostic and not an error bound**: a successive difference tells you a
    sequence settled, not where, and measured on a piecewise-smooth integrand at this
    geometry it ran four times below the true error at the finest rung.  Read ``rule sd``
    for the error and read this only for whether the rule is still moving.

    ``spread`` is the across-draw standard deviation of ``sqrt(n) R_rem`` a **one-replicate**
    study of this rule would see, which is the column C3c reported and the one E5 will.

    Reported and not thresholded.  §5 sets no number for any of these and E1b is not the
    place to invent one -- a rule frozen after a number exists is the thing the coverage
    study's own discipline refuses.
    """
    rows: list[list[str]] = []
    for cell, n in sorted({(r.cell, r.n) for r in records}):
        here = [r for r in records if r.cell == cell and r.n == n]
        ladder = sorted({r.points for r in here if r.rule == "sobol"})
        previous: int | None = None
        for points in ladder:
            rows.append(_rung(here, cell, n, "sobol", points, previous))
            previous = points
        if any(r.rule == "draw" for r in here):
            rows.append(_rung(here, cell, n, "draw", 0, None))
    return rows


def _rung(
    records: Sequence[GridRow],
    cell: str,
    n: int,
    rule: str,
    points: int,
    previous: int | None,
) -> list[str]:
    selected = [r for r in records if r.rule == rule and r.points == points and not r.error]
    if not selected:
        return [cell, f"{n:,}", rule, f"{points:,}" if points else "-", "-", "0", "0"] + ["-"] * 11
    grouped = _replicates(records, rule, points)
    decomposed = decompose(grouped)
    low, high = bootstrap_share(grouped)

    delta = "-"
    if previous is not None:
        coarser = {
            (r.data_seed, r.replicate): r
            for r in records
            if r.rule == "sobol" and r.points == previous and not r.error
        }
        # Paired within the draw *and* within the scramble: two rungs of one block are the
        # same fit at two grids, and pairing across scrambles would put the rule's own error
        # into a column that is supposed to be about refinement alone.
        moved = [
            abs(r.root_n_remaining - coarser[(r.data_seed, r.replicate)].root_n_remaining)
            for r in selected
            if (r.data_seed, r.replicate) in coarser
        ]
        delta = f"{_mean(moved):.5f}" if moved else "-"

    return [
        cell,
        f"{n:,}",
        rule,
        f"{points:,}" if points else "-",
        f"{selected[0].rows:,}",
        str(decomposed.draws),
        str(decomposed.replicates),
        f"{_mean([r.p0_curve for r in selected]):+.5f}",
        delta,
        f"{_mean([r.root_n_remaining for r in selected]):+.4f}",
        f"{decomposed.spread:.4f}",
        f"{decomposed.rule_sd:.4f}",
        f"{decomposed.share:.3f}" if np.isfinite(decomposed.share) else "-",
        f"[{low:.2f}, {high:.2f}]" if np.isfinite(low) else "-",
        f"{np.sqrt(max(decomposed.estimator_variance, 0.0)):.4f}",
        f"{_mean([r.branch_q for r in selected]):+.5f}",
        f"{_mean([r.branch_g for r in selected]):+.5f}",
        f"{_mean([r.branch_movement for r in selected]):.5f}",
    ]


#: Headers for :func:`cost_rows`, declared beside it.
COST_HEADERS = ("cell", "n", "rows", "fits", "secs/fit")


def cost_rows(records: Sequence[GridRow]) -> list[list[str]]:
    """What the design costs, because every replicate is paid for in companion predictions.

    One row per cell and size rather than one per rule: the fit is one, its companion holds
    every replicate of both rules, and the rungs are windows -- so what a dispatch actually
    buys is a single number against a single row count.
    """
    rows: list[list[str]] = []
    for cell, n in sorted({(r.cell, r.n) for r in records}):
        here = [r for r in records if r.cell == cell and r.n == n and np.isfinite(r.seconds)]
        if not here:
            continue
        per_draw = {r.data_seed: r.seconds for r in here}
        # The companion **one fit** paid for, so the widest window per block within a single
        # draw. Summing across draws would multiply it by the draw count -- and a replicate
        # seed is offset by its draw's, so `(rule, replicate)` is not a key that collapses
        # them. What this column prices is a fit, which is what a dispatch is billed in.
        one = min(per_draw)
        widest: dict[tuple[str, int], int] = {}
        for r in here:
            if r.data_seed != one:
                continue
            key = (r.rule, r.replicate)
            widest[key] = max(widest.get(key, 0), r.rows)
        rows.append(
            [
                cell,
                f"{n:,}",
                f"{sum(widest.values()):,}",
                str(len(per_draw)),
                f"{_mean(list(per_draw.values())):.2f}",
            ]
        )
    return rows


def write_records(records: Sequence[GridRow], directory: Path) -> Path:
    """Every replicate of every rung of every draw, one JSON object per line.

    The artefact is at the **replicate** grain, so every table above is recomputable from it
    and a reader need not take a summary's word for a decomposition.  That is E1b's third
    correction: E1's numbers came from a sandbox whose rows were git-ignored and are gone.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{time.strftime('%Y%m%dT%H%M%S')}.jsonl"
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record)) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cells",
        nargs="+",
        default=list(drtmle_injection.CELLS),
        choices=list(drtmle_injection.CELLS),
    )
    parser.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES))
    parser.add_argument(
        "--points",
        type=int,
        nargs="+",
        default=list(DEFAULT_POINTS),
        help="the ladder, in Sobol points; each must be a power of two so the grids nest, "
        "and every rung but the finest is read as a window on one fit",
    )
    parser.add_argument(
        "--scrambles",
        type=int,
        default=DEFAULT_SCRAMBLES,
        help="independent randomisations of the quasi-random rule per fit. This is what "
        "measures that rule's own error; below 2 there is no spread and the column is nan",
    )
    parser.add_argument(
        "--control-n",
        type=int,
        default=DEFAULT_CONTROL_N,
        help="rows of each i.i.d. companion the ladder is read against; 0 drops the control, "
        "which leaves the ladder without the rule it is a comparison to",
    )
    parser.add_argument(
        "--draw-replicates",
        type=int,
        default=DEFAULT_DRAW_REPLICATES,
        help="independent i.i.d. companions per fit, which measure that rule's error the "
        "same way rather than through the sd(D)/sqrt(m) formula",
    )
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20250801)
    parser.add_argument("--tier", type=int, default=2, choices=(1, 2))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmarks/results/drtmle-companion-grid"),
        help="where the per-replicate JSONL goes; git-ignored generated output",
    )
    args = parser.parse_args()

    global injection
    injection = TIERS[args.tier]

    drawn = np.random.SeedSequence(args.seed).generate_state(2 * args.draws)
    seeds = [
        (int(data), int(fold))
        for data, fold in zip(drawn[: args.draws], drawn[args.draws :], strict=True)
    ]
    points = tuple(sorted(args.points))
    payloads = [
        Payload(
            cell,
            n,
            data_seed,
            fold_seed,
            points,
            args.scrambles,
            args.control_n,
            args.draw_replicates,
        )
        for cell in args.cells
        for n in args.sizes
        for data_seed, fold_seed in seeds
    ]
    companion_rows = args.scrambles * 2 * points[-1] + args.draw_replicates * args.control_n
    print(
        f"tier {args.tier}: {len(payloads)} draws over cells {list(args.cells)} and sizes "
        f"{list(args.sizes)}, ladder {list(points)} points, {args.scrambles} scramble(s) and "
        f"{args.draw_replicates} draw replicate(s) of {args.control_n:,} rows "
        f"({companion_rows:,} companion rows a fit), jobs={args.jobs}"
    )

    started = time.perf_counter()
    collected = map_parallel(one_draw, [(payload,) for payload in payloads], n_jobs=args.jobs)
    elapsed = time.perf_counter() - started
    records = [record for batch in collected for record in batch]
    path = write_records(records, args.out)

    def table(title: str, headers: Sequence[str], rows: list[list[str]]) -> None:
        print(f"\n{title}")
        print("=" * len(title))
        print(format_table(list(headers), rows))

    table("Each rule's own error, and the share it accounts for", GRID_HEADERS, grid_rows(records))
    table("What it cost", COST_HEADERS, cost_rows(records))

    failures = [r for r in records if r.error]
    if failures:
        print(f"\n{len(failures)} replicate(s) failed: {sorted({r.error for r in failures})}")

    print("\nReading the numbers")
    print("=" * 19)
    print(
        "`rule sd` is the rule's own error, measured: the spread of the column across that\n"
        "rule's independent replicates at a *fixed fit*. It assumes no convergence rate and\n"
        "carries no model of a witness, which is what a standard error is and what a\n"
        "successive difference between two rungs is not.\n"
        "\n"
        "`share` is the fraction of a one-replicate study's across-draw variance that this\n"
        "rule accounts for -- Var(e) / (Var(X) + Var(e)), both estimated rather than one\n"
        "differenced out of the other. On the `draw` row it is what E5 sizes a replicate\n"
        "count against. `share 90%` is a bootstrap over *draws*, since a fit's replicates are\n"
        "conditional on its draw; read the interval, not the point.\n"
        "\n"
        "`est sd` is the estimator's own spread implied by the same decomposition, and both\n"
        "rules estimate it from the same fits. The two rows agreeing is the consistency\n"
        "check; the two disagreeing is a draw count rather than a finding. Where that term\n"
        "comes out negative -- which it can, at few draws, when a rule's error dominates --\n"
        "it is printed as measured and a `share` interval can then reach past one. Both are\n"
        "the same statement: too few draws. Neither is clipped, because clipping would make\n"
        "an unresolved reading look like a resolved one.\n"
        "\n"
        "`delta` is a STABILITY diagnostic and not an error bound. It says the sequence\n"
        "settled between two rungs; it does not say where. Measured on a piecewise-smooth\n"
        "integrand at this geometry, the finest rung's delta ran four times below the true\n"
        "error -- and three orders above it two rungs earlier. It is here to show a rule\n"
        "still moving, and `rule sd` is the column that says how large its error is.\n"
        "\n"
        "`branch move` is the same shape of statistic and carries the same caveat, one\n"
        "level down: the binned limits' movement between the two bin counts. It has no\n"
        "`rule sd` beside it, because randomising a scramble makes a QUADRATURE error\n"
        "mean-zero and does nothing to a SMOOTHING bias -- a regressogram's bias is stable\n"
        "across two resolutions and can be large at both. So a settled branch's error is\n"
        "unestablished rather than small, and establishing it is what E2 is for.\n"
        "\n"
        "The randomised scramble is what makes any of this identified: it puts the grid's\n"
        "error at mean zero given the fit, so the rule's contribution neither biases the\n"
        "study's mean nor covaries with the remainder. It also averages down over a study,\n"
        "which a fixed grid's does not.\n"
        "\n"
        "Nothing here is a coverage claim and nothing here selects a learner. A rule's own\n"
        "error, however well measured, says nothing whatever about the remainder."
    )
    print(f"\n{len(records)} replicate rows in {elapsed:.0f}s wall clock at jobs={args.jobs}.")
    print(f"Per-replicate rows: {path}")


if __name__ == "__main__":
    main()
