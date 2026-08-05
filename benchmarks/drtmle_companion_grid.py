r"""How much of item 13's column is the instrument: the companion's own error, on a ladder.

``docs/roadmap.md``'s **E1**.  C3c read :math:`\sqrt n R_{\text{remaining}}` as **flat** --
``1.427 / 1.264 / 1.252`` in ``q-drift`` over a fourfold ``n`` -- and the honest statement of
that reading is that a 9--13% decline was measured against Monte Carlo errors of 7--11%.  A
plateau and a slow decline are the same picture at that precision, so item 13 does not close
either way until the two can be told apart.  Part of that error is the estimator's own
sampling spread, which only a replicate count reduces.  Part is the **evaluation rule's**,
which lands directly in every replicate's remainder and is then multiplied by :math:`\sqrt n`.
Nothing separated them, and this module is what does.

**What it measures, and what it deliberately does not.**  It reports each rule's own error
and how that error falls as the rule is refined.  It reports no coverage, no gate verdict and
no learner comparison, and it selects nothing: E1's scope is *the precision half of the
quadrature question* and a module that also reported a coverage number would be answering a
question its own inputs were not frozen for.

**The ladder is one fit, not one fit per rung.**
:func:`~benchmarks.drtmle_remainder.quadrature_frame` interleaves the arms and
:meth:`~cleverly.datasets.DGP.quadrature`'s grids are nested prefixes, so the first
:math:`2k` rows of a companion **are** the grid at :math:`k` points.  A rung is therefore
``limit=`` on one fit, and the movement between two rungs *is* the discretisation rather than
a difference between two fits that would have to be argued bit-identical first.
``tests/unit/test_drtmle_remainder_study.py`` pins the prefix against an actual refit.

**Two rules, and they fail differently -- which is why the control row is not optional.**

*The i.i.d. draw* carries :math:`\mathrm{sd}(\hat D)/\sqrt m`, which is **noise**: independent
across replicates, so it averages down in a study's reported mean and inflates the spread
each entry's Monte Carlo error is computed from.

*The deterministic grid* carries a Sobol discretisation, which is a **bias**: the same points
at every replicate, so a study cannot average it down at all.  It is orders smaller, and
"orders smaller" is a claim that needs a number rather than an argument -- which is what the
``delta`` column is.  A ladder that has not flattened is a grid still moving, and the right
response is a finer one rather than a footnote.

**What this cannot see.**  It is an instrument for the *instrument*: no refinement of a
quadrature can detect a defect in the estimator, and a flat ladder bounds the rule's error
and says nothing whatever about the remainder.  It is also blind to the law, since every
column integrates ``dgp.propensity`` and ``dgp.outcome_mean`` against predictions of them.
And Tier 2's integrand is only piecewise smooth -- ``_smooth_one``'s kernel cutoff is a jump
of :math:`3.4\times10^{-4}` and its empty-neighbourhood branch is another -- so Sobol's rate
is not guaranteed there.  The ladder measures the consequence rather than assuming it away.
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

#: The reduced-regression learner, and it is **not** a knob here.  E1 selects no learner: the
#: question is how precise the instrument is at the configuration C3c ran, and changing the
#: learner would answer a different one (which is E2's and E2b's).
REDUCED_LEARNER = "glm"

#: The i.i.d. control's seed stream, disjoint from the study's for the reason
#: ``drtmle_coverage.EVALUATION_SEED`` is.
CONTROL_SEED = 91_000_000

DEFAULT_POINTS = (512, 1_024, 2_048, 4_096)
DEFAULT_SIZES = (600, 2_400)
DEFAULT_DRAWS = 6

#: What the coverage study's own companion held, so the control row is the rule C3c ran at the
#: size C3c ran it -- a comparison against the recorded configuration rather than against a
#: convenient one.
DEFAULT_CONTROL_N = 2_000


@dataclass
class GridRow:
    """One rule at one refinement, on one draw.  Flat and JSON-serialisable, as ``Replicate`` is.

    Attributes
    ----------
    rule:
        ``"sobol"`` or ``"draw"``.  The control rows carry the second, at ``--control-n``.
    points, rows:
        Sobol points and companion rows.  ``points`` is ``0`` on a control row: an i.i.d.
        draw has rows and no grid, and writing the row count in both columns would make a
        table look like a ladder it is not on.
    p0_curve, remaining, root_n_remaining:
        The three columns item 13 is read from, at this rule and this refinement.
    companion_se, companion_halving:
        The rule's own error, as :class:`~benchmarks.drtmle_remainder.RemainderRow` defines
        them.  ``companion_se`` belongs to the draw and ``companion_halving`` to the grid.
    branch_q, branch_g, branch_error:
        The appendix branches at this refinement, because the binned limits have a grid
        dependence of their own and it is coupled to this one: the cell count is fixed at
        ``BIN_COUNTS`` while the rows per cell grow with the rule.
    """

    cell: str
    n: int
    data_seed: int
    fold_seed: int
    rule: str
    points: int
    rows: int
    p0_curve: float
    remaining: float
    root_n_remaining: float
    companion_se: float
    companion_halving: float
    branch_q: float
    branch_g: float
    branch_error: float
    seconds: float
    error: str = ""


@dataclass(frozen=True)
class Payload:
    """One draw, read at every rung of the ladder and once at the control rule."""

    cell: str
    n: int
    data_seed: int
    fold_seed: int
    points: tuple[int, ...]
    control_n: int


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


def _row(
    payload: Payload,
    fit: Any,
    *,
    rule: str,
    points: int,
    rows: int,
    weights: Any,
    seconds: float,
) -> GridRow:
    dgp = injection.base_law()
    truth = drtmle_remainder.truth_at(dgp, points) if weights is not None else None
    (measured,) = [
        row
        for row in drtmle_remainder.remainder_rows(
            fit,
            dgp,
            n=payload.n,
            bounds=fit.config.g_bounds,
            row_weights=weights,
            limit=rows if weights is not None else None,
            truth=truth,
        )
        if row.estimand == "ate"
    ]
    return GridRow(
        cell=payload.cell,
        n=payload.n,
        data_seed=payload.data_seed,
        fold_seed=payload.fold_seed,
        rule=rule,
        points=points,
        rows=rows,
        p0_curve=measured.p0_curve,
        remaining=measured.remaining,
        root_n_remaining=measured.root_n_remaining,
        companion_se=measured.companion_se,
        companion_halving=measured.companion_halving,
        branch_q=measured.branch_q,
        branch_g=measured.branch_g,
        branch_error=measured.branch_error,
        seconds=seconds,
    )


def one_draw(payload: Payload) -> list[GridRow]:
    """Every rung of the ladder, plus the control, on one draw.

    **Two fits and not** ``len(points) + 1``: the grid rungs are prefixes of the finest
    companion, so one fit carries all of them, and only the i.i.d. control needs a companion
    of its own.  An exception is recorded as a row with its type in ``error`` rather than
    dropped -- a rung that could not be computed is a gap in the ladder and has to look like
    one.
    """
    dgp = injection.base_law()
    finest = max(payload.points)
    out: list[GridRow] = []

    frame, weights = drtmle_remainder.quadrature_frame(dgp, finest)
    started = time.perf_counter()
    try:
        fit = _fit(payload, frame)
    except Exception as exc:  # pragma: no cover - recorded, never swallowed
        return [_failed(payload, "sobol", finest, type(exc).__name__)]
    seconds = time.perf_counter() - started
    for points in sorted(payload.points):
        try:
            # The **whole** weight vector, with `limit` doing the slicing: every integrator
            # checks its weights against the companion's row count, and a pre-sliced vector
            # is exactly the stale-weights mistake that check exists to catch.
            out.append(
                _row(
                    payload,
                    fit,
                    rule="sobol",
                    points=points,
                    rows=2 * points,
                    weights=weights,
                    seconds=seconds,
                )
            )
        except Exception as exc:  # pragma: no cover - recorded, never swallowed
            out.append(_failed(payload, "sobol", points, type(exc).__name__))

    if payload.control_n > 0:
        started = time.perf_counter()
        try:
            control = _fit(
                payload,
                drtmle_remainder.evaluation_frame(
                    dgp, payload.control_n, CONTROL_SEED + payload.data_seed % 1_000_003
                ),
            )
            out.append(
                _row(
                    payload,
                    control,
                    rule="draw",
                    points=0,
                    rows=payload.control_n,
                    weights=None,
                    seconds=time.perf_counter() - started,
                )
            )
        except Exception as exc:  # pragma: no cover - recorded, never swallowed
            out.append(_failed(payload, "draw", 0, type(exc).__name__))
    return out


def _failed(payload: Payload, rule: str, points: int, error: str) -> GridRow:
    nan = float("nan")
    return GridRow(
        cell=payload.cell,
        n=payload.n,
        data_seed=payload.data_seed,
        fold_seed=payload.fold_seed,
        rule=rule,
        points=points,
        rows=2 * points,
        p0_curve=nan,
        remaining=nan,
        root_n_remaining=nan,
        companion_se=nan,
        companion_halving=nan,
        branch_q=nan,
        branch_g=nan,
        branch_error=nan,
        seconds=nan,
        error=error,
    )


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
    "P0 D-hat",
    "delta",
    "sqrt(n) R_rem",
    "spread",
    "rule err",
    "var removed",
    "R_Q",
    "R_g",
    "branch err",
)


def _mean(values: Sequence[float]) -> float:
    finite = np.array([v for v in values if np.isfinite(v)], dtype=float)
    return float(finite.mean()) if finite.size else float("nan")


def grid_rows(records: Sequence[GridRow]) -> list[list[str]]:
    r"""The ladder, one row per rung, with the control beneath each cell's rungs.

    **``delta`` is the demonstration.**  It is the mean absolute movement of
    :math:`\sqrt n P_0\hat D` between this rung and the next coarser one, paired within each
    draw -- so it is the rule's error read off the ladder rather than asserted, in the same
    idiom ``BIN_COUNTS`` reports the binned limits' error in.  A rung whose ``delta`` has not
    collapsed is a rule still moving.  The coarsest rung has no predecessor and reads ``-``.

    ``spread`` is the standard deviation of ``sqrt(n) R_rem`` **across draws**, which carries
    the estimator's own sampling variation *and* the rule's error together, and ``rule err``
    is the rule's own witness -- the movement when half the rows are dropped, averaged over
    draws.  On the deterministic grid that is a fair reading of the discretisation.  **On the
    i.i.d. draw it is not**: halving a noise-dominated rule doubles a variance, so the witness
    reads about :math:`1.4\times` the standard error it is standing in for, which is why
    ``rule err`` on a control row can exceed ``spread`` and must not be read as a share.

    ``var removed`` is the honest form of that share and it needs no model of the witness at
    all: :math:`1 - s^2 / s_{\text{control}}^2`, the fraction of the **control rule's**
    across-draw variance this rung removes.  It is a difference of two measured spreads on
    the *same draws* through the *same primary fits* -- the companion is inert to the fit --
    so the estimator's own contribution is common to both and what is left is the rule.  That
    is the number E5 sizes a replicate count against: how much of C3c's :math:`\pm 0.09` was
    the instrument.  ``-`` on the control row and wherever there is no control to compare to.

    Reported and not thresholded.  §5 sets no number for any of these and E1 is not the place
    to invent one -- a rule frozen after a number exists is the thing the coverage study's own
    discipline refuses.
    """
    rows: list[list[str]] = []
    for cell, n in sorted({(r.cell, r.n) for r in records}):
        here = [r for r in records if r.cell == cell and r.n == n]
        control = _spread(here, "draw", 0)
        ladder = sorted({r.points for r in here if r.rule == "sobol"})
        previous: int | None = None
        for points in ladder:
            rows.append(_rung(here, cell, n, "sobol", points, previous, control))
            previous = points
        if any(r.rule == "draw" for r in here):
            rows.append(_rung(here, cell, n, "draw", 0, None, float("nan")))
    return rows


def _spread(here: Sequence[GridRow], rule: str, points: int) -> float:
    """The across-draw standard deviation of ``sqrt(n) R_rem`` at one rung."""
    values = np.array(
        [r.root_n_remaining for r in here if r.rule == rule and r.points == points and not r.error],
        dtype=float,
    )
    finite = values[np.isfinite(values)]
    return float(np.std(finite, ddof=1)) if finite.size > 1 else float("nan")


def _rung(
    here: Sequence[GridRow],
    cell: str,
    n: int,
    rule: str,
    points: int,
    previous: int | None,
    control: float,
) -> list[str]:
    selected = [r for r in here if r.rule == rule and r.points == points and not r.error]
    if not selected:
        return [cell, f"{n:,}", rule, f"{points:,}", "-", "0"] + ["-"] * 9
    spread = _spread(here, rule, points)
    rule_error = _mean([r.companion_halving for r in selected])
    removed = (
        1.0 - spread**2 / control**2
        if np.isfinite(spread) and np.isfinite(control) and control > 0
        else float("nan")
    )

    delta = "-"
    if previous is not None:
        coarser = {r.data_seed: r for r in here if r.rule == "sobol" and r.points == previous}
        # Paired within the draw, since the two rungs are the *same fit* read at two grids and
        # an unpaired difference would carry the between-draw spread the pairing removes.
        moved = [
            abs(r.root_n_remaining - coarser[r.data_seed].root_n_remaining)
            for r in selected
            if r.data_seed in coarser
        ]
        delta = f"{_mean(moved):.5f}" if moved else "-"

    return [
        cell,
        f"{n:,}",
        rule,
        f"{points:,}" if points else "-",
        f"{selected[0].rows:,}",
        str(len(selected)),
        f"{_mean([r.p0_curve for r in selected]):+.5f}",
        delta,
        f"{_mean([r.root_n_remaining for r in selected]):+.4f}",
        f"{spread:.4f}",
        f"{rule_error:.5f}",
        f"{removed:.3f}" if np.isfinite(removed) else "-",
        f"{_mean([r.branch_q for r in selected]):+.5f}",
        f"{_mean([r.branch_g for r in selected]):+.5f}",
        f"{_mean([r.branch_error for r in selected]):.5f}",
    ]


#: Headers for :func:`cost_rows`, declared beside it.
COST_HEADERS = ("cell", "n", "rule", "rows", "fits", "secs/fit")


def cost_rows(records: Sequence[GridRow]) -> list[list[str]]:
    """What each rule costs, because a grid is paid for in companion predictions.

    A rung is free -- the ladder is prefixes of one fit -- so what this prices is the
    *finest* grid against the control draw, which is the choice a dispatch actually makes.
    """
    rows: list[list[str]] = []
    for cell, n in sorted({(r.cell, r.n) for r in records}):
        for rule in ("sobol", "draw"):
            here = [
                r
                for r in records
                if r.cell == cell and r.n == n and r.rule == rule and np.isfinite(r.seconds)
            ]
            if rule == "sobol":
                finest = max((r.points for r in here), default=0)
                here = [r for r in here if r.points == finest]
            if not here:
                continue
            rows.append(
                [
                    cell,
                    f"{n:,}",
                    rule,
                    f"{here[0].rows:,}",
                    str(len({r.data_seed for r in here})),
                    f"{_mean([r.seconds for r in here]):.2f}",
                ]
            )
    return rows


def write_records(records: Sequence[GridRow], directory: Path) -> Path:
    """Every rung of every draw, one JSON object per line, in a git-ignored directory."""
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
        "and every rung but the finest is read as a prefix of one fit",
    )
    parser.add_argument(
        "--control-n",
        type=int,
        default=DEFAULT_CONTROL_N,
        help="rows of the i.i.d. companion the ladder is read against; 0 drops the control, "
        "which leaves the ladder without the rule it is a comparison to",
    )
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20250801)
    parser.add_argument("--tier", type=int, default=2, choices=(1, 2))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmarks/results/drtmle-companion-grid"),
        help="where the per-rung JSONL goes; git-ignored generated output",
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
        Payload(cell, n, data_seed, fold_seed, points, args.control_n)
        for cell in args.cells
        for n in args.sizes
        for data_seed, fold_seed in seeds
    ]
    print(
        f"tier {args.tier}: {len(payloads)} draws over cells {list(args.cells)} and sizes "
        f"{list(args.sizes)}, ladder {list(points)} points "
        f"({2 * points[-1]:,} rows at the finest), control draw {args.control_n:,} rows, "
        f"jobs={args.jobs}"
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

    table("The companion's own error, rung by rung", GRID_HEADERS, grid_rows(records))
    table("What each rule costs", COST_HEADERS, cost_rows(records))

    failures = [r for r in records if r.error]
    if failures:
        print(f"\n{len(failures)} rung(s) failed: {sorted({r.error for r in failures})}")

    print("\nReading the numbers")
    print("=" * 19)
    print(
        "`delta` is the rule's error read off the ladder rather than asserted: the paired\n"
        "movement of the column between a rung and the next coarser one. A ladder that has\n"
        "flattened has a grid whose refinement no longer moves the answer; one that has not\n"
        "is a grid still moving, and the response is a finer rung rather than a footnote.\n"
        "\n"
        "`var removed` is the fraction of the *control* rule's across-draw variance a rung\n"
        "removes, and it is the number E5 sizes against -- how much of C3c's +/-0.09 was the\n"
        "instrument. It is two measured spreads on the same draws through the same primary\n"
        "fits, so no model of the witness enters it. The rest is the estimator's own sampling\n"
        "spread, which only a replicate count reduces.\n"
        "\n"
        "`rule err` is the halving witness and is a fair reading of the *grid's* error. On a\n"
        "control row it is not: halving a noise-dominated rule doubles a variance, so it reads\n"
        "about 1.4x the standard error it stands in for and can exceed the spread. Read it on\n"
        "the sobol rows and read `var removed` on the comparison.\n"
        "\n"
        "The two rules fail differently and the control row is what shows it. The draw's\n"
        "error is noise, independent per replicate, so a study averages it down; the grid's\n"
        "is a bias -- the same points every replicate -- so a study cannot. The grid's being\n"
        "orders smaller is what buys the trade, and this table is where that is a number.\n"
        "\n"
        "Nothing here is a coverage claim and nothing here selects a learner. A flat ladder\n"
        "bounds the *quadrature* and says nothing whatever about the remainder."
    )
    print(f"\n{len(records)} rungs in {elapsed:.0f}s wall clock at jobs={args.jobs}.")
    print(f"Per-rung rows: {path}")


if __name__ == "__main__":
    main()
