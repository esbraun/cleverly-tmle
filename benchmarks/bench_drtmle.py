"""How does the doubly-robust alternation actually exit?

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

**It has been run, and the table is in ``docs/drtmle-investigation-log.md`` under *How the
alternation exits*.**  What it found is worth knowing before running it again.  Item 4's "minority
behaviour rather than the norm" named the wrong minority: 8 of 96 fits ran out of rounds,
but only **2 reached the tolerance** and **86 stalled**.  Item 6 held exactly (94 of 96).
Item 7's disagreement showed on 68 of 96 and the criterion was changed on the strength of
it, so **a fresh sweep no longer measures what that table measures** -- rerun it to see the
new exit distribution, not to reproduce the old one.  And it turned up something none of
the three items was about: ``weak-overlap`` fails ``score_check`` on 23 of 24 fits, which
is now item 11.

The three questions, and the columns that answer them:

* **item 4 -- how often does equation (10)'s near-singular solve bite?**  ``ill>0`` and the
  ``exit`` split.  :math:`g_{r,2}` vanishes exactly where the mechanism is right, so the
  covariate is worst conditioned on the fits anybody wants; ``linear`` is in the sweep
  because it is the process a ``glm`` mechanism gets *right*.
* **item 6 -- does the closing pass's mechanism stage ever stop on its tolerance?**
  ``closing capped``.
* **item 7 -- how far apart are the relative and absolute criteria?**  ``rel eq10`` against
  ``worst |score|`` as a share of ``se/sqrt(n)``.  The loop exits on ``|score|/mean|h|``
  and ``mean|h|`` is ``O(1e-3)`` for that covariate by construction, so a score that is
  negligible beside a standard error can read as a large relative one.  The gap between
  those two columns is the whole of the question.

**This will not run in the Claude Code cloud sandbox**, and ``CLAUDE.md`` explains why: the
defaults here are ~96 fits of tens of seconds each.  Dispatch
``.github/workflows/drtmle-convergence.yml`` and read the table out of the job log, or run
it on a machine with cores to spare.  Interrupt with ``Ctrl-C`` and never ``kill -9``.

Usage::

    python benchmarks/bench_drtmle.py                                   # the full sweep
    python benchmarks/bench_drtmle.py --processes nonlinear --sizes 400 --seeds 2 --jobs 1
    python benchmarks/bench_drtmle.py --rows                            # every fit, not just cells
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from cleverly import DRTMLE
from cleverly.datasets import DGP, linear_dgp, nonlinear_dgp, weak_overlap_dgp
from cleverly.estimators.base import format_table
from cleverly.utils.parallel import map_parallel

#: Two sizes rather than the one the existing measurement used.  The alternation's cost is
#: round count times folds and barely depends on ``n`` (``tests/unit/test_drtmle_fit.py``
#: measures 400 rows as *slower* than 600), so a second size is nearly free and is the only
#: way to see whether the conditioning is a property of the sample size at all.
DEFAULT_SIZES = (600, 1200)

#: Fits per cell.  Twelve is not a coverage study -- nothing here needs a Monte Carlo
#: standard error -- it is enough draws to say whether an exit is the norm or the exception,
#: which is the only claim items 4 and 6 make.
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


@dataclass
class Exit:
    """One fit's convergence record.  Every field is read off the fit, none is derived."""

    process: str
    n: int
    data_seed: int
    fold_seed: int
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
    error: str = ""

    def row(self) -> list[str]:
        return [
            self.process,
            f"{self.n:,}",
            str(self.data_seed),
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


def _failed(process: str, n: int, data_seed: int, fold_seed: int, error: str) -> Exit:
    """A draw the estimator raised on, recorded rather than dropped.

    A sweep that swallowed these would report the exits of the fits that happened to
    survive and call that the distribution.
    """
    return Exit(
        process=process,
        n=n,
        data_seed=data_seed,
        fold_seed=fold_seed,
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
        error=error,
    )


def one_fit(process: str, n: int, data_seed: int, fold_seed: int) -> Exit:
    """Fit once and read the alternation's record off it."""
    frame, _ = PROCESSES[process]().sample(n, seed=data_seed)
    estimator = DRTMLE(**SETTINGS, random_state=fold_seed)  # type: ignore[arg-type]
    start = time.perf_counter()
    try:
        fit = estimator.fit(frame, outcome="Y", treatment="A").single()
    # Recorded and reported rather than swallowed: `main` prints how many raised and what
    # they raised, so a sweep cannot quietly report the exits of the survivors.
    except Exception as exc:
        return _failed(process, n, data_seed, fold_seed, type(exc).__name__)
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
    reference = fit.estimates["ate"].std_error / np.sqrt(n)
    worst = max(abs(row.score) for row in check.rows)

    return Exit(
        process=process,
        n=n,
        data_seed=data_seed,
        fold_seed=fold_seed,
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
    )


def _share(values: list[bool]) -> str:
    return f"{sum(values)}/{len(values)}"


def summarise(results: list[Exit]) -> list[list[str]]:
    """One row per (process, n) cell: what the twelve draws did, not what one of them did."""
    rows: list[list[str]] = []
    cells = sorted({(r.process, r.n) for r in results}, key=lambda c: (c[0], c[1]))
    for process, n in cells:
        cell = [r for r in results if r.process == process and r.n == n]
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
                f"{statistics.median([r.loop_reduced for r in ok]):.1e}",
                f"{statistics.median([r.scale_reduced for r in ok]):.1e}",
                f"{statistics.median([r.worst_share for r in ok]):.1e}",
                _share([not r.score_ok for r in ok]),
            ]
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processes", nargs="+", default=list(PROCESSES), choices=list(PROCESSES))
    parser.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES))
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20250801, help="the sweep's own seed")
    parser.add_argument("--rows", action="store_true", help="print every fit, not just the cells")
    args = parser.parse_args()

    # Both seeds vary, and the second is the point: the one pathological fit on record was
    # "a fit whose fold split was drawn unseeded", so a sweep holding `random_state` at
    # FAST_KWARGS's 0 would sweep straight past the thing it is measuring.
    drawn = np.random.SeedSequence(args.seed).generate_state(2 * args.seeds)
    data_seeds = [int(s) for s in drawn[: args.seeds]]
    fold_seeds = [int(s) for s in drawn[args.seeds :]]

    payloads = [
        (process, n, data_seed, fold_seed)
        for process in args.processes
        for n in args.sizes
        for data_seed, fold_seed in zip(data_seeds, fold_seeds, strict=True)
    ]
    print(f"{len(payloads)} fits over {len(args.processes)} processes, jobs={args.jobs}")
    started = time.perf_counter()
    results = map_parallel(one_fit, payloads, n_jobs=args.jobs)
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

    print("\nReading the numbers")
    print("=" * 19)
    ok = [r for r in results if not r.error]
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
        f"{len(ok) - len(capped) - len(stalled)} reached the tolerance. Item 4 of the "
        "roadmap called a capped exit a minority behaviour of particular draws; the first "
        "sweep found that true (8 of 96) and the contrast misleading -- only 2 of 96 "
        "converged, and stalling is what the loop mostly does. Read all three counts."
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
    print(
        f"\n{len(results)} fits in {elapsed:.0f}s wall clock at jobs={args.jobs} "
        f"({statistics.median([r.seconds for r in ok]):.1f}s median per fit)."
    )


if __name__ == "__main__":
    main()
