r"""Does the doubly-robust interval cover where the plain one does not?  The instrument.

``docs/roadmap.md``'s piece **C** is the demonstration the ``DRTMLE`` variant exists for, and
its definition of done is one sentence: *a demonstration that the interval attains its nominal
coverage where a plain* ``TMLE``'s *does not*.  This script is the **instrument**, in the sense
``benchmarks/bench_drtmle.py`` was the instrument for the B2 sweep: it asserts nothing, its
output is a set of tables a human reads, and running it does not settle anything on its own.

**It is Tier 1 only, and Tier 1 is not the demonstration.**
``docs/drtmle/validation-plan.md`` §5 asks for two tiers.  Tier 1 hands the estimator a
*prescribed* nuisance sequence (``benchmarks/drtmle_injection.py``), which is the only
construction in which "the intended asymptotic regime was entered" is true by definition -- so
it is where a remainder can be read off exactly, and it is **not an applied claim**.  Tier 2 is
prescribed-rate *learners*, it needs the fold-retained nuisance objects :math:`P_0\hat D`
requires, and it is the demonstration; both are piece C2's, and ``--tier 2`` is refused by name
here rather than approximated.

What the tables answer, and why each is here rather than in a summary line:

* **which regime the cells entered.**  ``n^alpha R2`` against the drift coefficient the design
  committed to, per arm and for the ATE, beside each nuisance's :math:`L_2` error and its
  log-log slope.  §5's instruction is blunt about this: *"without these columns a correct
  coverage number is still only a number"*, and the reason is that a nuisance norm falling at
  the right rate does not say the **inner product** did -- the remainder is one, so a
  coefficient can vanish with a nonzero norm.
* **coverage, against its Monte Carlo standard error**, by a Wilson interval and the
  *compatible with 0.95* rule §5 freezes, with the replication count and the interval width on
  the face of the table so that a wide interval cannot read as success.
* **the shortfall, paired on the draw.**  Both estimators fit the *same* draw at the *same*
  injected nuisances, so the difference in coverage is a paired quantity and its standard
  error is the standard deviation of the per-replicate difference -- several times tighter than
  treating the two arms as independent, which is what makes gate 2's ``0.05`` resolvable at 250
  replicates rather than at 1,000.
* **which estimator each cell is evidence about.**  ``contract``, ``clip share``, ``margin``
  and ``min gr1`` off :func:`~cleverly.validation.correction_check` -- gate 1's clause 0 and
  item 25.  A cell with any of the three truncations active is *empirically supported and
  outside Theorem 1*, and a coverage number read without that label is a number about an
  estimator nobody has named.  The cells here are designed to be inside it; the column is what
  checks that rather than assuming it.
* **the invalid-fit rate, three ways.**  The primary report counts an algorithmically invalid
  fit as a **failure of the procedure**, which is an intention-to-treat reading: coverage over
  the surviving fits is conditional on a non-random subset selected on a diagnostic correlated
  with the fit having gone wrong.  The other two accountings are reported beside it, and the
  rule is written down here rather than chosen after seeing which cells it helped.

**Why this is not** :class:`~cleverly.validation.CoverageStudy`.  That class is the right
instrument for "does this configuration cover", and four things it does are wrong for a
demonstration: it swallows every exception, so a draw the estimator raised on disappears; it
keeps no per-replicate row, and §5 asks for them; it cannot pair two estimators on one draw;
and it carries no per-fit diagnostic, so an invalid fit is indistinguishable from a valid one.
What *is* reused is :class:`~cleverly.validation.EstimandSummary`, constructed from the arrays
collected here, so that ``bias``, ``root_n_bias``, ``se_ratio``, ``coverage`` and
``coverage_se`` are the package's own definitions rather than a second set of them.

**Cost, and the sandbox.**  A Tier-1 fit is cheap by the standards of this variant, because the
primary nuisances are prescribed functions rather than learner fits: measured at ~7s for the
pair (``TMLE`` plus ``DRTMLE``) at ``n = 600`` on a four-core container, against the 43s per
``DRTMLE`` fit ``docs/roadmap.md`` costed C from.  A **pilot** is 2 cells x 3 sizes x 50
replicates, which is ~300 pairs and about half an hour at ``jobs=2`` -- too much for the
sandbox and comfortable on a runner.  ``CLAUDE.md``'s rules apply: dispatch
``.github/workflows/drtmle-coverage.yml``, read the tables out of the job log, and interrupt
with ``Ctrl-C`` rather than ``kill -9``.

Usage::

    python benchmarks/drtmle_coverage.py                        # the pilot's shape
    python benchmarks/drtmle_coverage.py --sizes 300 --replicates 4 --jobs 1   # smoke
    python benchmarks/drtmle_coverage.py --cells q-drift --rows
    python benchmarks/drtmle_coverage.py --seed 20250901        # the second seed batch
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cleverly import DRTMLE, TMLE
from cleverly.estimators.base import format_table
from cleverly.utils.parallel import map_parallel
from cleverly.validation import EstimandSummary

try:  # the benchmarks package is importable either way, depending on the entry point
    from benchmarks import drtmle_injection as injection
except ImportError:  # pragma: no cover - direct `python benchmarks/drtmle_coverage.py`
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from benchmarks import drtmle_injection as injection

#: The sizes §5 names, *"adjusted upward if the prescribed rate is not visible"*.  Three rather
#: than two because two are suggestive and three carry a rate.
DEFAULT_SIZES = (600, 1200, 2400)

#: Replicates per cell per size.  The **pilot's** count, not the frozen study's: §5 wants 250 at
#: minimum and 500 if the budget reaches, at which a coverage estimate's Monte Carlo standard
#: error is ``0.014`` and ``0.010``.  Fifty resolves the ``0.08``-to-``0.14`` shortfall the
#: design predicts and does not resolve a ``0.02`` one, which is what a pilot is for.
DEFAULT_REPLICATES = 50

#: The estimands, spelled out because the default binary report asks for ``att``/``atc``, which
#: ``DRTMLE`` refuses -- and because the ATE and the two arm means carry *different* drift
#: coefficients, so a study reporting only the contrast would miss an arm whose coefficient
#: cancelled.
ESTIMANDS = ("ate", "ey1", "ey0")

#: The nominal level every coverage number below is read against.
NOMINAL = 0.95

#: The reduced regressions' learner.  Named rather than defaulted: ``DRTMLE`` falls back to the
#: primary *specification*, which here is an injected instance -- see
#: :func:`benchmarks.drtmle_injection.settings`.
REDUCED_LEARNER = "glm"


@dataclass
class Replicate:
    """One estimator's answer on one draw, with everything a table below needs read off it.

    Flat and JSON-serialisable on purpose: the per-replicate file is the artefact §5 asks be
    kept, and a nested record would need a schema for a reader to do anything with it.
    """

    cell: str
    n: int
    data_seed: int
    fold_seed: int
    estimator: str
    estimand: str
    truth: float
    psi: float
    std_error: float
    lower: float
    upper: float
    covered: bool
    #: Whether this fit solved what it reports -- ``score_check().passed``, which since B1a
    #: covers the state identities and the corrections as well as the three fluctuation rows.
    #: The primary accounting counts ``False`` as a coverage failure.
    valid: bool
    #: Item 25's label and its three witnesses, ``""``/``nan`` for a plain ``TMLE`` fit, which
    #: has no mechanism tilt and so no contract to be inside or outside.
    contract: str
    initial_clip_share: float
    margin: float
    gr1_margin: float
    exit_reason: str
    failure: str
    rounds: int
    seconds: float
    error: str = ""


@dataclass(frozen=True)
class Payload:
    """One draw: the pair of estimators is fitted inside it, so they share the data."""

    cell: str
    n: int
    data_seed: int
    fold_seed: int


def _witnesses(fit: Any) -> dict[str, Any]:
    """Item 25's label and its three numbers, or the empty record for a fit without them."""
    check = fit.validation.correction_check()
    if not check.rows:
        return {
            "contract": "none",
            "initial_clip_share": float("nan"),
            "margin": float("nan"),
            "gr1_margin": float("nan"),
        }
    return {
        "contract": check.contract,
        "initial_clip_share": check.initial_clip_share,
        "margin": check.margin,
        "gr1_margin": check.gr1_margin,
    }


def _alternation(fit: Any) -> dict[str, Any]:
    """The loop's own record, or blanks for a plain fit that never entered one."""
    fluctuation = fit.repeats[0].fluctuations.get("mean")
    reduction = getattr(fluctuation, "reduction", None)
    if reduction is None:
        return {"exit_reason": "", "failure": "", "rounds": 0}
    return {
        "exit_reason": reduction.exit_reason,
        "failure": reduction.failure or "",
        "rounds": int(reduction.n_outer),
    }


def _failed(
    payload: Payload, estimator: str, error: str, truth: dict[str, float]
) -> list[Replicate]:
    """A draw the estimator raised on, recorded rather than dropped.

    Counted as **invalid and uncovered** by the primary accounting, which is the honest
    reading: a procedure that raises has not produced an interval, and dropping it would
    condition the coverage on the draws that happened to work.
    """
    return [
        Replicate(
            cell=payload.cell,
            n=payload.n,
            data_seed=payload.data_seed,
            fold_seed=payload.fold_seed,
            estimator=estimator,
            estimand=name,
            truth=truth[name],
            psi=float("nan"),
            std_error=float("nan"),
            lower=float("nan"),
            upper=float("nan"),
            covered=False,
            valid=False,
            contract="none",
            initial_clip_share=float("nan"),
            margin=float("nan"),
            gr1_margin=float("nan"),
            exit_reason="",
            failure="",
            rounds=0,
            seconds=float("nan"),
            error=error,
        )
        for name in ESTIMANDS
    ]


def one_draw(payload: Payload) -> list[Replicate]:
    """Both estimators on one draw at one cell's injected nuisances.

    Paired inside the worker rather than across two passes, which is what makes the shortfall a
    paired quantity: the two fits see the same rows, the same prescribed nuisance functions and
    the same fold split, so every difference between them is the two extra score equations.
    """
    import warnings

    dgp = injection.base_law()
    frame, _ = dgp.sample(payload.n, seed=payload.data_seed)
    truth = dgp.truth()
    shared = injection.settings(payload.cell, payload.n)

    records: list[Replicate] = []
    for estimator, factory in (
        ("tmle", lambda: TMLE(**shared, random_state=payload.fold_seed)),
        (
            "drtmle",
            lambda: DRTMLE(
                **shared,
                reduced_outcome_learner=REDUCED_LEARNER,
                reduced_treatment_learner=REDUCED_LEARNER,
                random_state=payload.fold_seed,
            ),
        ),
    ):
        started = time.perf_counter()
        try:
            with warnings.catch_warnings():
                # Positivity warnings are per-draw noise here; the tables carry the overlap
                # columns that would say if a cell had a positivity problem.
                warnings.simplefilter("ignore")
                fit = factory().fit(frame, outcome="Y", treatment="A").single()
        except Exception as exc:  # recorded and reported, never swallowed
            records.extend(_failed(payload, estimator, type(exc).__name__, truth))
            continue
        seconds = time.perf_counter() - started
        valid = fit.validation.score_check().passed
        witnesses, alternation = _witnesses(fit), _alternation(fit)
        for name in ESTIMANDS:
            estimate = fit.estimates[name]
            low, high = estimate.ci
            records.append(
                Replicate(
                    cell=payload.cell,
                    n=payload.n,
                    data_seed=payload.data_seed,
                    fold_seed=payload.fold_seed,
                    estimator=estimator,
                    estimand=name,
                    truth=truth[name],
                    psi=float(estimate.psi),
                    std_error=float(estimate.std_error),
                    lower=float(low),
                    upper=float(high),
                    covered=bool(low <= truth[name] <= high),
                    valid=valid,
                    seconds=seconds,
                    **witnesses,
                    **alternation,
                )
            )
    return records


# ------------------------------------------------------------------ the reported rules


def wilson(successes: int, trials: int, level: float = 1.96) -> tuple[float, float]:
    """A Wilson score interval, which §5 prefers to the Wald one near the boundary.

    At a coverage near 0.95 and 50 replicates the Wald interval reaches above 1, which is not
    merely inelegant: an upper limit that cannot be attained makes "compatible with 0.95" read
    as satisfied by construction on the high side.
    """
    if trials == 0:
        return (float("nan"), float("nan"))
    phat = successes / trials
    denominator = 1.0 + level**2 / trials
    centre = (phat + level**2 / (2 * trials)) / denominator
    spread = (
        level * math.sqrt(phat * (1.0 - phat) / trials + level**2 / (4 * trials**2)) / denominator
    )
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def compatible(successes: int, trials: int) -> bool:
    """§5's rule, verbatim: ``|coverage-hat - 0.95| <= 1.96 sqrt(p(1-p)/M)``.

    Reported **beside** the Wilson interval rather than derived from it, because the frozen
    rule is the Wald form and a rule restated is a rule changed.  The interval is what says
    whether the verdict is worth anything at this replication count.
    """
    if trials == 0:
        return False
    phat = successes / trials
    return abs(phat - NOMINAL) <= 1.96 * math.sqrt(phat * (1.0 - phat) / trials)


def _cells(records: Sequence[Replicate]) -> list[tuple[str, int]]:
    """Every ``(cell, n)`` present, in the order the design declares them."""
    seen = {(r.cell, r.n) for r in records}
    sizes = sorted({size for _, size in seen})
    return [(cell, size) for cell in injection.CELLS for size in sizes if (cell, size) in seen]


def _mean_width(rows: Sequence[Replicate]) -> float:
    """Mean interval width over the fits that produced one, which the coverage table reports.

    Beside a coverage number because the two are read together: an interval wide enough to
    cover by accident is not evidence, and §5 asks for the width for exactly that reason.
    """
    widths = [r.upper - r.lower for r in rows if np.isfinite(r.psi)]
    return float(np.mean(widths)) if widths else float("nan")


def _select(
    records: Sequence[Replicate], cell: str, n: int, estimator: str, estimand: str
) -> list[Replicate]:
    return [
        r
        for r in records
        if r.cell == cell and r.n == n and r.estimator == estimator and r.estimand == estimand
    ]


def summarise(rows: Sequence[Replicate]) -> EstimandSummary | None:
    """The package's own summary of the fits that produced an interval.

    ``EstimandSummary`` reused rather than reimplemented: ``bias``, ``root_n_bias``,
    ``monte_carlo_se``, ``mean_std_error``, ``se_ratio``, ``coverage`` and ``coverage_se`` are
    all defined there, and a second definition of any of them in a benchmark is a second
    definition of the thing the study reports.

    Built over the fits that **returned**, so ``bias`` and ``se_ratio`` describe estimates that
    exist; the coverage columns below re-derive their own numerator under each of the three
    invalid-fit accountings, which is where a raised or invalid fit is counted.
    """
    produced = [r for r in rows if np.isfinite(r.psi)]
    if len(produced) < 2:
        return None
    return EstimandSummary(
        estimand=produced[0].estimand,
        truth=float(np.mean([r.truth for r in produced])),
        n=produced[0].n,
        n_replicates=len(produced),
        estimates=np.array([r.psi for r in produced]),
        std_errors=np.array([r.std_error for r in produced]),
        covered=np.array([float(r.covered) for r in produced]),
        rejected=np.zeros(len(produced)),
    )


@dataclass(frozen=True)
class Accounting:
    """One cell's coverage under each of the three invalid-fit rules.

    Attributes
    ----------
    primary:
        An algorithmically invalid fit -- or one that raised -- counts as **not covered**.  The
        intention-to-treat reading, and the study's primary report: coverage computed over the
        surviving fits is conditional on a non-random subset selected on a diagnostic
        correlated with the fit having gone wrong, and reporting *that* as the coverage is the
        same class of error as reporting a per-protocol analysis as intention-to-treat.
    excluded:
        Invalid fits dropped, with :attr:`invalid_share` reported beside it -- never on its own.
    invalid_share:
        The rate itself, which is the third accounting: an algorithm-failure outcome reported
        as its own number rather than folded into a coverage figure.
    """

    trials: int
    primary: float
    excluded: float
    invalid_share: float
    valid_trials: int = 0
    #: The Wilson interval of the *primary* number, since that is the one a gate reads.
    interval: tuple[float, float] = (float("nan"), float("nan"))
    compatible: bool = False


def account(rows: Sequence[Replicate]) -> Accounting | None:
    """The three accountings for one ``(cell, n, estimator, estimand)``."""
    if not rows:
        return None
    trials = len(rows)
    valid = [r for r in rows if r.valid and np.isfinite(r.psi)]
    primary_hits = sum(1 for r in rows if r.covered and r.valid and np.isfinite(r.psi))
    return Accounting(
        trials=trials,
        primary=primary_hits / trials,
        excluded=(sum(1 for r in valid if r.covered) / len(valid)) if valid else float("nan"),
        invalid_share=1.0 - len(valid) / trials,
        valid_trials=len(valid),
        interval=wilson(primary_hits, trials),
        compatible=compatible(primary_hits, trials),
    )


def paired_shortfall(
    records: Sequence[Replicate], cell: str, n: int, estimand: str
) -> tuple[float, float, int]:
    """``coverage(DRTMLE) - coverage(TMLE)`` and its **paired** standard error.

    The pairing is the point.  Both estimators saw the same draw, so the per-replicate
    difference of coverage indicators has a standard deviation several times smaller than the
    two marginal rates suggest, and gate 2's ``0.05`` on the *difference* is resolvable at 250
    replicates because of it.  Treating the arms as independent would need roughly an order of
    magnitude more fits for the same resolution.

    Both indicators are the **primary** accounting's, so an invalid fit on either side counts
    against that side.
    """

    def indicator(estimator: str) -> dict[int, float]:
        return {
            r.data_seed: float(r.covered and r.valid and np.isfinite(r.psi))
            for r in _select(records, cell, n, estimator, estimand)
        }

    ours, theirs = indicator("drtmle"), indicator("tmle")
    seeds = sorted(set(ours) & set(theirs))
    if len(seeds) < 2:
        return (float("nan"), float("nan"), len(seeds))
    differences = np.array([ours[seed] - theirs[seed] for seed in seeds])
    return (
        float(np.mean(differences)),
        float(np.std(differences, ddof=1) / np.sqrt(len(seeds))),
        len(seeds),
    )


# ------------------------------------------------------------------ the tables


def design_rows() -> list[list[str]]:
    """What the design committed to, printed before any measurement is read."""
    return injection.summary_rows()


def regime_rows(records: Sequence[Replicate], sizes: Sequence[int]) -> list[list[str]]:
    """Whether each cell entered the regime it claims -- exactly, since the sequence is prescribed.

    The slope columns are of ``log ||error||`` against ``log n``: ``-alpha`` for the drifting
    nuisance and ``0`` for the misspecified one, which is the pair that says a *product* is
    shrinking because one factor is and not because both are.
    """
    rows = []
    for cell in injection.CELLS:
        if not any(r.cell == cell for r in records):
            continue
        declared = injection.drift_coefficients(cell)
        errors = {n: injection.nuisance_error(cell, n) for n in sizes}
        logs = np.log(np.asarray(sizes, dtype=float))
        for n in sizes:
            remainder = injection.exact_remainder(cell, n)
            rows.append(
                [
                    cell,
                    f"{n:,}",
                    f"{remainder['r2_ate']:+.5f}",
                    f"{n**injection.ALPHA * remainder['r2_ate']:+.4f}",
                    f"{declared['c_ate']:+.4f}",
                    f"{n**injection.ALPHA * remainder['r2_1']:+.4f}",
                    f"{n**injection.ALPHA * remainder['r2_0']:+.4f}",
                    f"{errors[n]['q_error_1']:.4f}",
                    f"{errors[n]['g_error']:.4f}",
                ]
            )
        # One slope row per cell rather than per size, since a slope is a property of the
        # sequence: the drifting nuisance must fall at -alpha and the wrong one must not move.
        if len(sizes) > 1:
            q_slope = np.polyfit(logs, [np.log(errors[n]["q_error_1"]) for n in sizes], 1)[0]
            g_slope = np.polyfit(logs, [np.log(errors[n]["g_error"]) for n in sizes], 1)[0]
            rows.append([cell, "slope", "", "", "", "", "", f"{q_slope:+.3f}", f"{g_slope:+.3f}"])
    return rows


def coverage_rows(records: Sequence[Replicate]) -> list[list[str]]:
    """Coverage and calibration per cell, size, estimator and estimand."""
    rows = []
    for cell, n in _cells(records):
        for estimand in ESTIMANDS:
            for estimator in ("tmle", "drtmle"):
                selected = _select(records, cell, n, estimator, estimand)
                summary, accounted = summarise(selected), account(selected)
                if summary is None or accounted is None:
                    continue
                low, high = accounted.interval
                rows.append(
                    [
                        cell,
                        f"{n:,}",
                        estimand,
                        estimator,
                        str(accounted.trials),
                        f"{summary.bias:+.4f}",
                        f"{summary.root_n_bias:+.3f}",
                        f"{summary.monte_carlo_se:.4f}",
                        f"{summary.mean_std_error:.4f}",
                        f"{summary.se_ratio:.3f}",
                        f"{accounted.primary:.3f}",
                        f"[{low:.3f}, {high:.3f}]",
                        "yes" if accounted.compatible else "NO",
                        f"{_mean_width(selected):.4f}",
                    ]
                )
    return rows


def shortfall_rows(records: Sequence[Replicate]) -> list[list[str]]:
    """The comparison the study is for, paired on the draw."""
    rows = []
    for cell, n in _cells(records):
        for estimand in ESTIMANDS:
            ours = account(_select(records, cell, n, "drtmle", estimand))
            theirs = account(_select(records, cell, n, "tmle", estimand))
            if ours is None or theirs is None:
                continue
            difference, error, pairs = paired_shortfall(records, cell, n, estimand)
            rows.append(
                [
                    cell,
                    f"{n:,}",
                    estimand,
                    str(pairs),
                    f"{theirs.primary:.3f}",
                    f"{ours.primary:.3f}",
                    f"{NOMINAL - theirs.primary:+.3f}",
                    f"{difference:+.3f} +/- {error:.3f}",
                    "yes" if np.isfinite(error) and abs(difference) > 1.96 * error else "no",
                ]
            )
    return rows


def contract_rows(records: Sequence[Replicate]) -> list[list[str]]:
    """Gate 1's clause 0: which estimator each cell's number is evidence about.

    ``DRTMLE`` rows only -- a plain fit has no mechanism tilt and so no contract to be inside.
    The count is of *draws* rather than a median, because one bound-active draw makes the cell's
    coverage number evidence about two estimators.
    """
    rows = []
    for cell, n in _cells(records):
        selected = [r for r in _select(records, cell, n, "drtmle", "ate") if r.contract != "none"]
        if not selected:
            continue
        active = [r for r in selected if r.contract == "bound-active"]
        rows.append(
            [
                cell,
                f"{n:,}",
                str(len(selected)),
                f"{len(active)}/{len(selected)}",
                "theorem" if not active else "BOUND-ACTIVE",
                f"{max(r.initial_clip_share for r in selected):.4f}",
                f"{min(r.margin for r in selected):.2e}",
                f"{min(r.gr1_margin for r in selected):.4f}",
            ]
        )
    return rows


def validity_rows(records: Sequence[Replicate]) -> list[list[str]]:
    """The invalid-fit rate, and what the three accountings do to the coverage number."""
    rows = []
    for cell, n in _cells(records):
        for estimator in ("tmle", "drtmle"):
            selected = _select(records, cell, n, estimator, "ate")
            accounted = account(selected)
            if accounted is None:
                continue
            raised = sorted({r.error for r in selected if r.error})
            rows.append(
                [
                    cell,
                    f"{n:,}",
                    estimator,
                    str(accounted.trials),
                    f"{accounted.invalid_share:.3f}",
                    ", ".join(raised) or "-",
                    f"{accounted.primary:.3f}",
                    f"{accounted.excluded:.3f}",
                    str(accounted.valid_trials),
                ]
            )
    return rows


def _payloads(
    cells: Sequence[str], sizes: Sequence[int], seeds: Sequence[tuple[int, int]]
) -> list[Payload]:
    return [
        Payload(cell, n, data_seed, fold_seed)
        for cell in cells
        for n in sizes
        for data_seed, fold_seed in seeds
    ]


def write_records(records: Sequence[Replicate], directory: Path) -> Path:
    """Every replicate, one JSON object per line, in a git-ignored directory.

    §5 asks for the per-replicate results and not only the summary tables, and the reason is
    the one this whole page keeps running into: a table nobody can recompute becomes folklore.
    The directory is under ``benchmarks/results/``, which is generated output -- a file from a
    two-core container reads as a fact about the package rather than about that box -- so the
    workflow uploads it as an artefact rather than committing it.
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
        "--cells", nargs="+", default=list(injection.CELLS), choices=list(injection.CELLS)
    )
    parser.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES))
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument(
        "--seed",
        type=int,
        default=20250801,
        help="the study's own seed; a different one is the independent second batch, which "
        "section 5 requires be run after the first is complete rather than beside it",
    )
    parser.add_argument("--tier", type=int, default=1, choices=(1, 2))
    parser.add_argument("--rows", action="store_true", help="print every replicate, not the cells")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmarks/results/drtmle-coverage"),
        help="where the per-replicate JSONL goes; git-ignored generated output",
    )
    args = parser.parse_args()

    if args.tier != 1:
        raise SystemExit(
            "tier 2 is not implemented here and is refused rather than approximated. It is "
            "prescribed-rate *learners* -- a series, spline or histogram regression with a "
            "smoothing sequence chosen in advance -- plus the fold-retained nuisance objects "
            "P_0 D-hat needs, and it is piece C2 in docs/roadmap.md. Tier 1's numbers are "
            "about a prescribed nuisance sequence and are not an applied claim; running this "
            "script and calling it the demonstration would be exactly the confusion the two "
            "tiers exist to prevent."
        )

    # Two streams from one seed, prefix-stable the way `bench_drtmle.py`'s are: raising
    # `--replicates` leaves every earlier data seed unchanged and moves the fold block
    # wholesale, so two runs at different counts share draws but not splits and neither
    # supersedes the other. The fold seed varies per replicate deliberately -- the split is
    # part of the procedure whose coverage is being measured, not a nuisance to hold fixed.
    drawn = np.random.SeedSequence(args.seed).generate_state(2 * args.replicates)
    seeds = [
        (int(data), int(fold))
        for data, fold in zip(drawn[: args.replicates], drawn[args.replicates :], strict=True)
    ]
    payloads = _payloads(args.cells, args.sizes, seeds)
    print(
        f"tier 1: {len(payloads)} draws over cells {list(args.cells)} and sizes "
        f"{list(args.sizes)}, two estimators each, jobs={args.jobs}"
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

    table(
        "What the design committed to, before any fit",
        injection.SUMMARY_HEADERS,
        design_rows(),
    )
    table(
        "Which regime the cells entered",
        (
            "cell",
            "n",
            "R2 (ate)",
            "n^a R2 (ate)",
            "declared c",
            "n^a R2 (arm 1)",
            "n^a R2 (arm 0)",
            "||Q-hat - Q0||",
            "||g-hat - g0||",
        ),
        regime_rows(records, args.sizes),
    )
    table(
        "Coverage and calibration",
        (
            "cell",
            "n",
            "estimand",
            "estimator",
            "reps",
            "bias",
            "sqrt(n) bias",
            "mc se",
            "mean se",
            "se ratio",
            "coverage",
            "wilson 95%",
            "compatible",
            "width",
        ),
        coverage_rows(records),
    )
    table(
        "The shortfall, paired on the draw",
        (
            "cell",
            "n",
            "estimand",
            "pairs",
            "tmle",
            "drtmle",
            "tmle shortfall",
            "drtmle - tmle",
            "resolved",
        ),
        shortfall_rows(records),
    )
    table(
        "Which estimator each cell is evidence about (gate 1, clause 0)",
        (
            "cell",
            "n",
            "fits",
            "bound-active",
            "contract",
            "worst clip share",
            "min margin",
            "min gr1 margin",
        ),
        contract_rows(records),
    )
    table(
        "Invalid fits, three accountings",
        (
            "cell",
            "n",
            "estimator",
            "reps",
            "invalid share",
            "raised",
            "coverage (primary)",
            "coverage (excluded)",
            "valid reps",
        ),
        validity_rows(records),
    )

    if args.rows:
        table(
            "Every replicate",
            (
                "cell",
                "n",
                "seed",
                "estimator",
                "estimand",
                "psi",
                "se",
                "covered",
                "valid",
                "contract",
            ),
            [
                [
                    r.cell,
                    f"{r.n:,}",
                    str(r.data_seed),
                    r.estimator,
                    r.estimand,
                    f"{r.psi:+.4f}",
                    f"{r.std_error:.4f}",
                    "yes" if r.covered else "no",
                    "yes" if r.valid else "NO",
                    r.contract,
                ]
                for r in records
            ],
        )

    print("\nReading the numbers")
    print("=" * 19)
    print(
        "This is tier 1 and tier 1 is not the demonstration. The nuisance sequence is\n"
        "prescribed, so 'the intended asymptotic regime was entered' is true by construction\n"
        "rather than measured -- which is what makes the remainder columns exact and what\n"
        "makes these coverage numbers evidence about an estimator fed a designed sequence and\n"
        "not about one fed a learner. Tier 2 is the demonstration and is piece C2."
    )
    print(
        "\nRead the contract table before the coverage table. A cell reading BOUND-ACTIVE is\n"
        "evidence about the constrained rendering rather than about Theorem 1's estimator\n"
        "(docs/roadmap.md item 25), and these cells are designed to read `theorem` -- the base\n"
        "law was chosen for overlap for exactly that reason, so a bound-active row here is a\n"
        "finding about the design and not about the variant."
    )
    print(
        "\nThe primary coverage column counts an algorithmically invalid fit as a failure of\n"
        "the procedure. The excluded column is beside it and is never to be quoted without\n"
        "the invalid share, which is the third accounting."
    )
    errored = [r for r in records if r.error]
    if errored:
        print(
            f"\n{len({(r.cell, r.n, r.data_seed, r.estimator) for r in errored})} fits raised "
            f"({', '.join(sorted({r.error for r in errored}))}); they are counted as invalid "
            "and uncovered rather than dropped."
        )
    finished = [r for r in records if np.isfinite(r.seconds) and r.estimand == "ate"]
    print(
        f"\n{len(payloads)} draws in {elapsed:.0f}s wall clock at jobs={args.jobs}; median "
        f"{float(np.median([r.seconds for r in finished])):.1f}s per fit."
        if finished
        else f"\nNo fit completed in {elapsed:.0f}s."
    )
    print(f"Per-replicate rows: {path}")


if __name__ == "__main__":
    main()
