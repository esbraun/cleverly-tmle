r"""Does the doubly-robust interval cover where the plain one does not?  The instrument.

``docs/roadmap.md``'s piece **C** is the demonstration the ``DRTMLE`` variant exists for, and
its definition of done is one sentence: *a demonstration that the interval attains its nominal
coverage where a plain* ``TMLE``'s *does not*.  This script is the **instrument**, in the sense
``benchmarks/bench_drtmle.py`` was the instrument for the B2 sweep: it asserts nothing, its
output is a set of tables a human reads, and running it does not settle anything on its own.

**Both tiers run from here, and only one of them is the demonstration.**
``docs/drtmle/validation-plan.md`` §5 asks for two.  Tier 1 hands the estimator a *prescribed*
nuisance sequence (``benchmarks/drtmle_injection.py``), which is the only construction in which
"the intended asymptotic regime was entered" is true by definition -- so it is where a remainder
can be read off exactly, and it is **not an applied claim**.  Tier 2 fits both nuisances, the
good one a smoother at a bandwidth sequence committed before any fit
(``benchmarks/drtmle_tier2.py``), and it **is** the demonstration.  ``--tier`` selects between
them and the run banner says which was chosen, because a table that does not say which tier
produced it is a table about an estimator nobody has named.

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
* **coverage within each contract population**, beside the share and never in place of the
  pooled number.  Cells are *mixed* -- C1's witness found a sixth to a third of well-overlapped
  draws exiting bound-active -- and how that is read is the decision C3 froze before its
  dispatch: pooled is primary, the strata are description, and neither is "the theorem-backed
  estimator's coverage", because the label is a post-fit property of the draw.
* **the invalid-fit rate, three ways, and split by cause.**  The primary report counts an
  algorithmically invalid fit as a **failure of the procedure**, which is an intention-to-treat
  reading: coverage over the surviving fits is conditional on a non-random subset selected on a
  diagnostic correlated with the fit having gone wrong.  The other two accountings are reported
  beside it, and the rule is written down here rather than chosen after seeing which cells it
  helped.  ``identity`` and ``score`` split the rate by cause because **gate 1 asks for them
  apart**: an identity residual is a software defect and a score above its threshold is a fit
  that did not converge, and one boolean answers neither clause on its own.

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
from cleverly.validation import DEFAULT_TOLERANCE, EstimandSummary

try:  # the benchmarks package is importable either way, depending on the entry point
    from benchmarks import drtmle_injection, drtmle_remainder, drtmle_tier2
except ImportError:  # pragma: no cover - direct `python benchmarks/drtmle_coverage.py`
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from benchmarks import drtmle_injection, drtmle_remainder, drtmle_tier2

#: The two tiers, each a module supplying the same seven names -- ``CELLS``, ``ALPHA``,
#: ``base_law``, ``settings``, ``drift_coefficients``, ``exact_remainder``,
#: ``nuisance_error`` -- so that one harness reads both and no table branches on which tier
#: it is printing.  What differs is what those names *mean*, and the tier banner below says
#: so on every run rather than leaving a reader to infer it from a filename.
TIERS = {1: drtmle_injection, 2: drtmle_tier2}

#: Which tier is in force.  Module-level because ``one_draw`` runs in a worker process and a
#: module attribute travels with the import rather than through the payload; ``main`` sets it
#: before any draw is dispatched.
injection: Any = drtmle_injection

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

#: Which key of a remainder dict an estimand reads.  ``benchmarks/drtmle_remainder`` indexes
#: its per-arm results the way the estimator names its arms and this module names estimands,
#: so the mapping is written once here rather than inline at each call site.
_REMAINDER_KEYS = {"ate": "r2_ate", "ey1": "r2_1", "ey0": "r2_0"}

#: The nominal level every coverage number below is read against.
NOMINAL = 0.95

#: **The predeclared validity rule**, which gate 1's clause 3 reads a fit against.  It is the
#: package's own default and is *named* here rather than left to it, for the reason §5 gives
#: about every rule in this study: a threshold nobody wrote down before the numbers existed is
#: a threshold that can be chosen after seeing which cells it helped.  Passed explicitly to
#: every ``score_check`` call below and printed in the run banner, so the record and the rule
#: cannot come apart.
VALIDITY_TOLERANCE = DEFAULT_TOLERANCE

#: The reduced regressions' learner.  Named rather than defaulted: ``DRTMLE`` falls back to the
#: primary *specification*, which here is an injected instance -- see
#: :func:`benchmarks.drtmle_injection.settings`.
REDUCED_LEARNER = "glm"

#: Rows of the independent draw ``P_0 D-hat`` is integrated over.  A quadrature rule rather
#: than a sample size: it controls the accuracy of the remainder columns and appears in no
#: root-``n`` scaling, which is taken at the *fitting* size.  ``0`` turns the columns off,
#: and the table then does not print rather than printing blanks.
DEFAULT_EVALUATION_N = 2_000

#: The seed stream the evaluation draws come from, disjoint from the study's own so that
#: raising ``--evaluation-n`` cannot change which rows a replicate was fitted on.
EVALUATION_SEED = 90_000_000


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
    #: The two causes of ``valid = False``, counted apart because **gate 1 asks for them
    #: apart**: clause 2 is *zero state-identity failures* and clause 3 is *every required
    #: final score negligible*, and a single boolean cannot answer both.  It is B1a's own
    #: distinction carried into the study that reads it -- an identity residual is a software
    #: defect and iterating longer cannot fix one, while a score above its threshold is a fit
    #: that did not solve its equations.  ``score_failures`` is every other failing row, so
    #: the two sum to ``len(score_check().failures)``.
    identity_failures: int
    score_failures: int
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
    #: Item 13's columns, on the ``drtmle`` rows of a run with an evaluation draw and
    #: ``nan`` everywhere else.  ``R_2`` is the *plain* remainder at the fitted nuisances,
    #: at the **initial** regression; ``remaining`` is the corrected one Theorem 1 assumes
    #: negligible.
    r2: float = float("nan")
    #: The same remainder at the **targeted** regression, which is what a fit's bias is and
    #: what §5's targeted-coefficient clause requires the regime be read off.  Tier 2's
    #: regime-entry column, in place of tier 1's quadrature.  It reads ``nan`` on nothing --
    #: unlike its siblings it needs no companion, since the law supplies both limits.
    r2_targeted: float = float("nan")
    p0_curve: float = float("nan")
    #: :math:`P_n\hat D`, which targeting drove to zero.  Carried on the record rather than
    #: only on :class:`benchmarks.drtmle_remainder.RemainderRow` so that a reader of the
    #: per-replicate file can *see* that it did -- never so that it can stand in for
    #: ``p0_curve``, which is the one the remainder is built from.
    pn_curve: float = float("nan")
    remaining: float = float("nan")
    root_n_remaining: float = float("nan")
    branch_q: float = float("nan")
    branch_g: float = float("nan")
    branch_error: float = float("nan")
    error: str = ""


@dataclass(frozen=True)
class Payload:
    """One draw: the pair of estimators is fitted inside it, so they share the data."""

    cell: str
    n: int
    data_seed: int
    fold_seed: int
    evaluation_n: int = 0


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


def _failure_counts(check: Any) -> tuple[int, int]:
    """A failing fit's two causes, counted apart, because **gate 1 reads them apart**.

    Clause 2 is *zero state-identity failures across the whole study*; clause 3 is *every
    required final score negligible*.  ``score_check().passed`` answers neither on its own, and
    the two are different findings rather than two shades of one: an identity residual says the
    score the loop recorded and the term the reported curve carries are not the same functional
    of the returned state -- a software defect, which iterating longer cannot fix -- while a
    score above its threshold says the fit did not solve the equation it posed.  B1a worded them
    apart for exactly this reason; collapsing them here would undo that in the one place it
    finally gets read.

    Taken off :attr:`ScoreCheck.identity_failures`, which is the class's own selector on the
    ``identity`` row kind, so there is one definition of "identity failure" rather than two.
    """
    identity = len(check.identity_failures)
    return identity, len(check.failures) - identity


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
            # A fit that raised solved nothing, and neither count is a *failure* of the kind
            # gate 1 reads: there is no state to check an identity against. The `error`
            # column is what such a row is read by, and `invalid share` already carries it.
            identity_failures=0,
            score_failures=0,
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
    # Drawn per replicate rather than once, so the quadrature error is independent across
    # replicates and averages down in the reported mean rather than biasing every row the
    # same way. The seed stream is disjoint from the study's -- see EVALUATION_SEED.
    evaluation = (
        None
        if payload.evaluation_n <= 0
        else drtmle_remainder.evaluation_frame(
            dgp, payload.evaluation_n, EVALUATION_SEED + payload.data_seed % 1_000_003
        )
    )

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
                evaluation=evaluation,
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
        check = fit.validation.score_check(tolerance=VALIDITY_TOLERANCE)
        valid = check.passed
        identity_failures, score_failures = _failure_counts(check)
        witnesses, alternation = _witnesses(fit), _alternation(fit)
        remainder: dict[str, Any] = {}
        # The regime-entry column, on **both** estimators.  It needs no companion -- the law
        # supplies both limits and the fit supplies its own targeted regression -- and the
        # arm that matters for gate 2 is the plain `TMLE`'s, since that is the estimator whose
        # interval a shortfall is claimed against.  Reading `DRTMLE`'s in its place would
        # answer for the corrected fit's bias, which is a different quantity.
        try:
            targeted = drtmle_remainder.targeted_remainder(fit, dgp, fit.config.g_bounds)
        except Exception as exc:  # pragma: no cover - reported, never hidden
            print(f"targeted remainder unavailable on {payload.cell} n={payload.n}: {exc!r}")
            targeted = {}
        if estimator == "drtmle" and evaluation is not None:
            # Never swallowed into the fit's own failure: a remainder that could not be
            # computed is a gap in item 13's evidence, not a draw the estimator raised on,
            # and the two must not be reported as one.
            try:
                remainder = {
                    row.estimand: row
                    for row in drtmle_remainder.remainder_rows(
                        fit, dgp, n=payload.n, bounds=fit.config.g_bounds
                    )
                }
            except Exception as exc:  # pragma: no cover - reported, never hidden
                print(f"remainder columns unavailable on {payload.cell} n={payload.n}: {exc!r}")
        for name in ESTIMANDS:
            estimate = fit.estimates[name]
            low, high = estimate.ci
            row = remainder.get(name)
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
                    identity_failures=identity_failures,
                    score_failures=score_failures,
                    seconds=seconds,
                    r2=float("nan") if row is None else row.r2,
                    r2_targeted=targeted.get(_REMAINDER_KEYS[name], float("nan")),
                    p0_curve=float("nan") if row is None else row.p0_curve,
                    pn_curve=float("nan") if row is None else row.pn_curve,
                    remaining=float("nan") if row is None else row.remaining,
                    root_n_remaining=float("nan") if row is None else row.root_n_remaining,
                    branch_q=float("nan") if row is None else row.branch_q,
                    branch_g=float("nan") if row is None else row.branch_g,
                    branch_error=float("nan") if row is None else row.branch_error,
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
#
# Every table's headers are declared **beside the function that builds its rows**, and
# ``main`` reads them from here rather than spelling them out at the call site.  The reason is
# a hazard this study cannot afford: the whole output of a dispatch is a log a human reads
# down a column, so a header tuple that has drifted one place from its row builder does not
# fail -- it relabels every number underneath it.  Inserting one column into
# :func:`remainder_rows` and forgetting its header is a two-line mistake that produces a
# complete, plausible, wrong table, and it happened while this pair was two declarations.
# ``TestEveryTablesRowsMatchItsHeaders`` is the pin.


def design_rows() -> list[list[str]]:
    """What the design committed to, printed before any measurement is read."""
    return injection.summary_rows()


#: Headers for :func:`regime_rows`, declared beside it -- see the note above.
REGIME_HEADERS = (
    "cell",
    "n",
    "R2(Q-hat)",
    "n^a R2(Q-hat)",
    "declared c",
    "R2(Qbar*)",
    "n^a R2(Qbar*)",
    "declared b",
    "||Q-hat - Q0||",
    "||g-hat - g0||",
)


def regime_rows(records: Sequence[Replicate], sizes: Sequence[int]) -> list[list[str]]:
    """Whether each cell entered the regime it claims -- exactly, since the sequence is prescribed.

    **Two remainder columns, not one, and the second is the one that answers the question.**
    ``R2(Q-hat)`` is the plug-in remainder at the initial regression: it says the *injection* is
    what the design says it is.  ``R2(Qbar*)`` is the same expression at the **targeted**
    regression, which is what a fit's bias is -- and §5's targeted-coefficient clause requires
    the regime be read off that one.  C3a's pilot had only the first, read it as the second, and
    dispatched a design whose drift the fluctuation was absorbing whole.

    The slope columns are of ``log ||error||`` against ``log n``: ``-alpha`` for the drifting
    nuisance and ``0`` for the misspecified one, which is the pair that says a *product* is
    shrinking because one factor is and not because both are.
    """
    rows = []
    for cell in injection.CELLS:
        if not any(r.cell == cell for r in records):
            continue
        declared = injection.drift_coefficients(cell)
        declared_b = injection.targeted_coefficients(cell)
        errors = {n: injection.nuisance_error(cell, n) for n in sizes}
        logs = np.log(np.asarray(sizes, dtype=float))
        for n in sizes:
            remainder = injection.exact_remainder(cell, n)
            targeted = injection.exact_targeted_remainder(cell, n)
            rows.append(
                [
                    cell,
                    f"{n:,}",
                    f"{remainder['r2_ate']:+.5f}",
                    f"{n**injection.ALPHA * remainder['r2_ate']:+.4f}",
                    f"{declared['c_ate']:+.4f}",
                    f"{targeted['r2_ate']:+.5f}",
                    f"{n**injection.ALPHA * targeted['r2_ate']:+.4f}",
                    f"{declared_b['b_ate']:+.4f}",
                    f"{errors[n]['q_error_1']:.4f}",
                    f"{errors[n]['g_error']:.4f}",
                ]
            )
        # One slope row per cell rather than per size, since a slope is a property of the
        # sequence: the drifting nuisance must fall at -alpha and the wrong one must not move.
        if len(sizes) > 1:
            q_slope = np.polyfit(logs, [np.log(errors[n]["q_error_1"]) for n in sizes], 1)[0]
            g_slope = np.polyfit(logs, [np.log(errors[n]["g_error"]) for n in sizes], 1)[0]
            rows.append(
                [cell, "slope", "", "", "", "", "", "", f"{q_slope:+.3f}", f"{g_slope:+.3f}"]
            )
    return rows


#: Headers for :func:`entry_rows`, declared beside it -- see the note above.
ENTRY_HEADERS = (
    "cell",
    "n",
    "estimator",
    "reps",
    "R2(Qbar*)",
    "n^a R2(Qbar*)",
    "declared b",
    "within",
    "sqrt(n) R2(Qbar*)",
)


def entry_rows(records: Sequence[Replicate]) -> list[list[str]]:
    r"""Pre-flight condition 1, measured: is the fit's **bias** the declared drift?

    ``docs/drtmle/validation-plan.md`` §5 requires this be read *"before a coverage dispatch,
    not inferred from one afterwards"*, and requires it at the **targeted** regression -- the
    clause C3a's pilot failed on precisely by not having it.  The table above it reports what
    the design predicts; this reports what the fits did.

    **Per estimator, and the row that matters is the plain ``TMLE``'s.**  It is the estimator
    whose interval a shortfall is claimed against, so it is the one that has to be in the
    regime; ``DRTMLE``'s row is its own bias and is here beside it as description, since a
    corrected fit's bias is a different quantity and one gate 2 makes no claim about.

    ``within`` is the ratio of the realised coefficient to the declared one.  Reported rather
    than thresholded, because §5 declares no number for it -- the study's write-up reads it and
    a threshold invented here would put a rule in a second place.
    """
    rows = []
    for cell, n in _cells(records):
        declared = injection.targeted_coefficients(cell)["b_ate"]
        for estimator in ("tmle", "drtmle"):
            selected = [
                r for r in _select(records, cell, n, estimator, "ate") if np.isfinite(r.r2_targeted)
            ]
            if not selected:
                continue
            values = np.array([r.r2_targeted for r in selected], dtype=float)
            mean = float(values.mean())
            error = float(np.std(values, ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0
            scaled = n**injection.ALPHA * mean
            rows.append(
                [
                    cell,
                    f"{n:,}",
                    estimator,
                    str(len(selected)),
                    f"{mean:+.5f} +/- {error:.5f}",
                    f"{scaled:+.4f}",
                    f"{declared:+.4f}",
                    f"{scaled / declared:.2f}x" if declared else "-",
                    f"{math.sqrt(n) * mean:+.3f}",
                ]
            )
    return rows


#: Headers for :func:`coverage_rows`, declared beside it -- see the note above.
COVERAGE_HEADERS = (
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
)


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


#: Headers for :func:`shortfall_rows`, declared beside it -- see the note above.
SHORTFALL_HEADERS = (
    "cell",
    "n",
    "estimand",
    "pairs",
    "tmle",
    "drtmle",
    "tmle shortfall",
    "drtmle - tmle",
    "resolved",
)


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


#: Headers for :func:`contract_rows`, declared beside it -- see the note above.
CONTRACT_HEADERS = (
    "cell",
    "n",
    "fits",
    "bound-active",
    "contract",
    "worst clip share",
    "min margin",
    "min gr1 margin",
)


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


#: Headers for :func:`stratum_rows`, declared beside it -- see the note above.
STRATUM_HEADERS = (
    "cell",
    "n",
    "population",
    "reps",
    "share",
    "coverage",
    "wilson 95%",
)


def stratum_rows(records: Sequence[Replicate]) -> list[list[str]]:
    """Coverage within each contract population, which is C3's mixed-cell decision.

    **Description, not a verdict, and the distinction is the whole rule.**  C1's witness found
    cells are *mixed* -- a sixth to a third of well-overlapped draws exit bound-active, because
    equation (9)'s covariate vanishes where the outcome regression is right rather than because
    overlap is poor -- so a cell's coverage number is partly evidence about the constrained
    rendering.  What gate 1's clause 0 reads is the **share**, in the contract table; what
    clauses 5 and 6 read is the **pooled** number, in the coverage table.

    These rows are neither.  The contract label is a *post-fit property of the draw*, so
    conditioning on it selects a non-random subset exactly as excluding invalid fits does --
    and the same objection applies: neither stratum may be quoted as "the theorem-backed
    estimator's coverage".  What they answer is the one question the share alone cannot, which
    is whether the two populations behave differently at all.

    ``DRTMLE`` rows only: a plain fit has no mechanism tilt and so no contract to be inside.
    """
    rows = []
    for cell, n in _cells(records):
        selected = [r for r in _select(records, cell, n, "drtmle", "ate") if r.contract != "none"]
        if not selected:
            continue
        for population in ("theorem", "bound-active"):
            stratum = [r for r in selected if r.contract == population]
            if not stratum:
                continue
            # The primary accounting, unchanged, applied within the stratum -- so a number
            # here differs from the pooled one only through which draws it is over.
            hits = sum(1 for r in stratum if r.covered and r.valid and np.isfinite(r.psi))
            low, high = wilson(hits, len(stratum))
            rows.append(
                [
                    cell,
                    f"{n:,}",
                    population,
                    str(len(stratum)),
                    f"{len(stratum) / len(selected):.3f}",
                    f"{hits / len(stratum):.3f}",
                    f"[{low:.3f}, {high:.3f}]",
                ]
            )
    return rows


#: Headers for :func:`validity_rows`, declared beside it -- see the note above.
VALIDITY_HEADERS = (
    "cell",
    "n",
    "estimator",
    "reps",
    "invalid share",
    "identity",
    "score",
    "raised",
    "coverage (primary)",
    "coverage (excluded)",
    "valid reps",
)


def validity_rows(records: Sequence[Replicate]) -> list[list[str]]:
    """The invalid-fit rate, and what the three accountings do to the coverage number.

    The two failure counts are reported **apart** because gate 1 asks for them apart: clause 2
    is *zero state-identity failures across the whole study* and clause 3 is *every required
    final score negligible*.  A single ``valid`` boolean answers neither on its own, and the
    two mean different things -- an identity residual is a software defect that iterating
    longer cannot fix, and a score above its threshold is a fit that did not converge.  Counted
    over the ``ate`` rows, which is one row per fit rather than one per estimand.
    """
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
                    str(sum(r.identity_failures for r in selected)),
                    str(sum(r.score_failures for r in selected)),
                    ", ".join(raised) or "-",
                    f"{accounted.primary:.3f}",
                    f"{accounted.excluded:.3f}",
                    str(accounted.valid_trials),
                ]
            )
    return rows


def _cancellation(branch_q: float, branch_g: float) -> str:
    """How much of the total the two appendix branches cancel out of, as a ratio.

    Gate 1's clause 4 is two claims and only the first had a column: the total has to trend to
    zero, **and** it must not do so because one large branch cancels the other.  This is the
    second, ``(|R_Q| + |R_g|) / |R_Q + R_g|``: one where the branches do not oppose each other
    at all, and unbounded as the total goes to zero with the branches apart.

    Deliberately a **ratio and not a verdict**.  §5 declares no threshold for it, and inventing
    one here would put a rule in a second place -- the number is reported and the clause is read
    against it in the study's write-up.
    """
    total = abs(branch_q + branch_g)
    magnitude = abs(branch_q) + abs(branch_g)
    if not np.isfinite(magnitude) or magnitude == 0.0:
        return "-"
    if total == 0.0:
        return "inf"
    return f"{magnitude / total:.2f}x"


#: Headers for :func:`remainder_rows`, declared beside it -- see the note above.
REMAINDER_HEADERS = (
    "cell",
    "n",
    "estimand",
    "reps",
    "R2 (fitted)",
    "sqrt(n) R2",
    "n^a R2",
    "declared c",
    "R_rem",
    "sqrt(n) R_rem",
    "R_Q",
    "R_g",
    "cancel",
    "branches resolved",
)


def remainder_rows(records: Sequence[Replicate]) -> list[list[str]]:
    """Item 13's columns, averaged over the replicates of each cell and size.

    Averaged rather than reported per draw because a single draw's ``R_remaining`` carries
    the estimator's own sampling noise as well as the remainder -- what item 13 asks about is
    :math:`\\sqrt n R_{\text{remaining}} \to 0`, a statement about the sequence, so the Monte
    Carlo standard error travels beside every entry.

    **Two plain-remainder columns, and the second is the regime-entry one.**
    ``R2(Q-hat)`` is at the initial regression and says the fitted nuisances are what the
    design predicts; ``R2(Qbar*)`` is the same expression at the **targeted** one, which is
    what a fit's bias is.  §5's targeted-coefficient clause requires the regime be read off
    the second, and ``sqrt(n) R2(Qbar*)`` is what **gate 2's** clause 1 reads -- its third
    condition is that the plain remainder fails to vanish in the cell the shortfall is claimed
    in, and a remainder the fluctuation has absorbed vanishes whatever was injected.
    ``sqrt(n) R_rem`` is the one gate 1's clause 4 reads.  The branch columns are ``-`` where
    the binned limits did not resolve them, which is a statement about the design rather than
    about the estimator -- see ``benchmarks/drtmle_remainder.py``.

    ``cancel`` is clause 4's **second half**, which had no column at all: a total trending to
    zero can conceal two large branches of opposite sign, so the ratio
    ``(|R_Q| + |R_g|) / |R_Q + R_g|`` says how much cancellation the total rests on.  One is
    no cancellation; a large value is the failure mode the clause names, and it is reported
    rather than thresholded because §5 sets no number for it.  ``-`` where the branches did
    not resolve, since a ratio of two unresolved quantities is not a measurement of anything.
    """
    rows = []
    for cell, n in _cells(records):
        declared = injection.drift_coefficients(cell)["c_ate"]
        for estimand in ESTIMANDS:
            selected = [
                r for r in _select(records, cell, n, "drtmle", estimand) if np.isfinite(r.remaining)
            ]
            if not selected:
                continue

            def column(name: str, rows_: Sequence[Replicate] = selected) -> tuple[float, float]:
                values = np.array([getattr(r, name) for r in rows_], dtype=float)
                finite = values[np.isfinite(values)]
                if finite.size == 0:
                    return (float("nan"), float("nan"))
                error = (
                    float(np.std(finite, ddof=1) / np.sqrt(finite.size)) if finite.size > 1 else 0.0
                )
                return (float(np.mean(finite)), error)

            r2_mean, _ = column("r2")
            root_mean, root_error = column("root_n_remaining")
            branch_q, _ = column("branch_q")
            branch_g, _ = column("branch_g")
            resolved = sum(1 for r in selected if np.isfinite(r.branch_q))
            rows.append(
                [
                    cell,
                    f"{n:,}",
                    estimand,
                    str(len(selected)),
                    f"{r2_mean:+.4f}",
                    f"{math.sqrt(n) * r2_mean:+.4f}",
                    f"{n**injection.ALPHA * r2_mean:+.4f}",
                    f"{declared:+.4f}" if estimand == "ate" else "",
                    f"{column('remaining')[0]:+.5f}",
                    f"{root_mean:+.4f} +/- {root_error:.4f}",
                    f"{branch_q:+.5f}" if resolved else "-",
                    f"{branch_g:+.5f}" if resolved else "-",
                    _cancellation(branch_q, branch_g) if resolved else "-",
                    f"{resolved}/{len(selected)}",
                ]
            )
    return rows


#: Headers for :func:`replicate_rows`, declared beside it -- see the note above.
REPLICATE_HEADERS = (
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
)


def replicate_rows(records: Sequence[Replicate]) -> list[list[str]]:
    """Every replicate, under ``--rows``.

    A ten-column projection of a twenty-nine-field record: the rest are in the per-replicate
    JSONL, which is where a reader who wants them should go rather than to a wider table.  What
    is here is what a human scans a log for -- which draw, which estimator, did it cover, did it
    solve what it reports, and which side of the contract it exited on.
    """
    return [
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
    ]


def _payloads(
    cells: Sequence[str],
    sizes: Sequence[int],
    seeds: Sequence[tuple[int, int]],
    evaluation_n: int,
) -> list[Payload]:
    return [
        Payload(cell, n, data_seed, fold_seed, evaluation_n)
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
        "--cells",
        nargs="+",
        default=list(drtmle_injection.CELLS),
        choices=list(drtmle_injection.CELLS),
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
    parser.add_argument(
        "--tier",
        type=int,
        default=1,
        choices=(1, 2),
        help="1 is the prescribed nuisance sequence and is not the demonstration; 2 is the "
        "prescribed-rate learners and is",
    )
    parser.add_argument(
        "--evaluation-n",
        type=int,
        default=DEFAULT_EVALUATION_N,
        help="rows of the independent draw P_0 D-hat is integrated over, which is item 13's "
        "instrument; 0 turns the remainder columns off",
    )
    parser.add_argument("--rows", action="store_true", help="print every replicate, not the cells")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmarks/results/drtmle-coverage"),
        help="where the per-replicate JSONL goes; git-ignored generated output",
    )
    args = parser.parse_args()

    # Set before any draw is dispatched, and read by `one_draw` in the worker: the tier is a
    # module of designs rather than a branch, so every table below reads one interface.
    global injection
    injection = TIERS[args.tier]

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
    payloads = _payloads(args.cells, args.sizes, seeds, args.evaluation_n)
    print(
        f"tier {args.tier}: {len(payloads)} draws over cells {list(args.cells)} and sizes "
        f"{list(args.sizes)}, two estimators each, jobs={args.jobs}, "
        f"evaluation draw {args.evaluation_n:,} rows"
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
        "Which regime the cells entered, as the design predicts it",
        REGIME_HEADERS,
        regime_rows(records, args.sizes),
    )
    table(
        "Which regime the fits entered (pre-flight condition 1)",
        ENTRY_HEADERS,
        entry_rows(records),
    )
    table(
        "Coverage and calibration",
        COVERAGE_HEADERS,
        coverage_rows(records),
    )
    table(
        "The shortfall, paired on the draw",
        SHORTFALL_HEADERS,
        shortfall_rows(records),
    )
    if args.evaluation_n > 0:
        table(
            "The remainder Theorem 1 assumes negligible (item 13)",
            REMAINDER_HEADERS,
            remainder_rows(records),
        )
    table(
        "Which estimator each cell is evidence about (gate 1, clause 0)",
        CONTRACT_HEADERS,
        contract_rows(records),
    )
    table(
        "Coverage within the two contract populations (description, not a verdict)",
        STRATUM_HEADERS,
        stratum_rows(records),
    )
    table(
        "Invalid fits, three accountings",
        VALIDITY_HEADERS,
        validity_rows(records),
    )

    if args.rows:
        table(
            "Every replicate",
            REPLICATE_HEADERS,
            replicate_rows(records),
        )

    print("\nReading the numbers")
    print("=" * 19)
    if args.tier == 1:
        print(
            "This is tier 1 and tier 1 is not the demonstration. The nuisance sequence is\n"
            "prescribed, so 'the intended asymptotic regime was entered' is true by\n"
            "construction rather than measured -- which is what makes these coverage numbers\n"
            "evidence about an estimator fed a designed sequence and not about one fed a\n"
            "learner. Tier 2 is the demonstration."
        )
    else:
        print(
            "This is tier 2 and it is the demonstration: both nuisances are fitted, and the\n"
            "good one is a smoother whose bandwidth sequence was committed before any fit.\n"
            "So 'the intended regime was entered' is a *measurement* here rather than a\n"
            "construction -- read `n^a R2` against `declared c` in the remainder table before\n"
            "reading any coverage number, because a cell whose realised coefficient missed is\n"
            "a cell whose coverage answers about a different regime."
        )
    print(
        "\nRead the contract table before the coverage table. A cell reading BOUND-ACTIVE is\n"
        "evidence about the constrained rendering rather than about Theorem 1's estimator\n"
        "(docs/roadmap.md item 25), and these cells are designed to read `theorem` -- the base\n"
        "law was chosen for overlap for exactly that reason, so a bound-active row here is a\n"
        "finding about the design and not about the variant."
    )
    print(
        "\nCells are mixed rather than pure, so the stratum table is beside the contract one\n"
        "and is *description*: the pooled coverage number is what gate 1's clauses 5 and 6\n"
        "read, the share is what clause 0 reads, and neither stratum is 'the theorem-backed\n"
        "estimator's coverage' -- the label is a post-fit property of the draw, so selecting\n"
        "on it conditions on a non-random subset exactly as dropping invalid fits would."
    )
    print(
        "\nThe primary coverage column counts an algorithmically invalid fit as a failure of\n"
        "the procedure. The excluded column is beside it and is never to be quoted without\n"
        "the invalid share, which is the third accounting. `identity` and `score` split that\n"
        "share by cause, which gate 1 asks for apart: clause 2 is zero state-identity\n"
        "failures and clause 3 is every required score negligible at tolerance "
        f"{VALIDITY_TOLERANCE:g},\nwhich is the predeclared validity rule this run was read "
        "under."
    )
    if args.evaluation_n > 0:
        print(
            "\nThe remainder table is item 13. `sqrt(n) R_rem` is what gate 1's clause 4 reads\n"
            "and it has to trend to zero across the sizes; `R_Q` and `R_g` are the two\n"
            "appendix branches, so that a total trending to zero cannot conceal cancellation\n"
            "between them, and `cancel` is that second half as a number -- (|R_Q| + |R_g|)\n"
            "over |R_Q + R_g|, one where the branches do not oppose each other and large\n"
            "where a small total rests on two large branches of opposite sign. `sqrt(n) R2`\n"
            "is the plain remainder and is gate 2's clause 1, whose third condition is that\n"
            "it fails to vanish in the cell the shortfall is claimed in.\n"
            "Where `branches resolved` is short of the replicate count, the\n"
            "binned limits those two are built from did not separate from their own\n"
            "discretisation error, which is a statement about this design and not about the\n"
            "estimator -- benchmarks/drtmle_remainder.py says what is approximated in them."
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
