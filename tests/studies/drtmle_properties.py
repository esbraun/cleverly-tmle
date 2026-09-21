"""Truth-based repeated-sampling properties for the DR-TMLE protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from cleverly.utils.parallel import map_parallel
from tests.parallel import STUDY_JOBS
from tests.studies.canonical_drtmle import STUDY, draw_from_seed, fit_cleverly
from tests.studies.evidence.properties import REPLICATE_COLUMNS, replicate_row
from tests.studies.evidence.property_verdicts import (
    CONTRACTION_SCENARIOS,
    control_role,
    summarize_contraction_properties,
)
from tests.studies.evidence.seeds import stream_seed

DOUBLE_ROBUST_REPLICATES = 800
RATE_REPLICATES = 800
CALIBRATION_REPLICATES = 2_400
RATE_SIZES = (500, 1500, 4500)
CALIBRATION_N = 3000

#: The sizes the contraction ladder is fitted over, and how many replications each rung gets.
#:
#: **Why this family exists.**  ``double_robustness`` judges the bias at one size against an
#: equivalence margin of a quarter of an empirical standard deviation, and on this law the two
#: one-correct cells exceed it at ``n = 1,500``.  A single red cell cannot say which of two
#: very different things happened: a second-order remainder that has not yet decayed, which is
#: what Theorem 1 predicts and leaves the interval eventually valid, or an estimator that is
#: not consistent at all.  Those have the same appearance at one size and opposite meanings.
#:
#: Fitting log |bias| on log ``n`` tests whether the observed bias contracts.  A second-order
#: remainder predicts a slope near ``-1``, a first-order term predicts one near ``-1/2``, and an
#: inconsistent estimator can produce one near ``0``.  Three finite-sample points do not identify
#: which term produced the slope.  The ``both_wrong`` arm rides along as the control that must fail
#: to contract.
#:
#: Raising the level margin instead was considered and rejected: measured over ``n`` in
#: (1500, 3000, 6000), the standardized bias under a correct mechanism runs 0.357, 0.171 and
#: 0.135, so no size on any affordable ladder brings the 99% interval inside 0.25.  The level
#: cell is left red and this family says whether its observed bias contracts.
#:
#: The rungs are judged on *coverage* rather than on that bias -- see
#: :func:`~tests.studies.evidence.property_verdicts.contraction_verdicts`. One rung is red:
#: at ``n = 1,500`` with the
#: outcome regression misspecified the exact coverage interval dips below the declared floor.
#: That is a small-sample statement in one regime, and it is a result the single-size study
#: had no way to reach.
CONTRACTION_SIZES = (1500, 3000, 6000)

#: How many replications each rung of :data:`CONTRACTION_SIZES` runs, in the same order, and
#: the rule that sets it.
#:
#: The rule reads two kinds of quantity and no others: the bias scale this family already
#: publishes, and the *control*'s empirical spread.  The bias scale is ``0.0036`` at the
#: first rung, carried down the ladder by the ``1/n`` decay a second-order remainder
#: predicts.  The control is ``both_wrong``, whose committed rows give
#: ``sd(n) * sqrt(n)`` of 1.0444, 1.0082 and 1.0844 at 1,500, 3,000 and 6,000, so
#: ``sd(n) ~ 1.05 / sqrt(n)``.  The rule does not read the slope, the interval or the verdict
#: of ``rate_outcome_correct`` or ``rate_treatment_correct``, because those cells are
#: ``role="positive"`` and the interval *is* their verdict.  Commit ``308467c`` had to undo
#: exactly that mistake in the bounded generated-design family, where two constants were
#: chosen by reading the positive cell's own verdict statistic.
#:
#: **Why the middle rung keeps 800.**  Three log-equally-spaced rungs give a centred weight
#: vector of ``(-0.693, 0, +0.693)``, so the middle rung's weight is exactly zero and the
#: fitted slope is ``(y3 - y1) / log 4``.  Replications there buy no resolution at all.  800
#: is the floor its own per-rung coverage row needs, and that row is why the rung runs.
#:
#: **The arithmetic at the two outer rungs.**  The resolution target is a slope of magnitude
#: one, which is the separation this family exists to make: the second-order prediction of
#: ``-1`` against the non-contraction alternative of ``0``.  Write the relative Monte Carlo
#: error of a rung's bias as ``sigma(n, R) = MCSE / |bias| = 0.1944 * sqrt(n / R)``, which is
#: ``1.05 / 5.4`` times ``sqrt(n / R)``.  The slope's variance is then
#: ``Var(slope) = 0.52034 * (sigma_1^2 + sigma_3^2)``, the middle rung contributing nothing.
#: At ``R = 2,400`` the projected 99% half-width is about 0.88, against about 2.0 at the
#: shipped 800.  ``R = 2,000`` gives about 1.05 and misses the declared target of 1.00, so
#: 2,400 is the budget.
#:
#: Those three figures are projections from a surrogate, and they are quoted to two digits
#: because that is the precision two independent simulations agreed on.  One gave 2.022,
#: 1.049 and 0.876; the other gave 1.93, 1.02 and 0.85.  They differ by about three per cent
#: and they order the budgets identically, so the 2,400 rung clears the target under both and
#: the 2,000 rung misses it under both.  The choice does not rest on the third digit, and the
#: run reports the half-width the ladder actually produces.
#:
#: The claim is about the instrument and not about the verdict.  The run publishes what it
#: produces.  A ``rate_outcome_correct`` interval that still covers zero is a result this
#: ladder reports, and not a failure of this rule.
#:
#: The extra draws at the outer rungs buy the slope and nothing else.  Each rung's own
#: coverage verdict is read at :data:`CONTRACTION_VERDICT_REPLICATES`, for the reason stated
#: there.
CONTRACTION_REPLICATES = (2_400, 800, 2_400)

#: How many replications each rung's *own* coverage verdict is read from.
#:
#: **Why 800.**  It is the budget every rung of this ladder published when the family shipped,
#: and every rung's coverage row was judged at it.  Restoring it restores those verdicts rather
#: than changing them.
#:
#: **Why it must not track** :data:`CONTRACTION_REPLICATES`.  Raising the outer rungs for the
#: slope also raised these gates, and ``treatment_correct_n1500`` turned green.  At 800
#: replications it covers 732 of 800, a 99% lower endpoint of 0.8864; at 2,400 it covers 2,220
#: of 2,400, a lower endpoint of 0.9101.
#:
#: The reason that is refused is *procedural* rather than statistical.  A larger budget walks
#: either endpoint towards the truth, not towards the margin, and it buys a pass exactly when
#: the truth already satisfies the gate.  This cell is the demonstration: its coverage is 0.9150
#: at 800 and 0.9250 at 2,400, both above the 0.90 floor, so the extra draws resolved a cell the
#: declared budget could not read.  The slope behaves the same way, and neither gate is immune.
#:
#: What differs is that the rung budget was *declared* and this one was not.
#: :data:`CONTRACTION_REPLICATES` carries a rule written before the run, which reads the law's
#: bias scale and the control's spread and names what it refuses to read.  The coverage budget
#: rose as a side effect nobody declared, and an undeclared budget cannot be told apart from one
#: chosen after reading the verdict.  ``docs/roadmap.md`` RM18 refuses that.  So the extra draws
#: serve the slope alone, and a future change to :data:`CONTRACTION_REPLICATES` must leave this
#: number where it is unless it means to restate the rungs' coverage claims.
CONTRACTION_VERDICT_REPLICATES = 800

if len(CONTRACTION_REPLICATES) != len(CONTRACTION_SIZES):  # pragma: no cover - import-time guard
    raise AssertionError(
        f"the contraction ladder declares {len(CONTRACTION_SIZES)} sizes and "
        f"{len(CONTRACTION_REPLICATES)} replication counts; they are indexed together in "
        f"cells() and a mismatch would silently give a rung the wrong budget"
    )

if min(CONTRACTION_REPLICATES) < CONTRACTION_VERDICT_REPLICATES:  # pragma: no cover - guard
    raise AssertionError(
        f"the contraction ladder reads each rung's verdict at {CONTRACTION_VERDICT_REPLICATES} "
        f"replications, but its smallest rung runs {min(CONTRACTION_REPLICATES)}; a verdict "
        f"cannot be read at a budget the rung never drew"
    )


@dataclass(frozen=True)
class Cell:
    property: str
    cell: str
    scenario: str
    n: int
    replicates: int
    seed_offset: int
    role: str = "positive"


def cells() -> tuple[Cell, ...]:
    out = [
        Cell(
            "double_robustness",
            scenario,
            scenario,
            1500,
            DOUBLE_ROBUST_REPLICATES,
            10_000 + index * 1000,
            control_role(scenario),
        )
        for index, scenario in enumerate(
            ("both_correct", "outcome_correct", "treatment_correct", "both_wrong")
        )
    ]
    out.extend(
        Cell(
            "root_n_and_efficiency",
            f"n_{size}",
            "both_correct",
            size,
            RATE_REPLICATES,
            20_000 + index * 1000,
            "control" if index == 0 else "positive",
        )
        for index, size in enumerate(RATE_SIZES)
    )
    out.append(
        Cell(
            "interval_calibration",
            "correctly_specified",
            "both_correct",
            CALIBRATION_N,
            CALIBRATION_REPLICATES,
            30_000,
        )
    )
    out.extend(
        Cell(
            "double_robust_contraction",
            f"{scenario}_n{size}",
            scenario,
            size,
            CONTRACTION_REPLICATES[size_index],
            # What keeps the rungs' replication streams disjoint is `cell.cell` below, which
            # is part of the hashed label `stream_seed` reads, and not this offset. The
            # offsets are 1,000 apart and the rungs run 2,400 replications, so their index
            # ranges overlap: rung 0 spans 40,000..42,399 and rung 1 starts at 41,000. The
            # labels still differ, so the streams do. Disjointness matters because the ladder
            # is fitted across sizes, and a shared stream would correlate the rungs and narrow
            # the slope interval for a reason that has nothing to do with the estimator.
            #
            # `size_index` is positional, so inserting a rung shifts every rung above it and
            # silently redraws them. Appending a rung, or replacing the top one, is safe.
            40_000 + scenario_index * 3_000 + size_index * 1_000,
            control_role(scenario),
        )
        for scenario_index, scenario in enumerate(CONTRACTION_SCENARIOS)
        for size_index, size in enumerate(CONTRACTION_SIZES)
    )
    return tuple(out)


def _property_replicate(payload: tuple[Cell, int]) -> dict[str, Any]:
    cell, replicate = payload
    frame, reference_truth = draw_from_seed(
        cell.scenario,
        cell.n,
        stream_seed(
            STUDY,
            "property",
            cell.property,
            cell.cell,
            str(replicate + cell.seed_offset),
        ),
    )
    estimate = fit_cleverly(frame, cell.scenario).estimates["ate"]
    return replicate_row(
        property_name=cell.property,
        cell=cell.cell,
        role=cell.role,
        replicate=replicate,
        n=cell.n,
        requested=cell.replicates,
        truth=float(reference_truth["ate"]),
        estimate=estimate,
        alpha=STUDY.margins.alpha,
    )


def generate_property_rows(*, n_jobs: int = STUDY_JOBS) -> pd.DataFrame:
    payloads = [((cell, replicate),) for cell in cells() for replicate in range(cell.replicates)]
    rows = map_parallel(_property_replicate, payloads, n_jobs=n_jobs)
    return pd.DataFrame(rows).loc[:, list(REPLICATE_COLUMNS)]


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    return summarize_contraction_properties(
        rows, STUDY, verdict_replicates=CONTRACTION_VERDICT_REPLICATES
    )
