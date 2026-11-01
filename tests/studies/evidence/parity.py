"""Exact parity gates between ``cleverly`` and a comparator that computes the same number (X59).

A margins comparison (:mod:`tests.studies.evidence.comparison`) asks whether two implementations
are similar and no worse.  Where the two compute the same estimate by construction -- the same
regressions on the same rows, the same targeting and the same ratios -- the stronger claim is
numerical parity, replicate by replicate.  This module holds that claim and the arithmetic
behind the X59 comparator runs of the three long-horizon studies.

Three gates read per-replicate absolute differences, each against a tolerance declared before the
run (``X59-plan.md`` section 4.2):

``G-P``
    ``cleverly``'s targeted estimate and standard error against R ``ltmle``'s.
``G-C``
    R ``ltmle(gcomp = TRUE)`` against ``cleverly``'s untargeted sequential regression
    (:func:`tests.studies.long_horizon_common.untargeted_plugin`): the regressions alone, with
    no targeting step, so a wrong regression formula cannot hide behind a fluctuation.
``G1``
    ``cleverly`` against R ``lmtp`` at an all-policy plan, estimate and standard error.

One replicate over its tolerance fails the gate.  A ``witness`` row per G-P cell records the RMS
targeting move (the committed estimate minus the untargeted plug-in) against
:data:`WITNESS_FACTOR` times the G-P tolerance: a scenario whose cells all move less than that
carries no targeting claim, because an update that does almost nothing passes parity whether or
not either implementation targets correctly.

``parity.csv`` holds one row per cell and quantity; ``parity-replicates.csv`` holds the G-C and
witness inputs that ``replicates.parquet`` does not carry.  The fast tier recomputes the first
from the second and the replicate file (``tests/unit/test_x59_parity.py``).

The section 6 sizing rule lives here too: it takes only measured timings and a memory-gated
worker count, never a gate value, and returns the replicate prefix of every scenario.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from tests.studies.evidence.registry import StudyRecord

#: The parity gates, in the order ``parity.csv`` lists them.
GATES = ("G-P", "G-C", "G1")
#: The absolute floor of a declared tolerance: five times the largest shipped two-node parity
#: difference, 2e-7 (``tests/canonical/ltmle_regimen_adapter.R``).
PARITY_FLOOR = 1e-6
#: The G1 tolerance, fixed absolute by the plan (``X59-plan.md`` section 4.2).  It is not a
#: declared tolerance: no smoke maximum moves it, and a declaration that names it is refused.
G1_TOLERANCE = 1e-6
#: The largest G-P or G-C tolerance a smoke may declare.  Two implementations of one estimator
#: on the same regressions and the same supplied mechanism differ by their solvers' stopping
#: rules: about 1e-8 here (IRLS to 1e-10 against ``glm``'s 1e-8 relative deviance), and at most
#: 2e-7 on the shipped two-node pairs.  A tolerance above 1e-5 needs a smoke maximum above 1e-6,
#: five times that shipped maximum, which says the two sides do not compute one thing; and 1e-5
#: is under one percent of the smallest per-replicate standard error at the declared sizes
#: (about 1.4e-3 for a binary mean at n = 128,000).  A smoke over it fails; it is not declared.
TOLERANCE_CEILING = 1e-5
#: The gates whose tolerance the declaration commit sets from the smoke maxima.
DECLARED_GATES = ("G-P", "G-C")
#: A smoke maximum is scaled by this to give the declared G-P and G-C tolerance.
SMOKE_SCALE = 10.0
#: A G-P cell carries a targeting claim when its RMS move is at least this many tolerances.
WITNESS_FACTOR = 100.0
#: A declared mutation must move some gate by at least this many tolerances.
MUTATION_FACTOR = 10.0

#: The columns of ``parity.csv``.
PARITY_COLUMNS = (
    "gate",
    "scenario",
    "estimand",
    "quantity",
    "paired_replicates",
    "max_abs_difference",
    "max_replicate",
    "rms",
    "tolerance",
    "passed",
)
#: The columns of ``parity-replicates.csv``: per replicate, what the gates need beyond the rows.
INPUT_COLUMNS = ("scenario", "replicate", "estimand", "subject_plugin", "reference_gcomp")


@dataclass(frozen=True)
class ReferenceSample:
    """One replicate's comparator input, built from its seed without a ``cleverly`` fit.

    Parameters
    ----------
    frame : pandas.DataFrame
        The sample with every column the comparator reads: the data, the law's per-node
        probabilities and every plan input, so the R side re-implements no rule.
    truths : dict
        Estimand -> its exact value.
    plugins : dict
        Estimand -> ``cleverly``'s untargeted plug-in on this sample, for G-C and the witness
        rows; empty for a study that declares neither.
    """

    frame: pd.DataFrame
    truths: Mapping[str, float]
    plugins: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ParityCell:
    """One gated cell.

    Parameters
    ----------
    gate : str
        ``"G-P"``, ``"G-C"`` or ``"G1"``.
    scenario : str
        The scenario.
    estimand : str
        The estimand, as ``replicates.parquet`` names it.
    """

    gate: str
    scenario: str
    estimand: str

    def __post_init__(self) -> None:
        if self.gate not in GATES:
            raise ValueError(f"unknown parity gate {self.gate!r}; expected one of {GATES}")


@dataclass(frozen=True)
class ParityDeclaration:
    """The parity cells of one study and the tolerance of each gate.

    Parameters
    ----------
    cells : tuple of ParityCell
        Every gated cell.
    tolerance : dict
        G-P and G-C -> the declared tolerance.  ``None`` until the declaration commit fixes it
        from the smoke maxima (:func:`declared_tolerance`); a gate without one refuses to read.
        G1 is fixed at :data:`G1_TOLERANCE` and is not declared here.
    """

    cells: tuple[ParityCell, ...]
    tolerance: Mapping[str, float | None] = field(
        default_factory=lambda: {"G-P": None, "G-C": None}
    )

    def __post_init__(self) -> None:
        unknown = set(self.tolerance) - set(DECLARED_GATES)
        if unknown:
            raise ValueError(
                f"{sorted(unknown)} take no declared tolerance: G1 is fixed at {G1_TOLERANCE:g}"
            )
        for gate, value in self.tolerance.items():
            if value is not None and not 0.0 < float(value) <= TOLERANCE_CEILING:
                raise ValueError(
                    f"the {gate} tolerance {value} is outside (0, {TOLERANCE_CEILING:g}]"
                )

    def tau(self, gate: str) -> float:
        """The tolerance of ``gate``; refuses a declared gate not yet declared."""
        if gate == "G1":
            return G1_TOLERANCE
        value = self.tolerance.get(gate)
        if value is None:
            raise ValueError(
                f"the {gate} tolerance is not declared yet: the declaration commit sets it from "
                "the smoke maxima before any run reads the gate"
            )
        return float(value)

    def gates(self) -> tuple[str, ...]:
        """The gates the declaration holds, in :data:`GATES` order."""
        present = {cell.gate for cell in self.cells}
        return tuple(gate for gate in GATES if gate in present)

    def check_declared(self) -> None:
        """Refuse, before any run reads a gate, a gate of these cells without its tolerance."""
        for gate in self.gates():
            self.tau(gate)


def smoke_declaration(declaration: ParityDeclaration) -> ParityDeclaration:
    """``declaration`` with every undeclared tolerance at :data:`PARITY_FLOOR`, for a smoke run.

    A smoke run (``--reference-prefix``) exists to measure the maxima the declaration commit
    reads, before any tolerance is declared.  Its table is written at the floor so the maxima
    are on disk; its verdicts are not read, and the driver refuses a smoke run into the
    committed directory.
    """
    tolerance = {
        gate: PARITY_FLOOR if value is None else value
        for gate, value in declaration.tolerance.items()
    }
    return ParityDeclaration(cells=declaration.cells, tolerance=tolerance)


def declared_tolerance(smoke_maximum: float) -> float:
    """``max(PARITY_FLOOR, SMOKE_SCALE * smoke_maximum)``, the M1 rule for G-P and G-C.

    Refuses a tolerance above :data:`TOLERANCE_CEILING`: such a smoke is a failed smoke.
    """
    if not math.isfinite(smoke_maximum) or smoke_maximum < 0.0:
        raise ValueError(f"a smoke maximum must be finite and non-negative; got {smoke_maximum}")
    tolerance = max(PARITY_FLOOR, SMOKE_SCALE * float(smoke_maximum))
    if tolerance > TOLERANCE_CEILING:
        raise ValueError(
            f"a smoke maximum of {smoke_maximum:.3g} would declare {tolerance:.3g}, above the "
            f"ceiling {TOLERANCE_CEILING:g}: the smoke fails, diagnose before any declaration"
        )
    return tolerance


def _paired(rows: pd.DataFrame, record: StudyRecord, scenario: str, estimand: str) -> pd.DataFrame:
    """Subject and reference columns of one cell, on the replicates both report."""
    cell = rows.loc[(rows["scenario"] == scenario) & (rows["estimand"] == estimand)]
    wide = cell.pivot(
        index="replicate", columns="implementation", values=["estimate", "std_error"]
    ).sort_index()
    subject, reference = record.implementation, str(record.reference)
    for side in (subject, reference):
        if ("estimate", side) not in wide.columns:
            raise ValueError(f"{record.slug}: {scenario}/{estimand} has no {side} rows")
    return wide.dropna()


def _row(
    gate: str,
    scenario: str,
    estimand: str,
    quantity: str,
    difference: pd.Series,
    tolerance: float,
) -> dict[str, Any]:
    absolute = difference.abs()
    if absolute.empty:
        raise ValueError(f"{gate} {scenario}/{estimand}: no paired replicate to read")
    worst = int(absolute.idxmax())
    largest = float(absolute.max())
    return {
        "gate": gate,
        "scenario": scenario,
        "estimand": estimand,
        "quantity": quantity,
        "paired_replicates": int(absolute.size),
        "max_abs_difference": largest,
        "max_replicate": worst,
        "rms": float(np.sqrt(np.mean(np.square(difference.to_numpy(dtype=float))))),
        "tolerance": tolerance,
        "passed": bool(largest <= tolerance),
    }


def parity_table(
    rows: pd.DataFrame,
    inputs: pd.DataFrame | None,
    *,
    record: StudyRecord,
    declaration: ParityDeclaration,
) -> pd.DataFrame:
    """``parity.csv``: one row per declared cell and quantity, then the witness rows.

    ``rows`` is ``replicates.parquet``; ``inputs`` is ``parity-replicates.csv``, which G-C and
    the witness rows read.  A G-P or G1 cell reads the estimate and the standard error.  A
    witness row's ``max_abs_difference`` is the largest targeting move and its ``rms`` the RMS
    move, against :data:`WITNESS_FACTOR` times the G-P tolerance.
    """
    if record.reference is None:
        raise ValueError(f"{record.slug} has no reference to read parity against")
    subject, reference = record.implementation, str(record.reference)
    out: list[dict[str, Any]] = []
    keyed = None
    if inputs is not None and not inputs.empty:
        keyed = inputs.set_index(["scenario", "estimand", "replicate"]).sort_index()
    for cell in declaration.cells:
        tau = declaration.tau(cell.gate)
        if cell.gate in ("G-P", "G1"):
            wide = _paired(rows, record, cell.scenario, cell.estimand)
            for quantity in ("estimate", "std_error"):
                difference = wide[(quantity, subject)] - wide[(quantity, reference)]
                out.append(_row(cell.gate, cell.scenario, cell.estimand, quantity, difference, tau))
            continue
        if keyed is None:
            raise ValueError(f"{record.slug}: G-C needs parity-replicates.csv")
        here = keyed.loc[(cell.scenario, cell.estimand)]
        difference = (here["reference_gcomp"] - here["subject_plugin"]).dropna()
        out.append(_row("G-C", cell.scenario, cell.estimand, "gcomp", difference, tau))
    if keyed is not None and "G-P" in declaration.gates():
        tau = declaration.tau("G-P")
        for cell in declaration.cells:
            if cell.gate != "G-P":
                continue
            committed = rows.loc[
                (rows["implementation"] == subject)
                & (rows["scenario"] == cell.scenario)
                & (rows["estimand"] == cell.estimand)
            ].set_index("replicate")["estimate"]
            plugin = keyed.loc[(cell.scenario, cell.estimand), "subject_plugin"]
            move = (committed.reindex(plugin.index) - plugin).dropna()
            row = _row("witness", cell.scenario, cell.estimand, "targeting_move", move, tau)
            row["tolerance"] = WITNESS_FACTOR * tau
            row["passed"] = bool(row["rms"] >= row["tolerance"])
            out.append(row)
    return pd.DataFrame.from_records(out, columns=list(PARITY_COLUMNS))


def witness_scenarios(table: pd.DataFrame) -> dict[str, bool]:
    """Scenario -> whether one of its G-P cells moves enough to carry a targeting claim."""
    witness = table.loc[table["gate"] == "witness"]
    return {str(name): bool(group["passed"].any()) for name, group in witness.groupby("scenario")}


def failed_gates(table: pd.DataFrame) -> pd.DataFrame:
    """The gate rows over their tolerance (witness rows state a claim's reach, not a gate)."""
    return table.loc[(table["gate"] != "witness") & ~table["passed"].astype(bool)]


def smoke_maxima(table: pd.DataFrame) -> dict[str, float]:
    """Gate -> the largest difference a smoke table reads, for :func:`declared_tolerance`."""
    gated = table.loc[table["gate"] != "witness"]
    return {
        str(gate): float(group["max_abs_difference"].max()) for gate, group in gated.groupby("gate")
    }


def mutation_detected(table: pd.DataFrame, declaration: ParityDeclaration) -> bool:
    """Whether a mutated smoke run moves some gate by at least ``MUTATION_FACTOR`` tolerances."""
    gated = table.loc[table["gate"] != "witness"]
    moved = [
        float(row.max_abs_difference) >= MUTATION_FACTOR * declaration.tau(str(row.gate))
        for row in gated.itertuples(index=False)
    ]
    return any(moved)


# ---- the section 6 sizing rule ----------------------------------------------------------------

#: The tier-2 (exact cells) prefixes the rule chooses from, smallest first.
TIER2_COUNTS = (50, 100, 150, 200, 300)
#: The tier-4 (descriptive pairs) prefixes.
TIER4_COUNTS = (25, 50, 100)
#: Wall hours of the run, after the 3-hour reserve of the 24-hour cap.
RUN_HOURS = 21.0
#: The share of the budget left after tiers 1 and 3 that tier 2 may take.
TIER2_SHARE = 0.65
#: The most workers the laptop offers.
MAX_WORKERS = 12
#: The memory factor of the M7 worker rule.
MEMORY_FACTOR = 1.25


def worker_count(container_mb: float, peak_mb: float, *, cores: int = MAX_WORKERS) -> int:
    """``w = min(cores, floor(container memory / (1.25 x peak)))``, at least one (M7)."""
    if peak_mb <= 0.0 or container_mb <= 0.0:
        raise ValueError("the container memory and the measured peak must be positive")
    return max(1, min(int(cores), math.floor(container_mb / (MEMORY_FACTOR * peak_mb))))


@dataclass(frozen=True)
class Sizing:
    """What the sizing rule chose, and the arithmetic behind it.

    Parameters
    ----------
    workers : int
        The memory-gated worker count.
    budget : float
        ``workers x RUN_HOURS``, in CPU-hours.
    tier2 : int
        The tier-2 prefix, or 0 when not even the smallest fits.
    tier4 : int
        The tier-4 prefix, or 0 when not even the smallest fits.
    cost : float
        The CPU-hours of tiers 2 and 4 at those prefixes.
    counts : dict
        Scenario -> its prefix; a scenario whose tier fits no prefix is absent.
    """

    workers: int
    budget: float
    tier2: int
    tier4: int
    cost: float
    counts: Mapping[str, int]

    @property
    def wall_hours(self) -> float:
        """The wall time of tiers 2 and 4 on :attr:`workers`."""
        return self.cost / self.workers


def reference_sizes(
    timings: Mapping[str, float],
    tiers: Mapping[str, int],
    *,
    workers: int,
    tier1_cpu_hours: float,
    tier3_cpu_hours: float,
) -> Sizing:
    """The section 6 rule: the prefix of every paired scenario, from smoke timings alone.

    ``timings`` maps a scenario to its measured R CPU-hours per replicate; ``tiers`` maps it to
    2 (holds an exact cell) or 4 (descriptive pairs only).  Tier 2 takes the largest count in
    :data:`TIER2_COUNTS` whose cost is at most ``TIER2_SHARE`` of what tiers 1 and 3 leave of
    ``workers x RUN_HOURS``; tier 4 the largest in :data:`TIER4_COUNTS` that fits the rest.
    No gate or margins value enters.
    """
    unknown = set(tiers) - set(timings)
    if unknown:
        raise ValueError(f"no timing for {sorted(unknown)}")
    if any(value not in (2, 4) for value in tiers.values()):
        raise ValueError("a scenario's tier is 2 (exact cells) or 4 (descriptive pairs)")
    if any(not math.isfinite(timings[name]) or timings[name] <= 0.0 for name in tiers):
        raise ValueError("every timing must be a positive number of CPU-hours")
    budget = workers * RUN_HOURS
    left = budget - tier1_cpu_hours - tier3_cpu_hours
    per = {
        tier: sum(timings[name] for name, value in tiers.items() if value == tier)
        for tier in (2, 4)
    }
    tier2 = _largest(TIER2_COUNTS, per[2], TIER2_SHARE * left)
    tier4 = _largest(TIER4_COUNTS, per[4], left - tier2 * per[2])
    counts = {
        name: count
        for name, value in tiers.items()
        for count in ((tier2 if value == 2 else tier4),)
        if count
    }
    return Sizing(
        workers=int(workers),
        budget=budget,
        tier2=tier2,
        tier4=tier4,
        cost=tier2 * per[2] + tier4 * per[4],
        counts=counts,
    )


def _largest(options: Sequence[int], per_replicate: float, room: float) -> int:
    fitting = [count for count in options if count * per_replicate <= room]
    return max(fitting) if fitting else 0
