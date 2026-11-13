"""Property families of ``isotonic-binding`` (Fix-8 plan section 5.8), and their verdicts.

The cells are fixed before any run:

=========================================  ==========  =====  =====  ===========================
cell                                       role        n      R      rule
=========================================  ==========  =====  =====  ===========================
``interval_calibration/projected_t1__*``   positive    2,000  R      shared calibration rule,
                                           and two                   with the derived controls
                                           controls
``binding_witness/never_t1``               positive    2,000  R      share lower limit >= 0.25
``material_move/never_t1``                 positive    2,000  R      share lower limit >= 0.10
``reported_levels/<cell>``                 reported    2,000  R      none
``reported_large_move/<cell>``             reported    2,000  R      none
=========================================  ==========  =====  =====  ===========================

R is :data:`REPLICATES`.  Stream ``lg2`` holds one draw of law ``LG2`` per replicate and one fit
(:func:`tests.studies._isotonic_binding_law.fit`); every cell of the first three families and
of ``reported_levels`` reads that fit.  Stream ``lg2_gs2`` holds one draw of the larger-move
law ``LG2-gs2`` per replicate, and ``reported_large_move`` reads its fit.

A *move* is ``abs(projected - unprojected)`` in plug-in SEs of the level, which the projection
does not change.  A share cell's row holds the move in ``estimate`` against a truth of 0 with a
unit placeholder SE, and its ``covered`` holds the event the cell declares, so the exact
coverage interval is the share's Clopper-Pearson interval:

* ``binding_witness``: the never regimen's horizon-1 level moved by more than
  :data:`BINDING_SE` SE;
* ``material_move``: the same level moved by at least :data:`MATERIAL_SE` SE;
* ``se_present_*``: the level reports ``std_error`` (no constant node-2 target);
* ``stalled_fits``: a logistic fit of the replication stopped at ``max_iter``.  Its row holds
  the replication's count of such fits in ``estimate``, and the summary publishes the total in
  ``stalled_fits``.

The placeholder SE makes a share row's ``bias``, ``se_ratio`` and ``rejection_rate`` columns
meaningless, so a document cites only the columns of :data:`SHARE_COLUMNS` for the cells of
:data:`SHARE_CELLS` (the design test checks the evidence page).

**Rules.**

- ``interval_calibration`` reads the shared calibration rule
  (:func:`~tests.studies.evidence.property_verdicts.calibration_verdicts`) on the interval at
  the projected level, with the derived ``shrunken_se_control`` and ``noise_control``; the noise
  arm adds one unit of the exact efficiency bound (:data:`EFFICIENCY_SD`).  The study claims no
  efficiency band.
- ``binding_witness`` and ``material_move`` read the lower limit of their share
  (:func:`share_floor_verdicts`) against :data:`SHARE_FLOORS`.
- ``reported_levels`` and ``reported_large_move`` state no verdict
  (:data:`~tests.studies.evidence.property_verdicts.DIAGNOSTIC_ROLE`).  Horizon-2 coverage is
  read on the draws that report an SE, a selection, so it is never gated.

**Published columns.**  The calibration positive and ``reported_large_move/projected_t1`` publish
the share of draws on which the projected and unprojected intervals disagree
(``disagree_share``), the coverage difference with its exact conditional interval given the
discordant draws (``coverage_difference``), both coverages on moved draws, and the move
distribution.  Each horizon-2 projected cell publishes its coverage on the draws with an SE.

A replication that raises is not redrawn; the failure accounting of
``docs/development/method-benchmarking.md`` applies at :data:`FAILURE_BOUND`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import beta, norm

from tests.parallel import STUDY_JOBS
from tests.studies import _isotonic_binding_law as law
from tests.studies import canonical_isotonic_binding as study
from tests.studies.evidence import property_failures as pf
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.properties import (
    REPLICATE_COLUMNS,
    PropertyCell,
    control_row,
    payload_cell,
    payload_replicate,
    replicate_row,
    summary_interval,
)
from tests.studies.evidence.property_verdicts import (
    DIAGNOSTIC_ROLE,
    apply_shared_verdicts,
    calibration_controls,
    calibration_derived,
    calibration_verdicts,
    finish,
)
from tests.studies.evidence.replication import fit_properties
from tests.studies.evidence.seeds import stream_seed

STUDY = study.STUDY
FAILURE_BOUND = study.FAILURE_BOUND
CRITICAL = float(norm.ppf(1.0 - STUDY.margins.alpha / 2.0))
SHRUNKEN_SE_FACTOR = 0.70
N = 2_000
#: Every cell's replications: the arm-pooled predictive rule
#: (:func:`tests.studies.repeated_sizing.calibration_replicates`) on the committed pre-read
#: of ``projected_t1``, never below plan 5.0's single-cell floor of 2,400.  The design test
#: recomputes it.
REPLICATES = 3_600
#: A level binds above this move, and moves materially at this one, in SEs of the level.
BINDING_SE = 1e-6
MATERIAL_SE = 0.1
#: The least lower 99% Clopper-Pearson limit of each witness share (plan 5.8).
SHARE_FLOORS = {"binding_witness": 0.25, "material_move": 0.10}


def binds(move: float) -> bool:
    """Whether a move in SEs counts as binding: strictly above :data:`BINDING_SE`."""
    return bool(move > BINDING_SE)


def moves_materially(move: float) -> bool:
    """Whether a move in SEs is material: at least :data:`MATERIAL_SE`."""
    return bool(move >= MATERIAL_SE)


CALIBRATION_LABELS = study.CALIBRATION_LABELS
#: ``sqrt(E[D*^2])`` of the never regimen's horizon-1 risk on ``LG2``, which sizes the noise
#: arm (:func:`tests.studies._isotonic_binding_law.efficiency_sd`; the design test pins it).
EFFICIENCY_SD = {"projected_t1": 0.879628}
DERIVED_CELLS = calibration_derived(CALIBRATION_LABELS)

#: Each stream: its law and the cells its fit writes as ``(family, cell)``.
STREAMS: dict[str, tuple[str, tuple[tuple[str, str], ...]]] = {
    "lg2": (
        law.LAW,
        (
            ("interval_calibration", "projected_t1__correctly_specified"),
            ("binding_witness", "never_t1"),
            ("material_move", "never_t1"),
            *(("reported_levels", cell) for cell in study.PROPERTY_CELLS["reported_levels"]),
        ),
    ),
    "lg2_gs2": (
        law.LARGE_MOVE_LAW,
        tuple(
            ("reported_large_move", cell) for cell in study.PROPERTY_CELLS["reported_large_move"]
        ),
    ),
}

#: The families read on their share's lower limit.
SHARE_FAMILIES = tuple(SHARE_FLOORS)
#: The families that state no verdict.
REPORTED_FAMILIES = ("reported_levels", "reported_large_move")
#: The paired cells: the projected cell, its unprojected twin and its binding cell.
PAIRED = {
    ("interval_calibration", "projected_t1__correctly_specified"): (
        ("reported_levels", "unprojected_t1"),
        ("binding_witness", "never_t1"),
    ),
    ("reported_large_move", "projected_t1"): (
        ("reported_large_move", "unprojected_t1"),
        ("reported_large_move", "binding_t1"),
    ),
}
#: Each horizon-2 projected cell and the cell that says which draws report an SE.
WITH_SE = {
    ("reported_levels", "projected_t2"): ("reported_levels", "se_present_t2"),
    ("reported_levels", "unprojected_t2"): ("reported_levels", "se_present_t2"),
    ("reported_levels", "always_projected_t2"): ("reported_levels", "always_se_present_t2"),
}
#: The summary columns this study adds.
EXTRA_COLUMNS = (
    "disagree_share",
    "coverage_difference",
    "coverage_difference_ci_lower",
    "coverage_difference_ci_upper",
    "coverage_moved",
    "coverage_unprojected_moved",
    "mean_move_moved",
    "max_move",
    "coverage_with_se",
    "stalled_fits",
)
#: The cells whose row is a share with a placeholder SE, and the only summary columns of those
#: rows a document may cite.
SHARE_CELLS = (
    ("binding_witness", "never_t1"),
    ("material_move", "never_t1"),
    ("reported_levels", "se_present_t2"),
    ("reported_levels", "always_binding_t1"),
    ("reported_levels", "always_se_present_t2"),
    ("reported_levels", "stalled_fits"),
    ("reported_large_move", "binding_t1"),
    ("reported_large_move", "stalled_fits"),
)
SHARE_COLUMNS = frozenset(
    {
        "coverage",
        "coverage_ci_lower",
        "coverage_ci_upper",
        "passed",
        "property_passed",
        "n",
        "replicates",
        "failed_replicates",
        "stalled_fits",
    }
)
#: The cells that count each replication's stalled logistic fits.
STALLED_CELLS = (("reported_levels", "stalled_fits"), ("reported_large_move", "stalled_fits"))

#: The draw-level statistic the accounting reports for each lost replicate.
PROPERTY_WITNESS = property_verdicts.SAMPLE_WITNESS


def _role(family: str, cell: str) -> str:
    if family in REPORTED_FAMILIES:
        return DIAGNOSTIC_ROLE
    if cell.endswith("_control"):
        return "control"
    return "positive"


def sample_seed(stream: str, replicate: int) -> int:
    """The seed of one replication of a stream; every cell of the stream reads that draw."""
    return stream_seed(STUDY, "property_sample", stream, replicate)


#: The estimand key of a share cell's rows, whose truth is 0 (the move against no move).
SHARE_ESTIMAND = "share"
#: Each level cell's regimen and horizon; every other cell is a share.
LEVEL_CELLS = {
    "projected_t1": ("never", 1),
    "unprojected_t1": ("never", 1),
    "projected_t2": ("never", 2),
    "unprojected_t2": ("never", 2),
    "always_projected_t1": ("always", 1),
    "always_projected_t2": ("always", 2),
}


class DeclaredLaw:
    """The law a declared cell reads, with the truths its rows carry.

    Parameters
    ----------
    label : str
        :data:`tests.studies._isotonic_binding_law.LAW` or ``LARGE_MOVE_LAW``.
    """

    def __init__(self, label: str) -> None:
        self.label = label
        self.name = f"isotonic_binding__{label}"

    def __eq__(self, other: object) -> bool:
        """Equal to another declared law of the same label, or to the label itself."""
        if isinstance(other, DeclaredLaw):
            return self.label == other.label
        return self.label == other

    def __hash__(self) -> int:
        return hash(self.label)

    def truth(self) -> dict[str, float]:
        """Every level's quadrature truth (both laws share them) and the shares' 0."""
        return {**law.TRUTH, SHARE_ESTIMAND: 0.0}


def cell_estimand(cell: str) -> str:
    """The estimand key a cell's rows are read against: its level, or a share's 0."""
    level = LEVEL_CELLS.get(cell.split("__")[0])
    return SHARE_ESTIMAND if level is None else law.name(*level)


def declared_cells() -> tuple[PropertyCell, ...]:
    """Every sampled cell, with its role, size and replicate count."""
    cells = []
    for stream, (law_name, outputs) in STREAMS.items():
        names = [
            expanded
            for family, cell in outputs
            for expanded in (
                f"{family}/{cell}",
                *DERIVED_CELLS.get(f"{family}/{cell}", ()),
            )
        ]
        for name in names:
            family, cell = name.split("/", 1)
            cells.append(
                PropertyCell(
                    property=family,
                    cell=cell,
                    dgp=DeclaredLaw(law_name),
                    outcome_learner=lambda: None,
                    treatment_learner=lambda: None,
                    n=N,
                    replicates=REPLICATES,
                    seed=sample_seed(stream, 0),
                    role=_role(family, cell),
                    estimand=cell_estimand(cell),
                    fit_kwargs={"stream": stream},
                )
            )
    return tuple(cells)


_Key = tuple[tuple[str, str, int, int, int, int, str]]


def _draw(key: _Key) -> pd.DataFrame:
    """The sample a key fits, a function of the key alone."""
    _, stream, _, n, _, seed, _ = key[0]
    return law.draw(n, seed, STREAMS[stream][0])


def _cells_of(key: _Key) -> list[str]:
    names = [f"{family}/{cell}" for family, cell in STREAMS[key[0][1]][1]]
    return [expanded for name in names for expanded in (name, *DERIVED_CELLS.get(name, ()))]


def _share_row(
    family: str, cell: str, replicate: int, n: int, requested: int, value: float, event: bool
) -> dict[str, Any]:
    """A share cell's row: ``value`` against 0, a unit placeholder SE, the event as ``covered``."""
    return {
        "property": family,
        "cell": cell,
        "role": _role(family, cell),
        "replicate": replicate,
        "n": n,
        "requested_replicates": requested,
        "failed_replicates": 0,
        "truth": 0.0,
        "estimate": float(value),
        "std_error": 1.0,
        "covered": int(event),
        "rejected": int(event),
    }


def _fit_replication(key: _Key, frame: pd.DataFrame) -> list[dict[str, Any]]:
    _, stream, replicate, n, requested, _, _ = key[0]
    result, stalls = law.fit_counted(frame)
    level = law.levels(result)
    common = {"replicate": replicate, "n": n, "requested": requested}

    def interval(family: str, cell: str, regimen: str, horizon: int) -> dict[str, Any]:
        return replicate_row(
            property_name=family,
            cell=cell,
            role=_role(family, cell),
            truth=law.TRUTH[law.name(regimen, horizon)],
            estimate=result.estimates[law.name(regimen, horizon)],
            alpha=STUDY.margins.alpha,
            **common,
        )

    def unprojected(family: str, cell: str, regimen: str, horizon: int) -> dict[str, Any]:
        found = level[(regimen, horizon)]
        return control_row(
            property_name=family,
            cell=cell,
            truth=law.TRUTH[law.name(regimen, horizon)],
            estimate=found.unprojected,
            standard_error=found.std_error,
            critical=CRITICAL,
            role=_role(family, cell),
            **common,
        )

    def moved(
        family: str, cell: str, regimen: str, event: Callable[[float], bool]
    ) -> dict[str, Any]:
        move = level[(regimen, 1)].move
        return _share_row(family, cell, replicate, n, requested, move, event(move))

    def stalled(family: str) -> dict[str, Any]:
        return _share_row(family, "stalled_fits", replicate, n, requested, stalls, stalls > 0)

    def present(family: str, cell: str, regimen: str) -> dict[str, Any]:
        found = level[(regimen, 2)]
        return _share_row(family, cell, replicate, n, requested, found.move, found.has_se)

    if stream == "lg2_gs2":
        family = "reported_large_move"
        return [
            interval(family, "projected_t1", "never", 1),
            unprojected(family, "unprojected_t1", "never", 1),
            moved(family, "binding_t1", "never", binds),
            stalled(family),
        ]
    family = "reported_levels"
    return [
        interval("interval_calibration", "projected_t1__correctly_specified", "never", 1),
        moved("binding_witness", "never_t1", "never", binds),
        moved("material_move", "never_t1", "never", moves_materially),
        unprojected(family, "unprojected_t1", "never", 1),
        interval(family, "projected_t2", "never", 2),
        unprojected(family, "unprojected_t2", "never", 2),
        present(family, "se_present_t2", "never"),
        interval(family, "always_projected_t1", "always", 1),
        moved(family, "always_binding_t1", "always", binds),
        interval(family, "always_projected_t2", "always", 2),
        present(family, "always_se_present_t2", "always"),
        stalled(family),
    ]


def payloads(budget: int | None = None) -> list[_Key]:
    """Every replication key; ``budget`` caps each stream for a smoke run.

    A key is ``(("fit", stream, replicate, n, requested, seed, stream),)``; its systematic
    prefix reads the fit ``fit/<stream>``.
    """
    requested = REPLICATES if budget is None else min(budget, REPLICATES)
    return [
        (("fit", stream, replicate, N, requested, sample_seed(stream, replicate), stream),)
        for stream in STREAMS
        for replicate in range(requested)
    ]


def with_controls(rows: pd.DataFrame) -> pd.DataFrame:
    """The fitted rows and the calibration controls derived from them."""
    controls = calibration_controls(
        rows,
        STUDY,
        labels=CALIBRATION_LABELS,
        efficiency_bounds=EFFICIENCY_SD,
        calibration_n=N,
        shrunken_se_factor=SHRUNKEN_SE_FACTOR,
        critical=CRITICAL,
    )
    return pd.concat([rows, controls], ignore_index=True)


def generate_property_rows(*, n_jobs: int = STUDY_JOBS, budget: int | None = None) -> pd.DataFrame:
    """Fit every property replication; ``budget`` caps each stream for a smoke run."""
    outcomes = fit_properties(
        payloads(budget),
        _fit_replication,
        cells_of=_cells_of,
        cell_of=payload_cell,
        replicate_of=payload_replicate,
        n_jobs=n_jobs,
        draw=_draw,
        derived=DERIVED_CELLS,
    )
    rows = with_controls(pd.DataFrame([row for result in outcomes for row in result]))
    return rows.loc[:, list(REPLICATE_COLUMNS)].sort_values(
        ["property", "cell", "replicate"], ignore_index=True
    )


# ---------------------------------------------------------------------------- verdicts


def share_floor_verdicts(summary: pd.DataFrame, *, family: str, floor: float) -> None:
    """Read a share family: a positive passes when its exact lower limit is at least ``floor``.

    Parameters
    ----------
    summary : pandas.DataFrame
        The cell summary; this writes ``passed`` on the family's rows.
    family : str
        The property family.
    floor : float
        The least lower limit.

    Raises
    ------
    ValueError
        When a row of the family is not a positive.
    """
    mask = summary["property"] == family
    for index in summary.index[mask]:
        if str(summary.loc[index, "role"]) != "positive":
            raise ValueError(f"{family} has a {summary.loc[index, 'role']!r} row")
        coverage = summary_interval(summary, index, "coverage")
        summary.loc[index, "passed"] = bool(coverage.low >= floor)


def share_floor_rule(family: str, *, floor: float) -> pf.FamilyRule:
    """The rule of :func:`share_floor_verdicts`: S2 on the lower limit of each positive."""
    return pf.FamilyRule(
        families=frozenset({family}),
        shared="share_floor_verdicts",
        clauses=(
            pf.proportion_clause(
                "share floor",
                statistic="S2",
                judge=pf.floor_judge(lambda ctx: floor),
                applies=pf.roles("positive"),
            ),
        ),
    )


def _cell(rows: pd.DataFrame, family: str, cell: str) -> pd.DataFrame:
    selected = rows.loc[(rows["property"] == family) & (rows["cell"] == cell)]
    return selected.set_index("replicate").sort_index()


def _locate(summary: pd.DataFrame, family: str, cell: str) -> Any:
    mask = (summary["property"] == family) & (summary["cell"] == cell)
    if int(mask.sum()) != 1:
        raise ValueError(f"{family}/{cell} is not one summary row")
    return summary.index[mask][0]


def coverage_difference(
    projected: Sequence[int], unprojected: Sequence[int], confidence_level: float
) -> tuple[float, float, float]:
    """The paired coverage difference and its exact interval given the discordant draws.

    With ``b`` draws covered only at the projected level and ``c`` only at the unprojected one,
    ``b`` given ``b + c`` is binomial, so the Clopper-Pearson interval of its share ``p`` gives
    the interval ``(b + c) (2 p - 1) / R`` of the difference ``(b - c) / R``.
    """
    first = np.asarray(projected, dtype=bool)
    second = np.asarray(unprojected, dtype=bool)
    gained = int(np.sum(first & ~second))
    lost = int(np.sum(~first & second))
    discordant = gained + lost
    total = len(first)
    point = (gained - lost) / total
    if discordant == 0:
        return point, 0.0, 0.0
    tail = (1.0 - confidence_level) / 2.0
    low = 0.0 if gained == 0 else float(beta.ppf(tail, gained, discordant - gained + 1))
    high = (
        1.0 if gained == discordant else float(beta.ppf(1 - tail, gained + 1, discordant - gained))
    )
    scale = discordant / total
    return point, scale * (2.0 * low - 1.0), scale * (2.0 * high - 1.0)


def paired_columns(summary: pd.DataFrame, rows: pd.DataFrame) -> None:
    """Publish each projected cell's agreement with its unprojected twin and its move sizes."""
    level = STUDY.margins.confidence_level
    for (family, cell), ((twin_family, twin), (move_family, move_cell)) in PAIRED.items():
        projected = _cell(rows, family, cell)
        other = _cell(rows, twin_family, twin)
        moves = _cell(rows, move_family, move_cell)
        shared = projected.index.intersection(other.index).intersection(moves.index)
        first = projected.loc[shared, "covered"].to_numpy(dtype=bool)
        second = other.loc[shared, "covered"].to_numpy(dtype=bool)
        moved = moves.loc[shared, "covered"].to_numpy(dtype=bool)
        size = moves.loc[shared, "estimate"].to_numpy(dtype=float)
        index = _locate(summary, family, cell)
        point, low, high = coverage_difference(first, second, level)
        summary.loc[index, "disagree_share"] = float(np.mean(first != second))
        summary.loc[index, "coverage_difference"] = point
        summary.loc[index, "coverage_difference_ci_lower"] = low
        summary.loc[index, "coverage_difference_ci_upper"] = high
        summary.loc[index, "max_move"] = float(size.max()) if len(size) else np.nan
        if moved.any():
            summary.loc[index, "coverage_moved"] = float(first[moved].mean())
            summary.loc[index, "coverage_unprojected_moved"] = float(second[moved].mean())
            summary.loc[index, "mean_move_moved"] = float(size[moved].mean())


def with_se_columns(summary: pd.DataFrame, rows: pd.DataFrame) -> None:
    """Publish each horizon-2 cell's coverage on the draws that report an SE."""
    for (family, cell), (present_family, present_cell) in WITH_SE.items():
        values = _cell(rows, family, cell)
        present = _cell(rows, present_family, present_cell)
        shared = values.index.intersection(present.index)
        keep = present.loc[shared, "covered"].to_numpy(dtype=bool)
        if keep.any():
            covered = values.loc[shared, "covered"].to_numpy(dtype=bool)[keep]
            summary.loc[_locate(summary, family, cell), "coverage_with_se"] = float(covered.mean())


def stalled_columns(summary: pd.DataFrame, rows: pd.DataFrame) -> None:
    """Publish each stream's total count of logistic fits that stopped at ``max_iter``."""
    for family, cell in STALLED_CELLS:
        counts = _cell(rows, family, cell)["estimate"].to_numpy(dtype=float)
        summary.loc[_locate(summary, family, cell), "stalled_fits"] = float(counts.sum())


def reported_verdicts(summary: pd.DataFrame) -> None:
    """Publish the reported families' rows, which state no verdict."""
    mask = summary["property"].isin(REPORTED_FAMILIES)
    roles = set(summary.loc[mask, "role"].astype(str))
    if roles - {DIAGNOSTIC_ROLE}:
        raise ValueError(f"a reported family publishes roles {sorted(roles)}")
    summary.loc[mask, "passed"] = True
    summary.loc[mask, "property_passed"] = True


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The calibration and share verdicts; the paired and reported columns published."""
    summary, rates = apply_shared_verdicts(rows, STUDY, extra_columns=EXTRA_COLUMNS, rate_labels=())
    calibration_verdicts(summary, margins=STUDY.margins, efficiency_band=None)
    for family, floor in SHARE_FLOORS.items():
        share_floor_verdicts(summary, family=family, floor=floor)
    paired_columns(summary, rows)
    with_se_columns(summary, rows)
    stalled_columns(summary, rows)
    reported_verdicts(summary)
    return finish(summary, rates)


#: The failure accounting's declared rules.  The calibration cells read the shared rule
#: (:data:`~tests.studies.evidence.property_verdicts.FAMILY_RULES`).
PROPERTY_RULES: tuple[Any, ...] = tuple(
    share_floor_rule(family, floor=floor) for family, floor in SHARE_FLOORS.items()
)


def smoke_keys(draws: int) -> list[_Key]:
    """Keys of a smoke on the first ``draws`` replicates of every stream."""
    return payloads(draws)


def fit_keys(keys: list[_Key]) -> list[list[dict[str, Any]]]:
    """Fit keys in this process, for a smoke outside the driver."""
    return [_fit_replication(key, _draw(key)) for key in keys]


def gated_cells() -> tuple[str, ...]:
    """The ``family/cell`` names whose red verdict gates the study."""
    return tuple(
        f"{cell.property}/{cell.cell}" for cell in declared_cells() if cell.role != DIAGNOSTIC_ROLE
    )


__all__ = [
    "DERIVED_CELLS",
    "EFFICIENCY_SD",
    "FAILURE_BOUND",
    "PROPERTY_RULES",
    "PROPERTY_WITNESS",
    "REPLICATES",
    "SHARE_CELLS",
    "SHARE_COLUMNS",
    "SHARE_FLOORS",
    "STREAMS",
    "declared_cells",
    "gated_cells",
    "generate_property_rows",
    "summarize_properties",
]
