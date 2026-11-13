"""Property families of ``option-patience`` (Fix-8 plan section 5.5), and their verdicts.

One draw of each stream (``lc1``, ``lc2_lasso``, ``lc2_learners``, ``lc3``) is fitted once per
declared fit (:data:`tests.studies._option_patience_laws.STREAM_FITS`), and every cell of the
stream reads those fits, so a patient fit and its whole-path fit always share their draw.  A
cell's *reading* fixes what its row holds:

============  ===============================================================================
reading       the row
============  ===============================================================================
``estimate``  the fit's ATE against the exact truth; ``std_error`` and ``covered`` are the
              plug-in diagnostic of the working mechanism, which makes no coverage claim
``stopped``   ``covered`` is 1 when the patience counter ended the path before the whole
              path's end, so the exact coverage interval is the stop share's
``excludes``  ``covered`` is 1 when the reported treatment model leaves out ``W1``
``paired``    ``estimate`` is the patient fit's ATE minus the whole-path fit's on the same
              draw and ``truth`` is 0; ``std_error`` is the whole-path plug-in diagnostic and
              ``covered`` is 1 when the two estimates are identical
============  ===============================================================================

**Rules.**

- ``patience_bias``, ``ordered_late_confounder`` and ``ordered_whole_path`` read the shared
  bias rule by role (:func:`~tests.studies.evidence.property_verdicts.bias_role_verdicts`):
  a positive's 99% bias interval inside 0.25 of its own SD, a control's beyond it.
- ``patience_witness`` and ``late_confounder_stop`` read the exact interval of their share
  (:func:`coverage_limit_verdicts`): the lower limit at least
  :data:`~tests.studies.canonical_option_patience.WITNESS_SHARE`.
- ``greedy_late_confounder`` reads the 99% Student interval of the mean paired difference
  against 0.25 SD of the greedy whole-path estimates on the same draws (:data:`PAIRED_MARGIN`,
  :func:`paired_verdicts`).  The default margin, 0.25 SD of the differences themselves, would
  shrink with them: the two fits agree on most draws.
- Every ``reported_*`` family states no verdict.  Each paired cell publishes
  ``paired_over_whole_sd`` and the stop-share cell of LC3's ordered fit publishes the bias on
  the draws that exclude ``W1`` (``conditional_bias``, ``conditional_bias_over_sd``), in SDs
  of the ordered whole path.

A replication that raises is not redrawn; the failure accounting of
``docs/development/method-benchmarking.md`` applies at :data:`FAILURE_BOUND`.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies import _option_patience_laws as law
from tests.studies import canonical_option_patience as study
from tests.studies.evidence import property_failures as pf
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.properties import (
    REPLICATE_COLUMNS,
    PropertyCell,
    replicate_row,
    summary_interval,
)
from tests.studies.evidence.property_verdicts import (
    DIAGNOSTIC_ROLE,
    apply_shared_verdicts,
    bias_role_rule,
    bias_role_verdicts,
    finish,
)
from tests.studies.evidence.replication import fit_properties
from tests.studies.evidence.schema import reported_inference
from tests.studies.evidence.seeds import stream_seed

STUDY = study.STUDY
FAILURE_BOUND = study.FAILURE_BOUND
PROPERTY_CELLS = study.PROPERTY_CELLS
REPLICATES = study.REPLICATES
N = law.N
TARGET = "ate"
#: The paired positive's margin, in SDs of the whole-path estimates (plan 5.5).
PAIRED_MARGIN = 0.25
READINGS = ("estimate", "stopped", "excludes", "paired")

#: The families read by role on their bias.
BIAS_FAMILIES = ("patience_bias", "ordered_late_confounder", "ordered_whole_path")
#: The families read on an exact share's lower limit.
SHARE_FAMILIES = ("patience_witness", "late_confounder_stop")
#: The paired family.
PAIRED_FAMILY = "greedy_late_confounder"
#: The families that state no verdict.
REPORTED_FAMILIES = ("reported_estimate", "reported_stop", "reported_displacement")
#: The summary columns this study adds.
EXTRA_COLUMNS = (
    "paired_over_whole_sd",
    "identical_share",
    "conditional_bias",
    "conditional_bias_over_sd",
)

#: The draw-level statistic the accounting reports for each lost replicate.
PROPERTY_WITNESS = property_verdicts.SAMPLE_WITNESS


@dataclass(frozen=True)
class Cell:
    """One declared cell: the fits it reads, how, and the role its verdict takes.

    Parameters
    ----------
    family : str
        The property family.
    name : str
        The cell name.
    role : str
        ``positive``, ``control`` or the diagnostic role.
    reading : str
        One of :data:`READINGS`.
    fit : str
        The fit the cell reads, a key of :data:`tests.studies._option_patience_laws.FITS`.
    whole : str or None
        The whole-path fit on the same draw, for the ``stopped`` and ``paired`` readings.
    """

    family: str
    name: str
    role: str
    reading: str
    fit: str
    whole: str | None = None

    def __post_init__(self) -> None:
        if self.reading not in READINGS:
            raise ValueError(f"{self.name}: reading {self.reading!r} is not one of {READINGS}")
        if (self.reading in ("stopped", "paired")) != (self.whole is not None):
            raise ValueError(f"{self.name}: the {self.reading} reading and its whole path")

    @property
    def stream(self) -> str:
        """The stream whose draws the cell reads."""
        return law.FITS[self.fit].stream

    @property
    def key(self) -> str:
        """``family/cell``."""
        return f"{self.family}/{self.name}"


def _whole(fit: str) -> str:
    return fit.removesuffix("_patience") + "_whole"


def _cells() -> tuple[Cell, ...]:
    gated = (
        Cell("patience_bias", "lc1_greedy", "positive", "estimate", "lc1_greedy_patience"),
        Cell(
            "patience_witness",
            "lc1_greedy",
            "positive",
            "stopped",
            "lc1_greedy_patience",
            "lc1_greedy_whole",
        ),
        Cell(
            "ordered_late_confounder", "lc3_ordered", "control", "estimate", "lc3_ordered_patience"
        ),
        Cell("late_confounder_stop", "lc3_ordered", "positive", "excludes", "lc3_ordered_patience"),
        Cell("ordered_whole_path", "lc3_ordered", "positive", "estimate", "lc3_ordered_whole"),
        Cell(
            PAIRED_FAMILY,
            "lc3_greedy",
            "positive",
            "paired",
            "lc3_greedy_patience",
            "lc3_greedy_whole",
        ),
    )
    reported = (
        *(
            Cell("reported_estimate", name, DIAGNOSTIC_ROLE, "estimate", name)
            for name in PROPERTY_CELLS["reported_estimate"]
        ),
        *(
            Cell(
                "reported_stop",
                label,
                DIAGNOSTIC_ROLE,
                "stopped",
                f"{label}_patience",
                f"{label}_whole",
            )
            for label in PROPERTY_CELLS["reported_stop"]
        ),
        *(
            Cell(
                "reported_displacement",
                label,
                DIAGNOSTIC_ROLE,
                "paired",
                f"{label}_patience",
                f"{label}_whole",
            )
            for label in PROPERTY_CELLS["reported_displacement"]
        ),
    )
    return (*gated, *reported)


#: Every declared cell.
CELLS = _cells()
#: Each paired cell's whole-path estimate cell, whose SD is the paired margin's unit.
WHOLE_CELLS: dict[str, tuple[str, str]] = {
    cell.key: ("reported_estimate", str(cell.whole)) for cell in CELLS if cell.reading == "paired"
}
#: The LC3 ordered whole path, whose SD is the conditional bias's unit.
WHOLE_PATH_CELL = ("ordered_whole_path", "lc3_ordered")


# ---------------------------------------------------------------------------- rows


def row(
    cell: Cell,
    fits: Mapping[str, Any],
    truth: Mapping[str, float],
    *,
    replicate: int,
    requested: int,
    alpha: float,
) -> dict[str, Any]:
    """The replicate row of one cell on one draw's fits."""
    result = fits[cell.fit]
    out = replicate_row(
        property_name=cell.family,
        cell=cell.name,
        role=cell.role,
        replicate=replicate,
        n=N,
        requested=requested,
        truth=float(truth[TARGET]),
        estimate=result[TARGET],
        alpha=alpha,
    )
    if cell.reading == "stopped":
        out["covered"] = int(law.stopped(result, fits[str(cell.whole)]))
        out["rejected"] = 0
    elif cell.reading == "excludes":
        out["covered"] = int(law.excludes_confounder(result))
        out["rejected"] = 0
    elif cell.reading == "paired":
        whole = fits[str(cell.whole)]
        difference = law.ate(result) - law.ate(whole)
        std_error, _, _ = reported_inference(whole[TARGET])
        out.update(
            truth=0.0,
            estimate=difference,
            std_error=std_error,
            covered=int(difference == 0.0),
            rejected=0,
        )
    return out


def stream_rows(
    stream: str,
    frame: pd.DataFrame,
    truth: Mapping[str, float],
    *,
    replicate: int,
    requested: int,
    alpha: float,
) -> list[dict[str, Any]]:
    """Every row of one draw of a stream: each fit once, every cell of the stream off it."""
    fits = law.fit_stream(frame, stream)
    return [
        row(cell, fits, truth, replicate=replicate, requested=requested, alpha=alpha)
        for cell in CELLS
        if cell.stream == stream
    ]


_Payload = tuple[str, int, int]
_Key = tuple[_Payload]


def _seed(stream: str, replicate: int) -> int:
    return stream_seed(STUDY, "property_sample", stream, replicate)


def _draw(key: _Key) -> tuple[pd.DataFrame, dict[str, float]]:
    """The sample a key fits, a function of the key alone."""
    stream, replicate, _ = key[0]
    return law.draw(stream, N, _seed(stream, replicate)), law.truth(stream)


def cells_of(key: _Key) -> list[str]:
    """The ``family/cell`` names one key's rows fill."""
    return [cell.key for cell in CELLS if cell.stream == key[0][0]]


def cell_of(key: _Key) -> str:
    return f"stream/{key[0][0]}"


def replicate_of(key: _Key) -> int:
    return key[0][1]


def _fit_key(key: _Key, drawn: tuple[pd.DataFrame, dict[str, float]]) -> list[dict[str, Any]]:
    stream, replicate, requested = key[0]
    frame, truth = drawn
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return stream_rows(
            stream,
            frame,
            truth,
            replicate=replicate,
            requested=requested,
            alpha=STUDY.margins.alpha,
        )


def payloads(budget: int | None = None, start: int = 0) -> list[_Key]:
    """Every replication key; ``budget`` caps each stream for a smoke run."""
    requested = REPLICATES if budget is None else min(budget, REPLICATES)
    return [
        ((stream, replicate, requested),)
        for stream in law.STREAMS
        for replicate in range(start, start + requested)
    ]


class DeclaredLaw:
    """The law a declared cell reads: a stream, with the truth its rows carry.

    Parameters
    ----------
    stream : str
        The stream.
    truth : Mapping
        The truths of the cell's rows: the law's, or 0 for a paired cell.
    """

    def __init__(self, stream: str, truth: Mapping[str, float]) -> None:
        self.stream = stream
        self.name = f"{study.SLUG}__{stream}"
        self._truth = dict(truth)

    def truth(self) -> dict[str, float]:
        return dict(self._truth)


def declared_cells() -> tuple[PropertyCell, ...]:
    """Every sampled cell, with its role, size and replicate count."""
    return tuple(
        PropertyCell(
            property=cell.family,
            cell=cell.name,
            dgp=DeclaredLaw(
                cell.stream,
                {TARGET: 0.0} if cell.reading == "paired" else law.truth(cell.stream),
            ),
            outcome_learner=lambda: None,
            treatment_learner=lambda: None,
            n=N,
            replicates=REPLICATES,
            seed=_seed(cell.stream, 0),
            role=cell.role,
            estimand=TARGET,
            fit_kwargs={"fit": cell.fit, "reading": cell.reading},
        )
        for cell in CELLS
    )


def check_cells(rows: pd.DataFrame) -> None:
    """Refuse rows whose cells differ from the declaration."""
    found = set(zip(rows["property"], rows["cell"], strict=True))
    expected = {(family, cell) for family, names in PROPERTY_CELLS.items() for cell in names}
    if found != expected:
        raise RuntimeError(f"cells {sorted(found ^ expected)} differ from the declaration")


def generate_property_rows(*, n_jobs: int = STUDY_JOBS, budget: int | None = None) -> pd.DataFrame:
    """Fit every property replication; ``budget`` caps each stream for a smoke run."""
    outcomes = fit_properties(
        payloads(budget),
        _fit_key,
        cells_of=cells_of,
        cell_of=cell_of,
        replicate_of=replicate_of,
        n_jobs=n_jobs,
        draw=_draw,
        seed_of=lambda key: _seed(key[0][0], key[0][1]),
    )
    rows = pd.DataFrame([row for result in outcomes for row in result])
    rows = rows.loc[:, list(REPLICATE_COLUMNS)].sort_values(
        ["property", "cell", "replicate"], ignore_index=True
    )
    check_cells(rows)
    return rows


# ---------------------------------------------------------------------------- verdicts


def coverage_limit_verdicts(summary: pd.DataFrame, *, family: str, floor: float) -> None:
    """Read a family's exact ``covered`` intervals: a positive passes at a lower limit >= floor.

    The rows' ``covered`` column holds the event the family declares (the reading), so the
    interval is that share's exact Clopper-Pearson interval.

    Parameters
    ----------
    summary : pandas.DataFrame
        The cell summary; this writes ``passed`` on the family's rows.
    family : str
        The property family.
    floor : float
        The least lower limit of a positive row.

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


def coverage_limit_rule(family: str, *, floor: float) -> pf.FamilyRule:
    """The rule of :func:`coverage_limit_verdicts`: S2 on the lower limit of each positive."""
    return pf.FamilyRule(
        families=frozenset({family}),
        shared="coverage_limit_verdicts",
        clauses=(
            pf.proportion_clause(
                "share floor",
                statistic="S2",
                judge=pf.floor_judge(lambda ctx: floor),
                applies=pf.roles("positive"),
            ),
        ),
    )


def _cell_rows(rows: pd.DataFrame, family: str, cell: str) -> pd.DataFrame:
    return rows.loc[(rows["property"] == family) & (rows["cell"] == cell)]


def _locate(summary: pd.DataFrame, family: str, cell: str) -> Any:
    mask = (summary["property"] == family) & (summary["cell"] == cell)
    if int(mask.sum()) != 1:
        raise ValueError(f"{family}/{cell} is not one summary row")
    return summary.index[mask][0]


def paired_columns(summary: pd.DataFrame, rows: pd.DataFrame) -> None:
    """Publish each paired cell's mean difference in SDs of its whole-path estimates.

    Beside it, ungated, the share of replicates whose difference is exactly zero: the patient fit
    reported the whole path's estimate.
    """
    for key, (family, cell) in WHOLE_CELLS.items():
        paired_family, paired_cell = key.split("/", 1)
        whole_sd = float(_cell_rows(rows, family, cell)["estimate"].std(ddof=1))
        index = _locate(summary, paired_family, paired_cell)
        difference = _cell_rows(rows, paired_family, paired_cell)["estimate"]
        summary.loc[index, "paired_over_whole_sd"] = float(difference.mean()) / whole_sd
        summary.loc[index, "identical_share"] = float((difference == 0.0).mean())


def paired_verdicts(summary: pd.DataFrame, rows: pd.DataFrame) -> None:
    """Read the paired positive against :data:`PAIRED_MARGIN` SDs of its whole-path estimates.

    The margin replaces the row's ``bias_margin``, so the failure accounting's bias clause
    reads the same threshold (:func:`paired_rule`).
    """
    key = f"{PAIRED_FAMILY}/lc3_greedy"
    family, cell = WHOLE_CELLS[key]
    margin = PAIRED_MARGIN * float(_cell_rows(rows, family, cell)["estimate"].std(ddof=1))
    index = _locate(summary, PAIRED_FAMILY, "lc3_greedy")
    interval = summary_interval(summary, index, "bias")
    summary.loc[index, "bias_margin"] = margin
    inside = bool(interval.within(-margin, margin))
    summary.loc[index, "bias_equivalent"] = inside
    summary.loc[index, "passed"] = inside


def paired_rule() -> pf.FamilyRule:
    """The rule of :func:`paired_verdicts`: S1 against the published whole-path margin."""
    return pf.FamilyRule(
        families=frozenset({PAIRED_FAMILY}),
        shared="paired_verdicts",
        clauses=(
            pf.bias_clause(
                "paired difference equivalent", kind="equivalent", applies=pf.roles("positive")
            ),
        ),
    )


def conditional_columns(summary: pd.DataFrame, rows: pd.DataFrame) -> None:
    """Publish the LC3 ordered patient fit's bias on the draws whose selection excludes ``W1``."""
    estimates = _cell_rows(rows, "ordered_late_confounder", "lc3_ordered").set_index("replicate")
    excluded = _cell_rows(rows, "late_confounder_stop", "lc3_ordered").set_index("replicate")
    shared = estimates.index.intersection(excluded.index)
    chosen = shared[excluded.loc[shared, "covered"].to_numpy(dtype=bool)]
    whole_sd = float(_cell_rows(rows, *WHOLE_PATH_CELL)["estimate"].std(ddof=1))
    index = _locate(summary, "late_confounder_stop", "lc3_ordered")
    if len(chosen):
        bias = float((estimates.loc[chosen, "estimate"] - estimates.loc[chosen, "truth"]).mean())
        summary.loc[index, "conditional_bias"] = bias
        summary.loc[index, "conditional_bias_over_sd"] = bias / whole_sd


def reported_verdicts(summary: pd.DataFrame) -> None:
    """Publish the reported families' rows, which state no verdict."""
    mask = summary["property"].isin(REPORTED_FAMILIES)
    roles = set(summary.loc[mask, "role"].astype(str))
    if roles - {DIAGNOSTIC_ROLE}:
        raise ValueError(f"a reported family publishes roles {sorted(roles)}")
    summary.loc[mask, "passed"] = True
    summary.loc[mask, "property_passed"] = True


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The bias, share and paired verdicts; every reported family published."""
    summary, rates = apply_shared_verdicts(rows, STUDY, extra_columns=EXTRA_COLUMNS, rate_labels=())
    bias_role_verdicts(summary, families=BIAS_FAMILIES)
    for family in SHARE_FAMILIES:
        coverage_limit_verdicts(summary, family=family, floor=study.WITNESS_SHARE)
    paired_columns(summary, rows)
    paired_verdicts(summary, rows)
    conditional_columns(summary, rows)
    reported_verdicts(summary)
    return finish(summary, rates)


#: The failure accounting's declared rules.
PROPERTY_RULES = (
    bias_role_rule(BIAS_FAMILIES),
    *(coverage_limit_rule(family, floor=study.WITNESS_SHARE) for family in SHARE_FAMILIES),
    paired_rule(),
)


def smoke_keys(draws: int, start: int = 0) -> list[_Key]:
    """Keys of a smoke on replicate ids ``start`` to ``start + draws - 1`` of every stream."""
    return payloads(draws, start)


def fit_keys(keys: Sequence[_Key]) -> list[list[dict[str, Any]]]:
    """Fit keys in this process, for a smoke outside the driver."""
    return [_fit_key(key, _draw(key)) for key in keys]


def gated_cells() -> tuple[str, ...]:
    """The ``family/cell`` names whose red verdict gates the study."""
    return tuple(cell.key for cell in CELLS if cell.role != DIAGNOSTIC_ROLE)


__all__ = [
    "CELLS",
    "FAILURE_BOUND",
    "PAIRED_MARGIN",
    "PROPERTY_RULES",
    "PROPERTY_WITNESS",
    "declared_cells",
    "gated_cells",
    "generate_property_rows",
    "summarize_properties",
]
