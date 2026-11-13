"""The cell grid shared by the two declared stopping-rule studies (X76 part (a)).

``option-stopping-point`` and ``option-stopping-ltmle`` read every cell off a few fits of one
draw.  A *stream* is one law: its draw, the fits made on each of its draws, and the rule-met
check of a fit.  A :class:`Cell` names the stream and fit it reads, the parameter, and its
*reading*:

============  ================================================================================
reading       the row's ``covered`` column
============  ================================================================================
``widened``   whether the reported interval (``ci``, widened by the remaining score under a
              declared rule) covers the truth
``plain``     whether the plain Wald interval of the same estimate (``rule_widened=False``)
              covers the truth
``witness``   whether the remaining score is at least :data:`WITNESS_FLOOR` standard errors,
              so the family's exact interval is that share's
============  ================================================================================

Every row also carries the signed residual in standard errors (``residual``) and whether the
fit met its rule within ``max_iter`` (``rule_met``).  :func:`rule_columns` publishes their
distribution for every cell; it writes no verdict.  The property modules read the verdicts
through the shared functions of :mod:`tests.studies.evidence.property_verdicts`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from tests.studies.evidence.properties import REPLICATE_COLUMNS, PropertyCell, replicate_row
from tests.studies.evidence.property_verdicts import CALIBRATION_CONTROL_KINDS
from tests.studies.evidence.registry import StudyRecord
from tests.studies.evidence.seeds import stream_seed

#: The residual a draw must reach to count for a witness share, in standard errors (plan 5.1).
WITNESS_FLOOR = 0.01
#: The calibration positive arm's suffix.  The cells are not all correctly specified (the
#: interval is the widened one), so the shared default name would be a false claim.
POSITIVE_SUFFIX = "widened"
#: The extra row columns.
ROW_COLUMNS = ("residual", "rule_met")
#: The summary columns :func:`rule_columns` writes.
SUMMARY_COLUMNS = (
    "rule_met_share",
    "mean_abs_residual",
    "median_abs_residual",
    "q90_abs_residual",
    "max_abs_residual",
)
READINGS = ("widened", "plain", "witness")


@dataclass(frozen=True)
class Stream:
    """One law of a study: its draw, its fits and the rule-met check.

    Parameters
    ----------
    name : str
        The stream's name, which keys its seeds.
    draw : Callable
        ``draw(n, seed)`` returns a sample and its exact truths.
    fits : Mapping
        Each fit's name to ``fit(frame)``, which returns a fitted result.
    rule_met : Callable
        Whether a fitted result met its declared rule within ``max_iter``.
    """

    name: str
    draw: Callable[[int, int], tuple[pd.DataFrame, dict[str, float]]]
    fits: Mapping[str, Callable[[pd.DataFrame], Any]]
    rule_met: Callable[[Any], bool]


@dataclass(frozen=True)
class Cell:
    """One declared cell: what it reads and the role its verdict takes.

    Parameters
    ----------
    family : str
        The property family.
    name : str
        The cell name, ``label__suffix``.
    role : str
        ``positive``, ``control`` or ``diagnostic``.
    stream : str
        The stream whose draws the cell reads.
    fit : str
        The fit of the stream the cell reads.
    estimand : str
        The parameter.
    reading : str
        One of :data:`READINGS`.
    """

    family: str
    name: str
    role: str
    stream: str
    fit: str
    estimand: str
    reading: str

    def __post_init__(self) -> None:
        if self.reading not in READINGS:
            raise ValueError(f"{self.name}: reading {self.reading!r} is not one of {READINGS}")

    @property
    def label(self) -> str:
        """The cell's label, its name before ``__``."""
        return self.name.split("__", 1)[0]


def property_cells(cells: Sequence[Cell]) -> dict[str, tuple[str, ...]]:
    """Every family's cell names, each calibration positive followed by its derived controls."""
    out: dict[str, list[str]] = {}
    for cell in cells:
        names = out.setdefault(cell.family, [])
        names.append(cell.name)
        if cell.family == "interval_calibration" and cell.role == "positive":
            names += [f"{cell.label}__{kind}" for kind in CALIBRATION_CONTROL_KINDS]
    return {family: tuple(names) for family, names in out.items()}


def calibration_labels(cells: Sequence[Cell]) -> tuple[str, ...]:
    """The labels of the gated calibration positives."""
    return tuple(
        cell.label
        for cell in cells
        if cell.family == "interval_calibration" and cell.role == "positive"
    )


def row(
    cell: Cell,
    result: Any,
    truth: Mapping[str, float],
    *,
    rule_met: bool,
    replicate: int,
    n: int,
    requested: int,
    alpha: float,
) -> dict[str, Any]:
    """The replicate row of one cell on one fitted result."""
    estimate = result[cell.estimand]
    read = replace(estimate, rule_widened=False) if cell.reading == "plain" else estimate
    out = replicate_row(
        property_name=cell.family,
        cell=cell.name,
        role=cell.role,
        replicate=replicate,
        n=n,
        requested=requested,
        truth=float(truth[cell.estimand]),
        estimate=read,
        alpha=alpha,
    )
    residual = float(estimate.score) / float(estimate.std_error)
    if cell.reading == "witness":
        out["covered"] = int(abs(residual) >= WITNESS_FLOOR)
    out["residual"] = residual
    out["rule_met"] = int(rule_met)
    return out


def stream_rows(
    stream: Stream,
    cells: Sequence[Cell],
    frame: pd.DataFrame,
    truth: Mapping[str, float],
    *,
    replicate: int,
    n: int,
    requested: int,
    alpha: float,
) -> list[dict[str, Any]]:
    """Every row of one draw of a stream: each fit once, every cell of the stream off it."""
    mine = [cell for cell in cells if cell.stream == stream.name]
    results = {name: stream.fits[name](frame) for name in dict.fromkeys(c.fit for c in mine)}
    met = {name: bool(stream.rule_met(result)) for name, result in results.items()}
    return [
        row(
            cell,
            results[cell.fit],
            truth,
            rule_met=met[cell.fit],
            replicate=replicate,
            n=n,
            requested=requested,
            alpha=alpha,
        )
        for cell in mine
    ]


def stream_cells(cells: Sequence[Cell], stream: str) -> list[str]:
    """The ``family/cell`` names one stream's draw fills, derived controls included."""
    names: list[str] = []
    for cell in cells:
        if cell.stream != stream:
            continue
        names.append(f"{cell.family}/{cell.name}")
        if cell.family == "interval_calibration" and cell.role == "positive":
            names += [f"interval_calibration/{cell.label}__{k}" for k in CALIBRATION_CONTROL_KINDS]
    return names


def published(rows: pd.DataFrame) -> pd.DataFrame:
    """The rows in their published columns and order."""
    columns = [*REPLICATE_COLUMNS, *ROW_COLUMNS]
    return rows.reindex(columns=columns).sort_values(
        ["property", "cell", "replicate"], ignore_index=True
    )


def check_cells(rows: pd.DataFrame, declared: Mapping[str, Sequence[str]]) -> None:
    """Refuse rows whose cells differ from the declaration."""
    found = set(zip(rows["property"], rows["cell"], strict=True))
    expected = {(family, cell) for family, names in declared.items() for cell in names}
    if found != expected:
        raise RuntimeError(f"cells {sorted(found ^ expected)} differ from the declaration")


def rule_columns(summary: pd.DataFrame, rows: pd.DataFrame) -> None:
    """Publish each cell's rule-met share and the distribution of its absolute residual.

    A derived control copies its positive arm's rows, so it reads that arm's numbers.  The
    columns are reported; no verdict reads them.
    """
    groups = rows.groupby(["property", "cell"], sort=False)
    for index in summary.index:
        key = (summary.loc[index, "property"], summary.loc[index, "cell"])
        if key not in groups.groups:
            continue
        group = groups.get_group(key)
        residual = group["residual"].abs().to_numpy(dtype=float)
        summary.loc[index, "rule_met_share"] = float(group["rule_met"].mean())
        summary.loc[index, "mean_abs_residual"] = float(residual.mean())
        summary.loc[index, "median_abs_residual"] = float(np.median(residual))
        summary.loc[index, "q90_abs_residual"] = float(np.quantile(residual, 0.9))
        summary.loc[index, "max_abs_residual"] = float(residual.max())


def synthetic_columns(cell: Any, rng: np.random.Generator) -> dict[str, Any]:
    """The extra columns of a synthetic row of the property differential.

    ``tests/canonical/property_differential.py`` builds rows from the declared cells while a
    study has no committed run; these are a residual near the one-step sizes and a met rule.
    """
    return {"residual": float(rng.normal(0.0, 0.05)), "rule_met": 1}


class DeclaredLaw:
    """The law a declared cell reads: a stream, with its exact truths.

    Parameters
    ----------
    study : str
        The study slug.
    stream : str
        The stream.
    truth : Mapping
        The stream's exact truths.
    """

    def __init__(self, study: str, stream: str, truth: Mapping[str, float]) -> None:
        self.stream = stream
        self.name = f"{study}__{stream}"
        self._truth = dict(truth)

    def truth(self) -> dict[str, float]:
        return dict(self._truth)


def declared_cells(
    record: StudyRecord,
    cells: Sequence[Cell],
    truths: Mapping[str, Mapping[str, float]],
    *,
    n: int,
    replicates: int,
) -> tuple[PropertyCell, ...]:
    """Every sampled cell, derived controls included, with its role, size and count."""
    out: list[PropertyCell] = []
    for cell in cells:
        names = [(cell.name, cell.role)]
        if cell.family == "interval_calibration" and cell.role == "positive":
            names += [(f"{cell.label}__{kind}", "control") for kind in CALIBRATION_CONTROL_KINDS]
        for name, role in names:
            out.append(
                PropertyCell(
                    property=cell.family,
                    cell=name,
                    dgp=DeclaredLaw(record.slug, cell.stream, truths[cell.stream]),
                    outcome_learner=lambda: None,
                    treatment_learner=lambda: None,
                    n=n,
                    replicates=replicates,
                    seed=stream_seed(record, "property_sample", cell.stream, 0),
                    role=role,
                    estimand=cell.estimand,
                )
            )
    return tuple(out)


# ---------------------------------------------------------------------------- design numbers


def floor_pass_probability(margins: Any, replicates: int, rate: float, floor: float) -> float:
    """P(the exact interval's lower limit is at least ``floor``) at a true rate ``rate``.

    The two-sided Clopper-Pearson interval at the study's confidence level; its lower limit
    rises with the count, so the gate passes from one count upward.
    """
    from scipy.stats import beta, binom

    tail = (1.0 - margins.confidence_level) / 2.0
    counts = np.arange(1, replicates + 1)
    passing = counts[beta.ppf(tail, counts, replicates - counts + 1) >= floor]
    if passing.size == 0:
        return 0.0
    return float(binom.sf(int(passing[0]) - 1, replicates, rate))


def ceiling_pass_probability(margins: Any, replicates: int, rate: float, ceiling: float) -> float:
    """P(the exact interval's upper limit is strictly below ``ceiling``) at a true rate ``rate``."""
    from scipy.stats import beta, binom

    tail = (1.0 - margins.confidence_level) / 2.0
    counts = np.arange(0, replicates)
    upper = beta.ppf(1.0 - tail, counts + 1, replicates - counts)
    passing = counts[upper < ceiling]
    if passing.size == 0:
        return 0.0
    return float(binom.cdf(int(passing[-1]), replicates, rate))
