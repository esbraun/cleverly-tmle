"""The property families of the ``targeting-curves`` study, and their verdicts.

Every group of :data:`tests.studies.targeting_curves.GROUPS` is one fit per replication, and its
rows fill the cells :func:`~tests.studies.targeting_curves.group_cells` names:

- ``interval_calibration``: each reported quantity's ``__correctly_specified`` row, the
  curve's Wald interval ``estimate +- z SE`` read against the truth (on cell ``S`` the truth of
  that replicate's fitted curve), and each SE control's row, the same estimate with the
  mutated draws' SE, which must leave the SE-ratio band on its declared side;
- ``bias``: the two point controls (cell ``D``'s whole tied block, cell ``S``'s in-sample
  curve), each wholly outside +-0.25 of its own SD;
- ``simultaneous_coverage``: cell ``H``'s band covering every grid truth at once, and the
  pointwise intervals read jointly, which must fall below nominal;
- ``type_i_error`` and ``power``: whether the band excludes 0 at some grid point, on cells
  ``U`` and ``N`` (TOC identically 0) and on cell ``P``.

A point control is a number, not a fit with an interval, so its row carries a unit standard
error and only its bias is read, as in ``conditional_projection_properties``.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies import targeting_curves as study
from tests.studies.evidence import property_failures as pf
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.properties import REPLICATE_COLUMNS, control_row
from tests.studies.evidence.property_verdicts import (
    apply_shared_verdicts,
    calibration_verdicts,
    finish,
    simultaneous_coverage_verdicts,
)
from tests.studies.evidence.replication import fit_properties, splat_draw, splat_drawn
from tests.studies.evidence.seeds import stream_seed

STUDY = study.STUDY
PROPERTY_CELLS = study.PROPERTY_CELLS

#: The share of a cell's requested replications that the accounting lets a run lose.
FAILURE_BOUND = 0.01

#: The draw-level statistic the accounting reports for each lost replicate.
PROPERTY_WITNESS = property_verdicts.SAMPLE_WITNESS

#: The families whose rows each answer to their own standardized-bias interval.
BIAS_FAMILIES = (study.BIAS,)


def _draw(label: str, replicate: int) -> Any:
    """The draw of replication ``replicate`` of a group, from the stream keyed by its label."""
    group = study.GROUP_BY_LABEL[label]
    seed = stream_seed(STUDY, "property_sample", label, replicate)
    return study.draw_from_seed(label, group.n, seed)


def _row(
    group: study.Group,
    replicate: int,
    *,
    family: str,
    cell: str,
    truth: float,
    estimate: float,
    standard_error: float,
    role: str,
) -> dict[str, Any]:
    return control_row(
        property_name=family,
        cell=cell,
        replicate=replicate,
        n=group.n,
        requested=group.replicates,
        truth=float(truth),
        estimate=float(estimate),
        standard_error=float(standard_error),
        critical=study.CRITICAL,
        role=role,
    )


def _band_row(
    group: study.Group, replicate: int, *, family: str, cell: str, role: str, covered: float
) -> dict[str, Any]:
    """A joint-coverage or rejection row: ``covered`` is the band's event, ``rejected`` its
    complement for a test family."""
    rejected = 1 - int(covered) if family != study.JOINT else 0
    return {
        "property": family,
        "cell": cell,
        "role": role,
        "replicate": replicate,
        "n": group.n,
        "requested_replicates": group.replicates,
        "failed_replicates": 0,
        "truth": 0.0,
        "estimate": 0.0,
        "std_error": 1.0,
        "covered": int(covered),
        "rejected": rejected,
    }


def _replicate(label: str, replicate: int, drawn: Any) -> list[dict[str, Any]]:
    """Every property row of one replication of one group."""
    group = study.GROUP_BY_LABEL[label]
    frame, truths = drawn
    fitted = study.fit_group(group, frame, random_state=replicate)
    truths = {**truths, **fitted.get("truths", {})}
    estimates = fitted["rows"]
    rows: list[dict[str, Any]] = []
    for name in group.rows:
        rows.append(
            _row(
                group,
                replicate,
                family=study.CALIBRATION,
                cell=study.calibration_cell(group.label, name),
                truth=truths[name],
                estimate=estimates[name]["estimate"],
                standard_error=estimates[name]["std_error"],
                role="positive",
            )
        )
    for control, names in group.controls.items():
        if control == "pointwise_joint":
            continue
        values = fitted["controls"][control]
        for name in names:
            if control in ("whole_block", "in_sample"):
                truth = values.get(f"truth:{name}", truths.get(name))
                rows.append(
                    _row(
                        group,
                        replicate,
                        family=study.BIAS,
                        cell=f"{group.label}_{study._label(name)}__{control}",
                        truth=float(truth),
                        estimate=values[name],
                        standard_error=1.0,
                        role="control",
                    )
                )
                continue
            kind = study.se_control_kind(control)
            if kind is None:
                continue
            rows.append(
                _row(
                    group,
                    replicate,
                    family=study.CALIBRATION,
                    cell=study.calibration_cell(group.label, f"{control}_{name}", kind),
                    truth=truths[name],
                    estimate=estimates[name]["estimate"],
                    standard_error=values[name],
                    role="control",
                )
            )
    band = fitted.get("band")
    if group.band == "coverage":
        rows.append(
            _band_row(
                group,
                replicate,
                family=study.JOINT,
                cell=f"{group.label}_band__{study.POSITIVE}",
                role="positive",
                covered=band["covered"],
            )
        )
        if "pointwise_joint" in group.controls:
            rows.append(
                _band_row(
                    group,
                    replicate,
                    family=study.JOINT,
                    cell=f"{group.label}_band__pointwise_joint",
                    role="control",
                    covered=band["pointwise_covered"],
                )
            )
    elif group.band == "type_i":
        rows.append(
            _band_row(
                group,
                replicate,
                family=study.TYPE_I,
                cell=f"{group.label}_band__null",
                role="positive",
                covered=1.0 - band["rejected"],
            )
        )
    elif group.band == "power":
        rows.append(
            _band_row(
                group,
                replicate,
                family=study.POWER,
                cell=f"{group.label}_band__slope",
                role="positive",
                covered=1.0 - band["rejected"],
            )
        )
    return rows


def _cells_of(key: tuple[str, int]) -> list[str]:
    group = study.GROUP_BY_LABEL[key[0]]
    return [
        f"{family}/{cell}" for family, cells in study.group_cells(group).items() for cell in cells
    ]


def _cell_of(key: tuple[str, int]) -> str:
    return f"group/{key[0]}"


def _replicate_of(key: tuple[str, int]) -> int:
    return key[1]


def payloads(
    budget: int | None = None, groups: tuple[str, ...] | None = None
) -> list[tuple[str, int]]:
    """Every ``(group, replicate)`` key, in group order; ``budget`` caps each group and
    ``groups`` keeps only those labels (a smoke run)."""
    return [
        (group.label, index)
        for group in study.GROUPS
        if groups is None or group.label in groups
        for index in range(group.replicates if budget is None else min(budget, group.replicates))
    ]


def generate_property_rows(
    *, n_jobs: int = STUDY_JOBS, budget: int | None = None, groups: tuple[str, ...] | None = None
) -> pd.DataFrame:
    """Fit every property replication; ``budget`` and ``groups`` restrict a pre-run check."""
    outcomes = fit_properties(
        payloads(budget, groups),
        splat_drawn(_replicate),
        draw=splat_draw(_draw),
        cells_of=_cells_of,
        cell_of=_cell_of,
        replicate_of=_replicate_of,
        n_jobs=n_jobs,
        seed_of=lambda key: stream_seed(STUDY, "property_sample", key[0], key[1]),
    )
    rows = pd.DataFrame([row for records in outcomes for row in records])
    if budget is not None:
        rows["requested_replicates"] = rows["requested_replicates"].clip(upper=budget)
    declared = {
        (family, cell)
        for group in study.GROUPS
        if groups is None or group.label in groups
        for family, cells in study.group_cells(group).items()
        for cell in cells
    }
    found = set(zip(rows["property"], rows["cell"], strict=True))
    if found != declared:
        raise RuntimeError(f"cells {sorted(found ^ declared)} differ from the declaration")
    return rows.loc[:, list(REPLICATE_COLUMNS)]


def bias_verdicts(summary: pd.DataFrame) -> None:
    """Every bias row here is a control, and passes on its discrimination endpoint."""
    for family in BIAS_FAMILIES:
        member = summary["property"] == family
        control = member & (summary["role"] == "control")
        summary.loc[control, "passed"] = summary.loc[control, "bias_discriminated"]
        if (member & ~control).any():
            raise ValueError(f"{family} cells of this study are controls only")


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The shared verdicts, the calibration kinds, the band's joint coverage and the controls."""
    summary, rates = apply_shared_verdicts(rows, STUDY, rate_labels=())
    calibration_verdicts(summary, margins=STUDY.margins, efficiency_band=None)
    simultaneous_coverage_verdicts(summary, margins=STUDY.margins)
    bias_verdicts(summary)
    return finish(summary, rates)


def _bias_rule() -> pf.FamilyRule:
    return pf.FamilyRule(
        families=frozenset(BIAS_FAMILIES),
        shared="bias_verdicts",
        clauses=(pf.bias_clause("bias discriminated", kind="beyond", applies=pf.roles("control")),),
    )


#: The failure accounting's declared rules.  The calibration, joint-coverage, type-I and power
#: families read the shared rules (:data:`~tests.studies.evidence.property_verdicts.FAMILY_RULES`).
PROPERTY_RULES = (_bias_rule(),)

__all__ = [
    "FAILURE_BOUND",
    "PROPERTY_RULES",
    "PROPERTY_WITNESS",
    "bias_verdicts",
    "generate_property_rows",
    "summarize_properties",
]
