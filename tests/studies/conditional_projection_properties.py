"""The property families of the ``conditional-projection`` study, and their verdicts.

Every group of :data:`tests.studies.conditional_projection.GROUPS` is one fit per replication,
and its rows fill the cells :func:`~tests.studies.conditional_projection.group_cells` names:

- ``interval_calibration``: each coefficient's ``__correctly_specified`` row (the shipped
  99% SE-ratio and coverage bands), and on ``clustered_4k`` the ``__row_variance_control``
  rows, the same estimates with the row-level SE, which must fall below the band;
- ``bias``: the ladder rungs' and the declared-error groups' TMLE rows, each inside +-0.25 of
  its own SD, and every control row, each wholly outside it
  (:func:`~tests.studies.evidence.property_verdicts.bias_role_verdicts`);
- ``type_i_error`` and ``power``: the df-2 heterogeneity test on ``hetero_trial``, its
  rejection read under the null and at slope 0.10.

A bias control is a number, not a fit with an interval, so its row carries a unit standard
error and only its bias is read.  The heterogeneity rows carry the statistic as the estimate
and the test's decision as the rejection; their coverage is the acceptance, which the test's
duality makes the coverage of ``0`` by the slopes' joint region.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies import conditional_projection as study
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.properties import REPLICATE_COLUMNS, control_row, replicate_row
from tests.studies.evidence.property_verdicts import (
    apply_shared_verdicts,
    bias_role_rule,
    bias_role_verdicts,
    calibration_verdicts,
    finish,
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


def _calibration_rows(
    group: study.Group, replicate: int, truths: dict[str, float], fitted: dict[str, Any]
) -> list[dict[str, Any]]:
    rows = []
    alpha = STUDY.margins.alpha
    if group.law == "assess":
        named = [
            (name, f"{group.label}_{study.ASSESS_LABELS[name]}") for name in study.assess_names()
        ]
    else:
        named = [
            (f"{study.stem(group)}[{term}]", f"{group.label}_{study.TERM_LABELS[term]}")
            for term in study.TERMS
        ]
    for name, prefix in named:
        rows.append(
            replicate_row(
                property_name=study.CALIBRATION,
                cell=f"{prefix}__{study.POSITIVE}",
                role="positive",
                replicate=replicate,
                n=group.n,
                requested=group.replicates,
                truth=float(truths[name]),
                estimate=fitted["estimates"][name],
                alpha=alpha,
            )
        )
    return rows


def _replicate(label: str, replicate: int, drawn: Any) -> list[dict[str, Any]]:
    """Every property row of one replication of one group."""
    group = study.GROUP_BY_LABEL[label]
    frame, truths = drawn
    fitted = study.fit_group(group, frame, random_state=replicate)
    rows: list[dict[str, Any]] = []
    if group.calibration or group.law == "assess":
        rows += _calibration_rows(group, replicate, truths, fitted)
    for term in group.bias_rows:
        name = f"{study.stem(group)}[{term}]"
        rows.append(
            replicate_row(
                property_name=study.BIAS,
                cell=study.bias_cell(group.label, term),
                role="positive",
                replicate=replicate,
                n=group.n,
                requested=group.replicates,
                truth=float(truths[name]),
                estimate=fitted["estimates"][name],
                alpha=STUDY.margins.alpha,
            )
        )
    for control, terms in group.controls.items():
        values = fitted["controls"][control]
        for term in terms:
            name = f"{study.stem(group)}[{term}]"
            if control == "iid_se":
                rows.append(
                    control_row(
                        property_name=study.CALIBRATION,
                        cell=f"{group.label}_{study.TERM_LABELS[term]}__{study.ROW_VARIANCE}",
                        replicate=replicate,
                        n=group.n,
                        requested=group.replicates,
                        truth=float(truths[name]),
                        estimate=float(fitted["estimates"][name].psi),
                        standard_error=float(values[term]),
                        critical=study.CRITICAL,
                    )
                )
                continue
            rows.append(
                control_row(
                    property_name=study.BIAS,
                    cell=study.bias_cell(study.control_label(control, group), term),
                    replicate=replicate,
                    n=group.n,
                    requested=group.replicates,
                    truth=float(truths[name]),
                    estimate=float(values[term]),
                    standard_error=1.0,
                    critical=study.CRITICAL,
                )
            )
    if group.law == "hetero_trial":
        family, cell = (
            (study.TYPE_I, "heterogeneity__sharp_null")
            if group.slope == 0.0
            else (study.POWER, "heterogeneity__slope_0.10")
        )
        rejected = int(fitted["p_value"] < STUDY.margins.alpha)
        rows.append(
            {
                "property": family,
                "cell": cell,
                "role": "positive",
                "replicate": replicate,
                "n": group.n,
                "requested_replicates": group.replicates,
                "failed_replicates": 0,
                "truth": 0.0,
                "estimate": float(fitted["statistic"]),
                "std_error": 1.0,
                "covered": 1 - rejected,
                "rejected": rejected,
            }
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


def payloads(budget: int | None = None) -> list[tuple[str, int]]:
    """Every ``(group, replicate)`` key, in group order; ``budget`` caps each group."""
    return [
        (group.label, index)
        for group in study.GROUPS
        for index in range(group.replicates if budget is None else min(budget, group.replicates))
    ]


def generate_property_rows(*, n_jobs: int = STUDY_JOBS, budget: int | None = None) -> pd.DataFrame:
    """Fit every property replication; ``budget`` caps each group for a pre-run check."""
    outcomes = fit_properties(
        payloads(budget),
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
    declared = {(family, cell) for family, cells in PROPERTY_CELLS.items() for cell in cells}
    found = set(zip(rows["property"], rows["cell"], strict=True))
    if found != declared:
        raise RuntimeError(f"cells {sorted(found ^ declared)} differ from the declaration")
    return rows.loc[:, list(REPLICATE_COLUMNS)]


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The shared verdicts, the calibration kinds and the bias rows."""
    summary, rates = apply_shared_verdicts(rows, STUDY, rate_labels=())
    calibration_verdicts(summary, margins=STUDY.margins, efficiency_band=None)
    bias_role_verdicts(summary, families=BIAS_FAMILIES)
    return finish(summary, rates)


#: The failure accounting's declared rules.  The calibration, type-I and power families read
#: the shared rules (:data:`~tests.studies.evidence.property_verdicts.FAMILY_RULES`).
PROPERTY_RULES = (bias_role_rule(BIAS_FAMILIES),)

__all__ = [
    "FAILURE_BOUND",
    "PROPERTY_RULES",
    "PROPERTY_WITNESS",
    "generate_property_rows",
    "summarize_properties",
]
