"""The property families of the ``projection-evidence`` study, and their verdicts.

Every group of :data:`tests.studies.projection_evidence.GROUPS` is one fit per replication (a CDE
group is one engine fit at both levels), and its rows fill the cells
:func:`~tests.studies.projection_evidence.group_cells` names:

- ``interval_calibration``: each coefficient's row in each of the group's arms, and the
  ``shrunken_se`` control (cell ``cde_4k_z0_intercept__shrunken_se_control``), the reported SE
  times 0.70, which must fall below the band;
- ``bias``: the ``cde_dr_4k`` TMLE rows at the declared wrong ``Qbar``, each inside +-0.25 of its
  own SD, and the ``plugin_stratum``, ``no_selection``, ``no_intermediate_weight``,
  ``plugin_wrong`` and ``total_effect`` control rows, each wholly outside it;
- ``simultaneous_coverage``: each gated band group's band over its grid, whose joint coverage
  must lie in the calibration band, and its pointwise joint control, which must read below
  0.95; the bands of ``cde_4k`` (at each level) and ``pooled_4k`` publish under the diagnostic
  role with no verdict (plan 5.2).

A joint row records the max-t statistic ``max_j |f_j - f_j^0| / se_j`` over the grid as its
estimate and the critical value its intervals used as its standard error, the convention of
:mod:`tests.studies.evidence.simultaneous`, and asserts ``covered == (statistic <= critical)``.

Every draw names its stream (:data:`STREAMS`): the declared run reads ``property``, the pilot
``pilot`` and its extension ``pilot_2``, and a smoke ``smoke``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies import projection_evidence as study
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.properties import REPLICATE_COLUMNS, control_row, replicate_row
from tests.studies.evidence.property_verdicts import (
    apply_shared_verdicts,
    bias_role_rule,
    bias_role_verdicts,
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

#: The sample and fold labels of each stream.  ``property`` is the declared run.
STREAMS: dict[str, tuple[str, str]] = {
    "property": ("property_sample", "fold_partition"),
    "pilot": ("pilot_sample", "pilot_fold_partition"),
    "pilot_2": ("pilot_2_sample", "pilot_2_fold_partition"),
    "smoke": ("smoke_sample", "smoke_fold_partition"),
}

#: The positive calibration kinds, one per gated arm (the cell suffix after ``__``).
POSITIVE_ARMS: tuple[str, ...] = tuple(arm.removeprefix("__") for arm in study.CALIBRATION_ARMS)

#: The role of a free band reading (plan 5.2).
DIAGNOSTIC_ROLE = property_verdicts.DIAGNOSTIC_ROLE

#: The tolerance inside which a joint row's statistic may disagree with its coverage indicator.
_TIE_TOLERANCE = 1e-9


def sample_seed(label: str, replicate: int, stream: str = "property") -> int:
    """The sample seed of replication ``replicate`` of a group on ``stream``."""
    return stream_seed(STUDY, STREAMS[stream][0], label, replicate)


def fold_seed(label: str, replicate: int, stream: str = "property") -> int:
    """The fold seed of replication ``replicate`` of a group on ``stream``."""
    return stream_seed(STUDY, STREAMS[stream][1], label, replicate)


def _draw(label: str, replicate: int, stream: str) -> Any:
    group = study.GROUP_BY_LABEL[label]
    return study.draw_from_seed(label, group.n, sample_seed(label, replicate, stream))


def _calibration_rows(
    group: study.Group, replicate: int, truths: dict[str, float], fitted: dict[str, Any]
) -> list[dict[str, Any]]:
    rows = []
    for arm in group.arms:
        level = study.arm_level(arm)
        for term in group.terms:
            name = group.name(term, level)
            rows.append(
                replicate_row(
                    property_name=study.CALIBRATION,
                    cell=study.calibration_cell(group, arm, term),
                    role="positive",
                    replicate=replicate,
                    n=group.n,
                    requested=group.replicates,
                    truth=float(truths[name]),
                    estimate=fitted["estimates"][name],
                    alpha=STUDY.margins.alpha,
                )
            )
    return rows


def _bias_rows(
    group: study.Group, replicate: int, truths: dict[str, float], fitted: dict[str, Any]
) -> list[dict[str, Any]]:
    if not group.bias_rows:
        return []
    rows = []
    for level in group.levels:
        for term in group.terms:
            name = group.name(term, level)
            rows.append(
                replicate_row(
                    property_name=study.BIAS,
                    cell=study.bias_cell(group, term, level),
                    role="positive",
                    replicate=replicate,
                    n=group.n,
                    requested=group.replicates,
                    truth=float(truths[name]),
                    estimate=fitted["estimates"][name],
                    alpha=STUDY.margins.alpha,
                )
            )
    return rows


def _control_rows(
    group: study.Group, replicate: int, truths: dict[str, float], fitted: dict[str, Any]
) -> list[dict[str, Any]]:
    rows = []
    for control, terms in group.controls.items():
        values = fitted["controls"][control]
        for level in study.control_levels(control, group):
            for term in terms:
                cell = study.control_cell(control, group, term, level)
                truth = study.control_truth(group, control, term, level, truths)
                if control in study.CALIBRATION_CONTROL_KINDS:
                    estimate = float(fitted["estimates"][group.name(term, level)].psi)
                    row = control_row(
                        property_name=study.CALIBRATION,
                        cell=cell,
                        replicate=replicate,
                        n=group.n,
                        requested=group.replicates,
                        truth=truth,
                        estimate=estimate,
                        standard_error=float(values[term, level]),
                        critical=study.CRITICAL,
                    )
                else:
                    row = control_row(
                        property_name=study.BIAS,
                        cell=cell,
                        replicate=replicate,
                        n=group.n,
                        requested=group.replicates,
                        truth=truth,
                        estimate=float(values[term, level]),
                        standard_error=1.0,
                        critical=study.CRITICAL,
                    )
                rows.append(row)
    return rows


def _joint_row(
    *,
    cell: str,
    role: str,
    replicate: int,
    group: study.Group,
    statistic: float,
    critical: float,
    covered: bool,
) -> dict[str, Any]:
    """One ``simultaneous_coverage`` row, after checking that ``covered`` is its statistic."""
    tied = abs(statistic - critical) <= _TIE_TOLERANCE * max(1.0, abs(critical))
    if not tied and covered != (statistic <= critical):
        raise AssertionError(f"{cell}: covered={covered} but statistic {statistic} vs {critical}")
    return {
        "property": study.SIMULTANEOUS,
        "cell": cell,
        "role": role,
        "replicate": replicate,
        "n": group.n,
        "requested_replicates": group.replicates,
        "failed_replicates": 0,
        "truth": 0.0,
        "estimate": float(statistic),
        "std_error": float(critical),
        "covered": int(covered),
        "rejected": int(not covered),
    }


def joint_readings(
    reading: dict[str, Any], truth: np.ndarray
) -> tuple[tuple[float, float, bool], tuple[float, float, bool]]:
    """``(statistic, critical, covered)`` of the band and of its pointwise joint control."""
    deviation = np.abs(reading["estimate"] - truth) / reading["std_error"]
    statistic = float(np.max(deviation))
    band_covered = bool(np.all((reading["lower"] <= truth) & (truth <= reading["upper"])))
    pointwise = bool(np.all(deviation <= study.CRITICAL))
    return (
        (statistic, reading["critical"], band_covered),
        (statistic, study.CRITICAL, pointwise),
    )


def _band_rows(
    group: study.Group, replicate: int, truths: dict[str, float], fitted: dict[str, Any]
) -> list[dict[str, Any]]:
    if not group.band:
        return []
    rows = []
    for level, label in zip(group.levels, study.band_labels(group), strict=True):
        band, control = joint_readings(
            fitted["bands"][label], study.band_truth(group, truths, level)
        )
        role = "positive" if group.band == "gated" else DIAGNOSTIC_ROLE
        rows.append(
            _joint_row(
                cell=f"{label}__{study.BAND_KIND}",
                role=role,
                replicate=replicate,
                group=group,
                statistic=band[0],
                critical=band[1],
                covered=band[2],
            )
        )
        if group.band == "gated":
            rows.append(
                _joint_row(
                    cell=f"{label}__{study.JOINT_CONTROL_KIND}",
                    role="control",
                    replicate=replicate,
                    group=group,
                    statistic=control[0],
                    critical=control[1],
                    covered=control[2],
                )
            )
    return rows


def _replicate(label: str, replicate: int, stream: str, drawn: Any) -> list[dict[str, Any]]:
    """Every property row of one replication of one group."""
    group = study.GROUP_BY_LABEL[label]
    frame, truths = drawn
    fitted = study.fit_group(group, frame, random_state=fold_seed(label, replicate, stream))
    return [
        *_calibration_rows(group, replicate, truths, fitted),
        *_bias_rows(group, replicate, truths, fitted),
        *_control_rows(group, replicate, truths, fitted),
        *_band_rows(group, replicate, truths, fitted),
    ]


def _cells_of(key: tuple[str, int, str]) -> list[str]:
    group = study.GROUP_BY_LABEL[key[0]]
    return [
        f"{family}/{cell}" for family, cells in study.group_cells(group).items() for cell in cells
    ]


def _cell_of(key: tuple[str, int, str]) -> str:
    return f"group/{key[0]}"


def _replicate_of(key: tuple[str, int, str]) -> int:
    return key[1]


def payloads(budget: int | None = None, stream: str = "property") -> list[tuple[str, int, str]]:
    """Every ``(group, replicate, stream)`` key, in group order; ``budget`` caps each group."""
    return [
        (group.label, index, stream)
        for group in study.GROUPS
        for index in range(group.replicates if budget is None else min(budget, group.replicates))
    ]


def generate_property_rows(
    *,
    n_jobs: int = STUDY_JOBS,
    budget: int | None = None,
    groups: tuple[str, ...] | None = None,
    stream: str = "property",
) -> pd.DataFrame:
    """Fit every property replication; ``budget`` caps each group and ``groups`` picks some.

    A run of every group must fill exactly the declared cells; a run over ``groups`` checks
    only that each row it wrote is declared.
    """
    keys = [key for key in payloads(budget, stream) if groups is None or key[0] in groups]
    outcomes = fit_properties(
        keys,
        splat_drawn(_replicate),
        draw=splat_draw(_draw),
        cells_of=_cells_of,
        cell_of=_cell_of,
        replicate_of=_replicate_of,
        n_jobs=n_jobs,
        seed_of=lambda key: sample_seed(*key),
    )
    rows = pd.DataFrame([row for records in outcomes for row in records])
    if budget is not None:
        rows["requested_replicates"] = rows["requested_replicates"].clip(upper=budget)
    declared = {(family, cell) for family, cells in PROPERTY_CELLS.items() for cell in cells}
    found = set(zip(rows["property"], rows["cell"], strict=True))
    if found - declared or (groups is None and found != declared):
        raise RuntimeError(f"cells {sorted(found ^ declared)} differ from the declaration")
    return rows.loc[:, list(REPLICATE_COLUMNS)]


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The shared verdicts: calibration kinds, bias roles and joint coverage."""
    summary, rates = apply_shared_verdicts(
        rows.loc[:, list(REPLICATE_COLUMNS)], STUDY, rate_labels=()
    )
    calibration_verdicts(
        summary, margins=STUDY.margins, efficiency_band=None, positive_suffix=POSITIVE_ARMS
    )
    bias_role_verdicts(summary, families=BIAS_FAMILIES)
    simultaneous_coverage_verdicts(summary, margins=STUDY.margins)
    return finish(summary, rates)


#: The failure accounting's declared rules.  The calibration and joint-coverage families read
#: the shared rules (:data:`~tests.studies.evidence.property_verdicts.FAMILY_RULES`).
PROPERTY_RULES = (bias_role_rule(BIAS_FAMILIES),)

__all__ = [
    "FAILURE_BOUND",
    "PROPERTY_RULES",
    "PROPERTY_WITNESS",
    "generate_property_rows",
    "summarize_properties",
]
