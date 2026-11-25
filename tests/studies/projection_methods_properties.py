"""The property families of the ``projection-methods`` study, and their verdicts.

Every group of :data:`tests.studies.projection_methods.GROUPS` is one fit per replication, and its
rows fill the cells :func:`~tests.studies.projection_methods.group_cells` names:

- ``interval_calibration``: each ``dr_known_4k`` coefficient in its own arm
  (``__dr_known_intercept``, ``__dr_known_slope``), and the ``shrunken_se`` control on the
  intercept, the reported SE times 0.70, which must fall below the band;
- ``bias``: the coefficients of ``dr_known_4k``, ``dr_both_4k`` and ``ctmle_points_4k``, each
  inside +-0.25 of its own SD, and the ``wrong_q_witness`` rows, the plug-in at the wrong
  outcome regression, each wholly outside it;
- ``one_wrong_inference``: the shipped rule of
  :func:`~tests.studies.evidence.property_verdicts.one_wrong_inference_verdicts`.  The
  ``dr_glm_4k`` intercept reads ``conservative`` and its slope ``valid``; each rung of the slow
  ladder reads ``first_order`` on both terms with the declared negative sign, and the two
  ``growth`` cells read the smallest and largest rungs;
- ``spread``: the SE of ``D*`` alone beside the spread on ``dr_known_4k``, and the plug-in SE of
  ``dr_both_4k`` and ``ctmle_points_4k``, each published with the diagnostic role and no
  verdict.

A withheld group's rows read ``plugin_std_error`` and ``plugin_interval``
(:func:`~tests.studies.evidence.schema.reported_inference`), so its SE ratio and coverage
columns are the plug-in ones the one-wrong rule reads (plan 5.2).

Every draw names its stream (:data:`STREAMS`): the declared run reads ``property``, the pilot
``pilot`` and its extension ``pilot_2``, and a smoke ``smoke``.  In the declared run a group whose
:attr:`~tests.studies.projection_methods.Group.stream` names another stream draws there instead
(rev 3 R3.6: ``dr_glm_4k`` and the ladder on ``property_r3``).  :func:`sample_seed` and
:func:`fold_seed` both route through :func:`stream_labels`, and so do the draw, the fit and the
failure accounting's ``seed_of``, so a lost replicate records the seed it was drawn with.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies import projection_methods as study
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.properties import REPLICATE_COLUMNS, control_row, replicate_row
from tests.studies.evidence.property_verdicts import (
    DIAGNOSTIC_ROLE,
    ONE_WRONG_COLUMNS,
    apply_shared_verdicts,
    bias_role_rule,
    bias_role_verdicts,
    calibration_verdicts,
    finish,
    one_wrong_inference_verdicts,
    one_wrong_rule,
)
from tests.studies.evidence.replication import fit_properties, splat_draw, splat_drawn
from tests.studies.evidence.seeds import stream_seed

STUDY = study.STUDY

#: The factor of the shrunken-SE control.
SHRUNKEN_SE_FACTOR = study.SHRUNKEN_FACTOR
PROPERTY_CELLS = study.PROPERTY_CELLS

#: The share of a cell's requested replications that the accounting lets a run lose.
FAILURE_BOUND = 0.01

#: The draw-level statistic the accounting reports for each lost replicate.
PROPERTY_WITNESS = property_verdicts.SAMPLE_WITNESS

#: The families whose rows each answer to their own standardized-bias interval.
BIAS_FAMILIES = (study.BIAS,)

#: The sample and fold labels of each stream.  ``property`` is the declared run, and
#: ``property_r3`` the rev 3 stream of the re-declared groups.  The three probe streams of
#: :mod:`tests.studies.projection_methods_limit` are named here for the uniqueness test; no group
#: reads them.
STREAMS: dict[str, tuple[str, str]] = {
    "property": ("property_sample", "fold_partition"),
    "property_r3": ("property_r3_sample", "property_r3_fold_partition"),
    "pilot": ("pilot_sample", "pilot_fold_partition"),
    "pilot_2": ("pilot_2_sample", "pilot_2_fold_partition"),
    "smoke": ("smoke_sample", "smoke_fold_partition"),
    "glm4k_probe": ("glm4k_probe_sample", "glm4k_probe_fold_partition"),
    "ladder_probe": ("ladder_probe_sample", "ladder_probe_fold_partition"),
    "limit_probe": ("limit_probe_sample", "limit_probe_fold_partition"),
}

#: The positive calibration kinds, one per gated arm (the cell suffix after ``__``).
POSITIVE_ARMS: tuple[str, ...] = tuple(arm.removeprefix("__") for arm in study.CALIBRATION_ARMS)


def stream_labels(label: str, stream: str) -> tuple[str, str]:
    """The sample and fold labels ``label`` draws on ``stream``.

    On ``property`` a declared group reads its own :attr:`~tests.studies.projection_methods.
    Group.stream`; every other stream is read as named, for every group.
    """
    group = study.GROUP_BY_LABEL.get(label)
    if stream == study.PROPERTY_STREAM and group is not None:
        return STREAMS[group.stream]
    return STREAMS[stream]


def sample_seed(label: str, replicate: int, stream: str = "property") -> int:
    """The sample seed of replication ``replicate`` of a group on ``stream``."""
    return stream_seed(STUDY, stream_labels(label, stream)[0], label, replicate)


def fold_seed(label: str, replicate: int, stream: str = "property") -> int:
    """The fold seed of replication ``replicate`` of a group on ``stream``."""
    return stream_seed(STUDY, stream_labels(label, stream)[1], label, replicate)


def _draw(label: str, replicate: int, stream: str) -> Any:
    group = study.GROUP_BY_LABEL[label]
    return study.draw_from_seed(label, group.n, sample_seed(label, replicate, stream))


def _estimate_row(
    family: str, cell: str, role: str, group: study.Group, replicate: int, truth: float, value: Any
) -> dict[str, Any]:
    return replicate_row(
        property_name=family,
        cell=cell,
        role=role,
        replicate=replicate,
        n=group.n,
        requested=group.replicates,
        truth=truth,
        estimate=value,
        alpha=STUDY.margins.alpha,
    )


def _rows(
    group: study.Group, replicate: int, truths: dict[str, float], fitted: dict[str, Any]
) -> list[dict[str, Any]]:
    """Every property row of one fit of ``group``."""
    estimates = fitted["estimates"]
    rows = []
    for term in group.terms:
        truth = float(truths[group.name(term)])
        estimate = estimates[term]
        if term in group.arms:
            cell = study.calibration_cell(group, term)
            rows.append(
                _estimate_row(
                    study.CALIBRATION, cell, "positive", group, replicate, truth, estimate
                )
            )
        if group.bias_rows:
            cell = study.bias_cell(group, term)
            rows.append(
                _estimate_row(study.BIAS, cell, "positive", group, replicate, truth, estimate)
            )
        if term in group.one_wrong:
            cell = study.one_wrong_cell(group, term)
            kind = group.one_wrong[term]
            role = "positive" if kind in ("conservative", "valid") else "control"
            rows.append(
                _estimate_row(study.ONE_WRONG, cell, role, group, replicate, truth, estimate)
            )
        for control, terms in group.controls.items():
            if term not in terms:
                continue
            value = float(fitted["controls"][control][term])
            cell = study.control_cell(control, group, term)
            calibration = control in study.CALIBRATION_CONTROL_KINDS
            rows.append(
                control_row(
                    property_name=study.CALIBRATION if calibration else study.BIAS,
                    cell=cell,
                    replicate=replicate,
                    n=group.n,
                    requested=group.replicates,
                    truth=truth,
                    estimate=float(estimate.psi) if calibration else value,
                    standard_error=value if calibration else 1.0,
                    critical=study.CRITICAL,
                )
            )
        for reading, terms in group.readings.items():
            if term not in terms:
                continue
            row = control_row(
                property_name=study.SPREAD,
                cell=study.reading_cell(reading, group, term),
                replicate=replicate,
                n=group.n,
                requested=group.replicates,
                truth=truth,
                estimate=float(estimate.psi),
                standard_error=float(fitted["readings"][reading][term]),
                critical=study.CRITICAL,
                role=DIAGNOSTIC_ROLE,
            )
            rows.append(row)
    return rows


def _replicate(label: str, replicate: int, stream: str, drawn: Any) -> list[dict[str, Any]]:
    """Every property row of one replication of one group."""
    group = study.GROUP_BY_LABEL[label]
    frame, truths = drawn
    fitted = study.fit_group(group, frame, random_state=fold_seed(label, replicate, stream))
    return _rows(group, replicate, truths, fitted)


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
    """Every ``(group, replicate, stream)`` key, the largest fits first; ``budget`` caps each."""
    ordered = sorted(study.GROUPS, key=lambda group: -group.n)
    return [
        (group.label, index, stream)
        for group in ordered
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

    A run of every group must fill exactly the declared cells (the growth cells read the rungs'
    rows and have none); a run over ``groups`` checks only that each row it wrote is declared.
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
    declared = {
        (family, cell)
        for family, cells in PROPERTY_CELLS.items()
        for cell in cells
        if cell not in study.GROWTH_CELLS
    }
    found = set(zip(rows["property"], rows["cell"], strict=True))
    if found - declared or (groups is None and found != declared):
        raise RuntimeError(f"cells {sorted(found ^ declared)} differ from the declaration")
    return rows.loc[:, list(REPLICATE_COLUMNS)]


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The shared verdicts: calibration kinds, bias roles, the one-wrong rule, the readings."""
    shared = rows.loc[:, list(REPLICATE_COLUMNS)]
    summary, rates = apply_shared_verdicts(
        shared, STUDY, extra_columns=ONE_WRONG_COLUMNS, rate_labels=()
    )
    calibration_verdicts(
        summary, margins=STUDY.margins, efficiency_band=None, positive_suffix=POSITIVE_ARMS
    )
    bias_role_verdicts(summary, families=BIAS_FAMILIES)
    rates.extend(
        one_wrong_inference_verdicts(
            summary,
            shared,
            STUDY,
            predicted_signs=study.PREDICTED_SIGNS,
            ladders=study.LADDERS,
        )
    )
    summary.loc[summary["property"] == study.SPREAD, "passed"] = True
    return finish(summary, rates)


#: The failure accounting's declared rules.  The calibration family reads the shared rule
#: (:data:`~tests.studies.evidence.property_verdicts.FAMILY_RULES`).
PROPERTY_RULES = (
    bias_role_rule(BIAS_FAMILIES),
    one_wrong_rule(predicted_signs=study.PREDICTED_SIGNS, ladders=study.LADDERS),
)

__all__ = [
    "FAILURE_BOUND",
    "PROPERTY_RULES",
    "PROPERTY_WITNESS",
    "generate_property_rows",
    "summarize_properties",
]
