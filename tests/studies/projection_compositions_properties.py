"""The property families of the ``projection-compositions`` study, and their verdicts.

Every group of :data:`tests.studies.projection_compositions.GROUPS` is one fit per
replication, and its rows fill the cells
:func:`~tests.studies.projection_compositions.group_cells` names:

- ``interval_calibration``: each reported name's row in each of the group's arms.  A Wald arm
  reads the reported interval; a bootstrap arm (``__boot_*``) reads the percentile interval and
  the bootstrap standard error the fit publishes, and its row carries the failed-replicate
  count.  The ``boot_shrunk`` control shrinks the percentile interval by 0.70 about the
  replicates' median and the ``se_over_root_r`` control divides the reported SE by ``sqrt(3)``;
  both must fall below the band.
- ``bias``: the ``mtreat_dr_4k`` TMLE rows, each inside +-0.25 of its own SD, and the
  ``complete_case``, ``no_treatment_observation`` and ``marginal_as_stratum`` control rows,
  each wholly outside it.
- ``type_i_error`` and ``power``: the df-2 heterogeneity test of a ``repeats=3`` fit on
  ``hetero_trial``, through the matrix rule.

Two readings are declared beside the verdicts (plan 5.0).  Every ``rep3_*`` row publishes the
SE ratio of ``se_without_displacement`` (the median rule with the displacement dropped); a row
red above the band whose ratio of that column lies in the band reads as the median rule's split
adjustment.  A bootstrap row whose mean failed-replicate share exceeds 1% reports
``bootstrap_conditional`` and claims no coverage.  ``boot_stacked_2k`` publishes without a
verdict: it is the evidence the cross-fitted bootstrap's experimental reason cites.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies import projection_compositions as study
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

#: The extra row columns: the bootstrap's requested and failed replicates, and a repeated fit's
#: ``se_without_displacement``.
EXTRA_COLUMNS = ("bootstrap_requested", "bootstrap_failed", "se_without_displacement")

#: The summary columns of the two declared readings.
SUMMARY_COLUMNS = (
    "bootstrap_failed_fraction",
    "bootstrap_conditional",
    "se_ratio_without_displacement",
)

#: The mean failed-replicate share above which a bootstrap row claims no coverage.
BOOTSTRAP_CONDITIONAL_SHARE = 0.01


def _draw(label: str, replicate: int) -> Any:
    """The draw of replication ``replicate`` of a group, from the stream keyed by its label."""
    group = study.GROUP_BY_LABEL[label]
    seed = stream_seed(STUDY, "property_sample", label, replicate)
    return study.draw_from_seed(label, group.n, seed)


def _extras(group: study.Group, fitted: dict[str, Any], name: str) -> dict[str, float]:
    without = fitted.get("se_without_displacement", {})
    return {
        "bootstrap_requested": float(study.N_BOOTSTRAP if group.bootstrap else 0),
        "bootstrap_failed": float(fitted["bootstrap_failed"]),
        "se_without_displacement": float(without.get(name, np.nan)),
    }


def _calibration_rows(
    group: study.Group, replicate: int, truths: dict[str, float], fitted: dict[str, Any]
) -> list[dict[str, Any]]:
    rows = []
    role = "diagnostic" if group.reporting else "positive"
    for arm in group.arms:
        for name in group.names():
            estimate = fitted["estimates"][name]
            truth = float(truths[name])
            cell = study.calibration_cell(group, arm, name)
            if arm.startswith("__boot"):
                summary = estimate.bootstrap
                low, high = summary.ci
                row = control_row(
                    property_name=study.CALIBRATION,
                    cell=cell,
                    replicate=replicate,
                    n=group.n,
                    requested=group.replicates,
                    truth=truth,
                    estimate=float(estimate.psi),
                    standard_error=float(summary.std_error),
                    critical=study.CRITICAL,
                    role=role,
                )
                row["covered"] = int(low <= truth <= high)
                row["rejected"] = int(not low <= 0.0 <= high)
            else:
                row = replicate_row(
                    property_name=study.CALIBRATION,
                    cell=cell,
                    role=role,
                    replicate=replicate,
                    n=group.n,
                    requested=group.replicates,
                    truth=truth,
                    estimate=estimate,
                    alpha=STUDY.margins.alpha,
                )
            rows.append({**row, **_extras(group, fitted, name)})
    return rows


def _control_rows(
    group: study.Group, replicate: int, truths: dict[str, float], fitted: dict[str, Any]
) -> list[dict[str, Any]]:
    rows = []
    for control, terms in group.controls.items():
        values = fitted["controls"][control]
        for term in terms:
            name = f"{group.stem}[{term}]"
            cell = f"{study.control_label(control, group)}_{study.term_label(term)}"
            truth = study.control_truth(group, control, term, truths)
            estimate = float(fitted["estimates"][name].psi)
            if control == "boot_shrunk":
                low, high, se = values[term]
                row = control_row(
                    property_name=study.CALIBRATION,
                    cell=cell,
                    replicate=replicate,
                    n=group.n,
                    requested=group.replicates,
                    truth=truth,
                    estimate=estimate,
                    standard_error=se,
                    critical=study.CRITICAL,
                )
                row["covered"] = int(low <= truth <= high)
                row["rejected"] = int(not low <= 0.0 <= high)
            elif control == "se_over_root_r":
                row = control_row(
                    property_name=study.CALIBRATION,
                    cell=cell,
                    replicate=replicate,
                    n=group.n,
                    requested=group.replicates,
                    truth=truth,
                    estimate=estimate,
                    standard_error=float(values[term]),
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
                    estimate=float(values[term]),
                    standard_error=1.0,
                    critical=study.CRITICAL,
                )
            rows.append({**row, **_extras(group, fitted, name)})
    return rows


def _replicate(label: str, replicate: int, drawn: Any) -> list[dict[str, Any]]:
    """Every property row of one replication of one group."""
    group = study.GROUP_BY_LABEL[label]
    frame, truths = drawn
    fitted = study.fit_group(group, frame, random_state=replicate)
    rows = _calibration_rows(group, replicate, truths, fitted)
    for term in group.bias_rows:
        name = f"{group.stem}[{term}]"
        row = replicate_row(
            property_name=study.BIAS,
            cell=f"{group.label}_{study.term_label(term)}",
            role="positive",
            replicate=replicate,
            n=group.n,
            requested=group.replicates,
            truth=float(truths[name]),
            estimate=fitted["estimates"][name],
            alpha=STUDY.margins.alpha,
        )
        rows.append({**row, **_extras(group, fitted, name)})
    rows += _control_rows(group, replicate, truths, fitted)
    if group.law == "hetero_trial":
        family, cell = (
            (study.TYPE_I, "heterogeneity__rep3_sharp_null")
            if group.slope == 0.0
            else (study.POWER, "heterogeneity__rep3_slope_0.10")
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
                **dict.fromkeys(EXTRA_COLUMNS, np.nan),
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


def generate_property_rows(
    *, n_jobs: int = STUDY_JOBS, budget: int | None = None, groups: tuple[str, ...] | None = None
) -> pd.DataFrame:
    """Fit every property replication; ``budget`` caps each group and ``groups`` picks some.

    A run of every group must fill exactly the declared cells; a smoke over ``groups`` checks
    only that each row it wrote is declared.
    """
    keys = [key for key in payloads(budget) if groups is None or key[0] in groups]
    outcomes = fit_properties(
        keys,
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
    if found - declared or (groups is None and found != declared):
        raise RuntimeError(f"cells {sorted(found ^ declared)} differ from the declaration")
    return rows.loc[:, [*REPLICATE_COLUMNS, *EXTRA_COLUMNS]]


def declared_readings(summary: pd.DataFrame, rows: pd.DataFrame) -> None:
    """Add the two declared readings' columns to ``summary`` in place (plan 5.0).

    ``se_ratio_without_displacement`` is the mean ``se_without_displacement`` over the
    empirical SD of the estimates, on every repeated row; ``bootstrap_failed_fraction`` is the
    mean failed-replicate share on every bootstrap row, and ``bootstrap_conditional`` marks a
    share above :data:`BOOTSTRAP_CONDITIONAL_SHARE`.
    """
    for column in SUMMARY_COLUMNS:
        summary[column] = np.nan
    for index, row in summary.iterrows():
        cell_rows = rows.loc[(rows["property"] == row["property"]) & (rows["cell"] == row["cell"])]
        without = cell_rows["se_without_displacement"].astype(float)
        if without.notna().all() and len(without) > 1:
            spread = float(cell_rows["estimate"].astype(float).std(ddof=1))
            summary.loc[index, "se_ratio_without_displacement"] = float(without.mean()) / spread
        requested = cell_rows["bootstrap_requested"].astype(float)
        if (requested > 0).all() and len(requested):
            share = float((cell_rows["bootstrap_failed"].astype(float) / requested).mean())
            summary.loc[index, "bootstrap_failed_fraction"] = share
            summary.loc[index, "bootstrap_conditional"] = float(share > BOOTSTRAP_CONDITIONAL_SHARE)


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The shared verdicts, the calibration kinds, the bias rows and the declared readings."""
    summary, rates = apply_shared_verdicts(
        rows.loc[:, list(REPLICATE_COLUMNS)], STUDY, rate_labels=()
    )
    calibration_verdicts(summary, margins=STUDY.margins, efficiency_band=None)
    bias_role_verdicts(summary, families=BIAS_FAMILIES)
    declared_readings(summary, rows)
    return finish(summary, rates)


#: The failure accounting's declared rules.  The calibration, type-I and power families read
#: the shared rules (:data:`~tests.studies.evidence.property_verdicts.FAMILY_RULES`).
PROPERTY_RULES = (bias_role_rule(BIAS_FAMILIES),)

__all__ = [
    "FAILURE_BOUND",
    "PROPERTY_RULES",
    "PROPERTY_WITNESS",
    "declared_readings",
    "generate_property_rows",
    "summarize_properties",
]
