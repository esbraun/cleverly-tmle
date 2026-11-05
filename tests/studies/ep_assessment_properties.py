"""The property families of the ``ep-assessment`` study, and their verdicts.

One row per replicate and reading (:data:`tests.studies.ep_assessment.CELLS`):

- ``calibration``: the out-of-fold slope (positive: the 99% interval of the mean inside
  ``[0.85, 1.15]``) and the stretched curve's slope (control: the 99% upper bound below
  0.85); the weak law's slope is diagnostic;
- ``stability``: ``relative_sd`` and the indicator that it exceeds the 0.5 split-sensitivity
  warning, at both laws (diagnostic: reported, not gated; plan amendment rev 2.4);
- ``risk_comparison``: the indicator that the reported EP-minus-constant DR risk difference
  has the exact value's sign (positive at the strong law, diagnostic at the weak), and both
  differences (diagnostic).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies import conditional_projection as projection
from tests.studies import ep_assessment as study
from tests.studies.ep_cate_properties import _oracle_read
from tests.studies.evidence import property_failures as pf
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.properties import REPLICATE_COLUMNS, control_row
from tests.studies.evidence.property_verdicts import apply_shared_verdicts, finish
from tests.studies.evidence.replication import fit_properties, splat_draw, splat_drawn
from tests.studies.evidence.seeds import stream_seed

STUDY = study.STUDY
PROPERTY_CELLS = study.PROPERTY_CELLS
FAILURE_BOUND = study.FAILURE_BOUND
PROPERTY_WITNESS = property_verdicts.SAMPLE_WITNESS
STREAM = "property_sample"

#: The columns the verdicts publish.
READING_COLUMNS = ("reading", "reading_lower", "reading_upper", "threshold")

ROLES = {
    (family, study.cell_name(group, suffix)): role for family, suffix, role, group, _ in study.CELLS
}
REQUESTED = {
    (family, study.cell_name(group, suffix)): requested
    for family, suffix, _, group, requested in study.CELLS
}

__all__ = [
    "FAILURE_BOUND",
    "PROPERTY_RULES",
    "PROPERTY_WITNESS",
    "generate_property_rows",
    "summarize_properties",
]


def _draw(label: str, replicate: int) -> Any:
    return study.draw_from_seed(label, study.N, stream_seed(STUDY, STREAM, label, replicate))


def _replicate(label: str, replicate: int, drawn: Any) -> list[dict[str, Any]]:
    group = study.GROUP_BY_LABEL[label]
    frame, _ = drawn
    fitted = study.fit_replicate(group, frame, replicate)
    rows = []
    for (family, cell), value in study.row_values(group, fitted).items():
        role = ROLES[(family, cell)]
        rows.append(
            control_row(
                property_name=family,
                cell=cell,
                replicate=replicate,
                n=study.N,
                requested=REQUESTED[(family, cell)],
                truth=0.0,
                estimate=value,
                standard_error=1.0,
                critical=projection.CRITICAL,
                role=property_verdicts.DIAGNOSTIC_ROLE if role == study.DIAGNOSTIC else role,
            )
        )
    return rows


def _cells_of(key: tuple[str, int]) -> list[str]:
    label, replicate = key
    return [
        f"{family}/{study.cell_name(group, suffix)}"
        for family, suffix, _, group, requested in study.CELLS
        if group == label and replicate < requested
    ]


def payloads(budget: int | None = None) -> list[tuple[str, int]]:
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
        cell_of=lambda key: f"group/{key[0]}",
        replicate_of=lambda key: key[1],
        n_jobs=n_jobs,
        seed_of=lambda key: stream_seed(STUDY, STREAM, key[0], key[1]),
    )
    rows = pd.DataFrame([row for records in outcomes for row in records])
    if budget is not None:
        rows["requested_replicates"] = rows["requested_replicates"].clip(upper=budget)
    declared = {(family, cell) for family, cells in PROPERTY_CELLS.items() for cell in cells}
    found = set(zip(rows["property"], rows["cell"], strict=True))
    if found != declared:
        raise RuntimeError(f"cells {sorted(found ^ declared)} differ from the declaration")
    return rows.loc[:, list(REPLICATE_COLUMNS)]


# ---------------------------------------------------------------- the verdicts


#: The cells whose reading is the share of replicates at 1: the rest are continuous.
SHARE_SUFFIXES = ("__warns", "__sign_agrees")


def verdict(family: str, cell: str, values: np.ndarray) -> dict[str, Any]:
    """The reading, its bounds, the threshold and the verdict of one gated cell."""
    if family == study.CALIBRATION:
        low, high = study.mean_interval(values)
        if cell.endswith("__stretched_slope"):
            threshold = study.CALIBRATION_BAND[0]
            passed = high < threshold
        else:
            threshold = float("nan")
            passed = study.CALIBRATION_BAND[0] < low and high < study.CALIBRATION_BAND[1]
        return {
            "reading": float(np.mean(values)),
            "reading_lower": low,
            "reading_upper": high,
            "threshold": threshold,
            "passed": bool(passed),
        }
    if not cell.endswith(SHARE_SUFFIXES):
        # A continuous diagnostic (relative_sd, risk_difference, exact_difference) reads its
        # mean, with no threshold (design review S4 of PR-b).
        low, high = study.mean_interval(values)
        return {
            "reading": float(np.mean(values)),
            "reading_lower": low,
            "reading_upper": high,
            "threshold": float("nan"),
            "passed": True,
        }
    successes = int(np.sum(values > 0.5))
    lower = study.share_lower(successes, values.size)
    return {
        "reading": successes / values.size,
        "reading_lower": lower,
        "reading_upper": float("nan"),
        "threshold": study.SHARE_FLOOR,
        "passed": bool(lower >= study.SHARE_FLOOR),
    }


def reading_verdicts(summary: pd.DataFrame, rows: pd.DataFrame) -> list[dict[str, Any]]:
    """Fill each row's reading; set the verdict of every positive and control row."""
    for column in READING_COLUMNS:
        if column not in summary:
            summary[column] = np.nan
    for index in summary.index:
        family, cell = str(summary.loc[index, "property"]), str(summary.loc[index, "cell"])
        if (family, cell) not in ROLES:
            continue
        values = rows.loc[
            (rows["property"] == family) & (rows["cell"] == cell), "estimate"
        ].to_numpy(dtype=float)
        result = verdict(family, cell, values)
        for column in READING_COLUMNS:
            summary.loc[index, column] = result[column]
        if ROLES[(family, cell)] == study.DIAGNOSTIC:
            summary.loc[index, "passed"] = True
        else:
            summary.loc[index, "passed"] = result["passed"]
    return []


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    summary, rates = apply_shared_verdicts(rows, STUDY, rate_labels=())
    extra = reading_verdicts(summary, rows)
    return finish(summary, [*rates, *extra])


def _rule() -> pf.FamilyRule:
    # Not accountable: a lost replicate of a refit-based reading has no candidate value.
    return pf.FamilyRule(
        families=frozenset({study.CALIBRATION, study.RISK}),
        shared="reading_verdicts",
        clauses=(
            pf.unaccountable_clause(
                "assessment reading",
                statistic="mean interval or Clopper-Pearson share bound",
                read=_oracle_read,
                applies=pf.roles("positive", "control"),
            ),
        ),
    )


PROPERTY_RULES = (_rule(),)
