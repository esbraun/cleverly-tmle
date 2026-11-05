"""The property families of the ``ep-crr`` study, and their verdicts.

Every group of :data:`tests.studies.ep_crr.GROUPS` is one draw per replication:

- ``oracle_comparison``: one row per replicate and estimator, the paired log ISE ratio of the
  log CRR against the oracle.  The EP row at 4,000 is a positive gate (upper bound below
  ``log 1.5``, at the intercept learner too), the intercept-only T-learner a control (lower
  bound above ``log 1.5``); the other rows are diagnostic.  Each law adds a growth row per
  gated estimator (upper bound of the 1k-to-4k change below ``log 1.25``);
- ``oracle_hessian``: 1 when the oracle's Hessian is positive definite at its solution, else
  0 (diagnostic).  A replicate with a 0 is excluded from every comparison of its group, and a
  gated row is red when its group excluded more than ``floor(0.01 R)``;
- ``nuisance_error``: the L2 errors of ``mu`` and ``pi`` and their product (diagnostic).
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies import conditional_projection as projection
from tests.studies import ep_crr as study
from tests.studies.ep_cate_properties import ORACLE_COLUMNS, _oracle_read, comparison_verdicts
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

__all__ = [
    "FAILURE_BOUND",
    "ORACLE_COLUMNS",
    "PROPERTY_RULES",
    "PROPERTY_WITNESS",
    "generate_property_rows",
    "summarize_properties",
]


def _draw(label: str, replicate: int) -> Any:
    group = study.GROUP_BY_LABEL[label]
    return study.draw_from_seed(label, group.n, stream_seed(STUDY, STREAM, label, replicate))


def _row(family: str, cell: str, role: str, group: Any, replicate: int, value: float) -> dict:
    return control_row(
        property_name=family,
        cell=cell,
        replicate=replicate,
        n=group.n,
        requested=group.replicates,
        truth=0.0,
        estimate=float(value),
        standard_error=1.0,
        critical=projection.CRITICAL,
        role=role,
    )


def oracle_values(group: Any, fitted: dict[str, Any]) -> dict[str, float]:
    """Each ``oracle_comparison`` cell's log ISE ratio in one replicate."""
    ensemble = fitted["ensemble"]
    out = {f"{group.label}__ep": math.log(ensemble["ep"] / ensemble["oracle"])}
    if group.kind == "cv":
        return out
    out[f"{group.label}__t_learner"] = math.log(ensemble["t_learner"] / ensemble["oracle"])
    if "misspecified" in fitted:
        wrong = fitted["misspecified"]
        out[f"{group.label}__ep_misspecified"] = math.log(wrong["ep"] / wrong["oracle"])
        out[f"{group.label}__t_misspecified"] = math.log(wrong["t_learner"] / wrong["oracle"])
    return out


def _replicate(label: str, replicate: int, drawn: Any) -> list[dict[str, Any]]:
    group = study.GROUP_BY_LABEL[label]
    frame, _ = drawn
    law = study.LAW_BY_LABEL[group.law]
    if group.kind == "cv":
        fitted = study.fit_cv_replicate(law, frame, replicate)
    else:
        fitted = study.fit_replicate(law, group.n, frame, replicate)
    roles = study.oracle_cells(group)
    rows = [
        _row(study.ORACLE, cell, roles[cell], group, replicate, value)
        for cell, value in oracle_values(group, fitted).items()
    ]
    for learner in study.learners_of(group):
        errors = fitted[learner]
        rows.append(
            _row(
                study.HESSIAN,
                f"{group.label}__{learner}_oracle_pd",
                property_verdicts.DIAGNOSTIC_ROLE,
                group,
                replicate,
                errors["oracle_pd"],
            )
        )
        if group.kind != "gated":
            continue
        values = {
            "mu_l2": errors["mu_l2"],
            "pi_l2": errors["pi_l2"],
            "product": errors["mu_l2"] * errors["pi_l2"],
        }
        for name, value in values.items():
            rows.append(
                _row(
                    study.NUISANCE,
                    f"{group.label}__{learner}_{name}",
                    property_verdicts.DIAGNOSTIC_ROLE,
                    group,
                    replicate,
                    value,
                )
            )
    return rows


def _cells_of(key: tuple[str, int]) -> list[str]:
    cells = study.group_cells(study.GROUP_BY_LABEL[key[0]])
    return [f"{family}/{cell}" for family, names in cells.items() for cell in names]


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


def margin_of(cell: str) -> tuple[float, str]:
    if cell.endswith("__t_misspecified"):
        return study.RESCUE_MARGIN, "above"
    if cell.endswith("__ep_misspecified"):
        return study.RESCUE_MARGIN, "below"
    return study.THM5_MARGIN, "below"


def excluded_replicates(rows: pd.DataFrame) -> dict[str, set[int]]:
    """Group label -> the replicates whose oracle Hessian (any learner) was not PD."""
    hessian = rows.loc[(rows["property"] == study.HESSIAN) & (rows["estimate"] < 0.5)]
    out: dict[str, set[int]] = {}
    for cell, replicate in zip(hessian["cell"], hessian["replicate"], strict=True):
        out.setdefault(str(cell).split("__", 1)[0], set()).add(int(replicate))
    return out


def oracle_verdicts(summary: pd.DataFrame, rows: pd.DataFrame) -> list[dict[str, Any]]:
    """The comparisons without the non-PD replicates, and the growth rows."""
    allowed = {group.label: pf.allowed(FAILURE_BOUND, group.replicates) for group in study.GROUPS}
    return comparison_verdicts(
        summary,
        rows,
        margin_of=margin_of,
        growth=[
            (law.label, study.gated_n(law), estimator)
            for law in study.LAWS
            for estimator in ("ep",) + ("ep_misspecified",) * law.rescue
        ],
        excluded=excluded_replicates(rows),
        allowed=allowed,
    )


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    summary, rates = apply_shared_verdicts(rows, STUDY, rate_labels=())
    for family in (study.NUISANCE, study.HESSIAN):
        summary.loc[summary["property"] == family, "passed"] = True
    growth = oracle_verdicts(summary, rows)
    return finish(summary, [*rates, *growth])


def growth_sources(row: pd.Series, rows: pd.DataFrame) -> list[tuple[str, str]]:
    cell = str(row["cell"])
    if not cell.endswith("_growth"):
        return [(str(row["property"]), cell)]
    label, estimator = cell.removesuffix("_growth").split("__")
    law = study.LAW_BY_LABEL[label]
    return [
        (study.ORACLE, f"{study.group_label(label, n)}__{estimator}") for n in study.gated_n(law)
    ]


def _oracle_rule() -> pf.FamilyRule:
    # Not accountable, as in ep-cate: a lost replicate of an ISE comparison has no candidate
    # value.  The oracle's non-PD Hessians are not lost replicates: they are recorded rows,
    # excluded and bounded by the verdict itself.
    return pf.FamilyRule(
        families=frozenset({study.ORACLE}),
        shared="oracle_verdicts",
        clauses=(
            pf.unaccountable_clause(
                "oracle comparison bound",
                statistic="one-sided mean bound",
                read=_oracle_read,
                applies=pf.roles("positive", "control"),
            ),
        ),
        sources=growth_sources,
    )


PROPERTY_RULES = (_oracle_rule(),)
