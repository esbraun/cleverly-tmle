"""The property families of the ``ep-cate`` study, and their verdicts.

Every group of :data:`tests.studies.ep_cate.GROUPS` is one draw per replication, and its fits
fill the cells :func:`~tests.studies.ep_cate.group_cells` names:

- ``oracle_comparison``: one row per replicate and estimator, the paired log ISE ratio
  ``log ISE(estimator) - log ISE(oracle DR-learner)``.  The EP row at the largest ``n`` is a
  positive gate (upper bound below ``log 1.5``; ``log 3`` at the misspecified learner), the
  T-learner at the misspecified learner a control (lower bound above ``log 3``), and every
  other row is published as a diagnostic.  Each law adds a ``growth`` summary row per gated
  estimator: the upper bound of the change in mean log ratio from the smallest to the
  largest ``n``, from an independent-samples interval, below ``log 1.25``;
- ``nuisance_error``: the L2 errors of ``mu`` and ``pi`` and their product (diagnostic);
- ``bias``: each ``ep_fixed_obs_4k`` coefficient's standardized bias inside +-0.25 (the
  shipped endpoint), and the paired EP-minus-TMLE difference as a diagnostic.

A row here is a number, not a fit with an interval, so it carries a unit standard error and
only its spread is read.  Every bound is a one-sided 99% Student bound
(:func:`tests.studies.ep_cate.upper_bound`).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from tests.parallel import STUDY_JOBS
from tests.studies import conditional_projection as projection
from tests.studies import ep_cate as study
from tests.studies.evidence import property_failures as pf
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.properties import REPLICATE_COLUMNS, control_row
from tests.studies.evidence.property_verdicts import apply_shared_verdicts, finish
from tests.studies.evidence.replication import fit_properties, splat_draw, splat_drawn
from tests.studies.evidence.seeds import stream_seed

STUDY = study.STUDY
PROPERTY_CELLS = study.PROPERTY_CELLS

#: The share of a cell's requested replications that the accounting lets a run lose.
FAILURE_BOUND = 0.01

PROPERTY_WITNESS = property_verdicts.SAMPLE_WITNESS

#: The stream every property draw comes from.
STREAM = "property_sample"

#: The columns the ``oracle_comparison`` verdicts publish.
ORACLE_COLUMNS = ("ratio_mean", "ratio_sd", "ratio_lower", "ratio_upper", "margin")


def _draw(label: str, replicate: int) -> Any:
    group = study.GROUP_BY_LABEL[label]
    return study.draw_from_seed(label, group.n, stream_seed(STUDY, STREAM, label, replicate))


def _row(
    family: str,
    cell: str,
    role: str,
    group: Any,
    replicate: int,
    value: float,
    *,
    truth: float = 0.0,
    requested: int | None = None,
) -> dict[str, Any]:
    return control_row(
        property_name=family,
        cell=cell,
        replicate=replicate,
        n=group.n,
        requested=group.replicates if requested is None else requested,
        truth=truth,
        estimate=float(value),
        standard_error=1.0,
        critical=projection.CRITICAL,
        role=role,
    )


def oracle_values(group: Any, fitted: dict[str, Any]) -> dict[str, float]:
    """Each ``oracle_comparison`` cell's log ISE ratio in one replicate."""
    out: dict[str, float] = {}
    if "ensemble" in fitted:
        ensemble = fitted["ensemble"]
        out[f"{group.label}__ep"] = math.log(ensemble["ep"] / ensemble["oracle"])
        if group.kind == "cv":
            return out
        out[f"{group.label}__t_learner"] = math.log(ensemble["t_learner"] / ensemble["oracle"])
        out[f"{group.label}__dr_learner"] = math.log(ensemble["dr_learner"] / ensemble["oracle"])
    if "misspecified" in fitted:
        wrong = fitted["misspecified"]
        out[f"{group.label}__ep_misspecified"] = math.log(wrong["ep"] / wrong["oracle"])
        out[f"{group.label}__t_misspecified"] = math.log(wrong["t_learner"] / wrong["oracle"])
    if "greedy" in fitted:
        oracle = fitted["ensemble"]["oracle"]
        out[f"{group.label}__ep_greedy_hgb"] = math.log(fitted["greedy"] / oracle)
    return out


def _replicate(label: str, replicate: int, drawn: Any) -> list[dict[str, Any]]:
    group = study.GROUP_BY_LABEL[label]
    frame, _ = drawn
    rows: list[dict[str, Any]] = []
    if group.kind == "fixed":
        fitted = study.fit_fixed(frame, replicate)
        truth = study.fixed_truth()
        for term, cell in zip(projection.TERMS, study.fixed_cells(), strict=True):
            rows.append(
                _row(
                    study.BIAS,
                    cell,
                    "positive",
                    group,
                    replicate,
                    fitted["ep"][term],
                    truth=truth[term],
                )
            )
            rows.append(
                _row(
                    study.DIAGNOSTIC,
                    f"{cell}__ep_minus_tmle",
                    property_verdicts.DIAGNOSTIC_ROLE,
                    group,
                    replicate,
                    fitted["ep"][term] - float(fitted["tmle"][term].psi),
                )
            )
        return rows
    law = study.LAW_BY_LABEL[group.law]
    if group.kind == "cv":
        fitted = study.fit_cv_replicate(law, frame, replicate)
    else:
        fitted = study.fit_replicate(law, group.n, frame, replicate)
    roles = study.oracle_cells(group)
    for cell, value in oracle_values(group, fitted).items():
        requested = study.DESCRIPTIVE_REPLICATES if cell.endswith("greedy_hgb") else None
        rows.append(
            _row(study.ORACLE, cell, roles[cell], group, replicate, value, requested=requested)
        )
    if group.kind == "gated":
        for learner in ("ensemble", "misspecified"):
            if learner not in fitted:
                continue
            errors = fitted[learner]
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
    group = study.GROUP_BY_LABEL[key[0]]
    cells = study.group_cells(group)
    out = [f"{family}/{cell}" for family, names in cells.items() for cell in names]
    if key[1] >= study.DESCRIPTIVE_REPLICATES:
        out = [cell for cell in out if not cell.endswith("greedy_hgb")]
    return out


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
    """The declared margin and side of an ``oracle_comparison`` cell's verdict."""
    if cell.endswith("__t_misspecified"):
        return study.RESCUE_MARGIN, "above"
    if cell.endswith("__ep_misspecified"):
        return study.RESCUE_MARGIN, "below"
    return study.THM5_MARGIN, "below"


def _bounds(values: np.ndarray) -> tuple[float, float, float, float]:
    r = values.size
    mean, sd = float(np.mean(values)), float(np.std(values, ddof=1))
    half = float(stats.t.ppf(study.CONFIDENCE, r - 1)) * sd / math.sqrt(r)
    return mean, sd, mean - half, mean + half


def oracle_verdicts(summary: pd.DataFrame, rows: pd.DataFrame) -> list[dict[str, Any]]:
    """Each ``oracle_comparison`` row's bounds and verdict, and the growth rows."""
    return comparison_verdicts(
        summary,
        rows,
        margin_of=margin_of,
        growth=[
            (law.label, study.gated_n(law), estimator)
            for law in study.LAWS
            for estimator in ("ep",) * law.ensemble + ("ep_misspecified",) * law.rescue
        ],
    )


#: The shared summary columns that an ``oracle_comparison`` row leaves blank.
MEANINGLESS_ON_RATIOS = (
    "truth",
    "bias",
    "bias_se",
    "bias_ci_lower",
    "bias_ci_upper",
    "bias_margin",
    "standardized_bias",
    "bias_equivalent",
    "bias_discriminated",
    "root_n_bias",
    "mean_std_error",
    "se_ratio",
    "coverage",
    "coverage_ci_lower",
    "coverage_ci_upper",
    "rejection_rate",
    "rejection_ci_lower",
    "rejection_ci_upper",
    "nominal_size",
)


def comparison_verdicts(
    summary: pd.DataFrame,
    rows: pd.DataFrame,
    *,
    margin_of: Any,
    growth: list[tuple[str, tuple[int, int], str]],
    excluded: dict[str, set[int]] | None = None,
    allowed: dict[str, int] | None = None,
    reported: frozenset[str] = frozenset(),
) -> list[dict[str, Any]]:
    """The bounds and verdicts of the ``oracle_comparison`` rows, and the growth rows.

    Parameters
    ----------
    summary : pandas.DataFrame
        The shared summary; its ``oracle_comparison`` rows gain the bound columns.
    rows : pandas.DataFrame
        The replication rows.
    margin_of : Callable
        A cell's declared margin and side.
    growth : list
        ``(law, (small_n, large_n), estimator)`` for each growth row.
    excluded : dict or None
        Group label -> replicates excluded from that group's comparisons (``ep-crr``'s
        non-positive-definite oracle Hessians).
    allowed : dict or None
        Group label -> the exclusions the failure bound allows; a gated row of a group with
        more is red.
    reported : frozenset of str
        The growth cells (``<law>__<estimator>_growth``) that are reported, not gated: their
        row is diagnostic and always passes.

    Returns
    -------
    list of dict
        The growth rows.
    """
    excluded = excluded or {}
    allowed = allowed or {}
    for column in ORACLE_COLUMNS:
        if column not in summary:
            summary[column] = np.nan
    # Each oracle_comparison row is a log ratio carried with a unit "standard error" and a
    # zero "truth", so the shared bias, coverage and rejection columns mean nothing there
    # (design review S3 of PR-b): they are blanked, and the ratio columns are the reading.
    oracle = summary["property"] == study.ORACLE
    for column in MEANINGLESS_ON_RATIOS:
        if column in summary:
            summary[column] = summary[column].astype(object)
            summary.loc[oracle, column] = np.nan

    def group_of(cell: str) -> str:
        return cell.split("__", 1)[0]

    values: dict[str, np.ndarray] = {}
    for cell, group in rows.loc[rows["property"] == study.ORACLE].groupby("cell"):
        drop = excluded.get(group_of(str(cell)), set())
        keep = ~group["replicate"].isin(drop)
        values[str(cell)] = group.loc[keep, "estimate"].to_numpy(dtype=float)
    family = summary["property"] == study.ORACLE
    for index in summary.index[family.to_numpy()]:
        cell = str(summary.loc[index, "cell"])
        mean, sd, low, high = _bounds(values[cell])
        margin, side = margin_of(cell)
        summary.loc[index, ["ratio_mean", "ratio_sd", "ratio_lower", "ratio_upper"]] = (
            mean,
            sd,
            low,
            high,
        )
        role = str(summary.loc[index, "role"])
        if role == property_verdicts.DIAGNOSTIC_ROLE:
            summary.loc[index, "passed"] = True
            continue
        label = group_of(cell)
        within = len(excluded.get(label, set())) <= allowed.get(label, 0)
        summary.loc[index, "margin"] = margin
        clears = bool(high < margin if side == "below" else low > margin)
        summary.loc[index, "passed"] = clears and within
    out: list[dict[str, Any]] = []
    for law, (small_n, large_n), estimator in growth:
        small = values[f"{study.group_label(law, small_n)}__{estimator}"]
        large = values[f"{study.group_label(law, large_n)}__{estimator}"]
        change = float(np.mean(large) - np.mean(small))
        spread = math.sqrt(np.var(small, ddof=1) / small.size + np.var(large, ddof=1) / large.size)
        upper = change + float(stats.t.ppf(study.CONFIDENCE, small.size + large.size - 2)) * spread
        within = all(
            len(excluded.get(study.group_label(law, n), set()))
            <= allowed.get(study.group_label(law, n), 0)
            for n in (small_n, large_n)
        )
        cell = f"{law}__{estimator}_growth"
        gated = cell not in reported
        row = dict.fromkeys(summary.columns, np.nan)
        row.update(
            {
                "property": study.ORACLE,
                "cell": cell,
                "role": "positive" if gated else property_verdicts.DIAGNOSTIC_ROLE,
                "n": large_n,
                "replicates": int(min(small.size, large.size)),
                "failed_replicates": 0,
                "ratio_mean": change,
                "ratio_sd": spread,
                "ratio_upper": upper,
                "margin": study.GROWTH_MARGIN if gated else np.nan,
                "passed": (bool(upper < study.GROWTH_MARGIN) and within) or not gated,
                "property_passed": None,
            }
        )
        out.append(row)
    return out


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The shared summary, the bias rows, the oracle comparisons and the growth rows."""
    summary, rates = apply_shared_verdicts(rows, STUDY, rate_labels=())
    for family in (study.NUISANCE, study.DIAGNOSTIC):
        summary.loc[summary["property"] == family, "passed"] = True
    bias = (summary["property"] == study.BIAS) & (summary["role"] == "positive")
    summary.loc[bias, "passed"] = summary.loc[bias, "bias_equivalent"]
    growth = oracle_verdicts(summary, rows)
    return finish(summary, [*rates, *growth])


def growth_sources(row: pd.Series, rows: pd.DataFrame) -> list[tuple[str, str]]:
    """The cells an ``oracle_comparison`` summary row reads: its own, or a growth row's two."""
    cell = str(row["cell"])
    if not cell.endswith("_growth"):
        return [(str(row["property"]), cell)]
    label, estimator = cell.removesuffix("_growth").split("__")
    law = study.LAW_BY_LABEL[label]
    return [
        (study.ORACLE, f"{study.group_label(label, n)}__{estimator}") for n in study.gated_n(law)
    ]


def _oracle_read(ctx: Any, row: pd.Series) -> Any:
    passed = bool(row["passed"])
    return pf.Judged(0.0 if passed else -1.0, passed)


def _oracle_rule() -> pf.FamilyRule:
    # Not accountable (rev 3 section 2): the gate is a one-sided bound on a mean log ratio,
    # and a lost replicate of an ISE comparison has no candidate value the study can defend,
    # so a row that lost one is red at any bound.
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


def _bias_rule() -> pf.FamilyRule:
    return pf.FamilyRule(
        families=frozenset({study.BIAS}),
        shared="bias_verdicts",
        clauses=(
            pf.bias_clause("bias equivalent", kind="equivalent", applies=pf.roles("positive")),
        ),
    )


PROPERTY_RULES = (_oracle_rule(), _bias_rule())

__all__ = [
    "FAILURE_BOUND",
    "PROPERTY_RULES",
    "PROPERTY_WITNESS",
    "generate_property_rows",
    "summarize_properties",
]
