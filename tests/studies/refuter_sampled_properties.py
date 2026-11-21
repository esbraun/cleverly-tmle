"""Size and power of sampled refutation by ``m`` and ``m/n`` (X106 part H), and their verdicts.

Every trial of a cell of :data:`tests.studies.refuter_sampled.CELLS` draws its own data from
``stream_seed(STUDY, "property_sample", cell, trial)`` and writes one row per reading. Four
gated families and one reported family:

- ``refuter_size``: the I2 cells' size readings; a cell passes at no more than 73 rejections of
  1,000, read through the exact 99% Clopper-Pearson interval with a ceiling between the upper
  endpoints at 73 and 74 (:func:`count_rule_agrees`).
- ``refuter_size_relative``: each sampled cell's size readings against its full partner at
  ``n = m``. The partner's rows are published in the same family under ``<cell>__partner``
  with the diagnostic role. A pair passes when the 90% Wald interval of the difference of
  rejection rates lies inside ``±0.05`` (the two one-sided tests at 0.05).
- ``refuter_power``: the I2 cells' strong defects; a cell passes at no fewer than 38 detections
  of 50, through the 99% interval with a floor between the lower endpoints at 37 and 38.
- ``refuter_mutation``: the subset ``rho`` read against the full fit's SE and unit count; a cell
  passes when that reading fails both the absolute count and the relative test against its
  partner, that is, when the study detects the rejected reference.
- ``refuter_reported`` (diagnostic, no verdict): the absolute sizes of the I1 and clustered
  cells, every weak defect, the strong defects outside I2, and the P cells.

:data:`~tests.studies.refuter_sampled.REPORTED_FILE` adds, per cell, the share of trials that
printed a verdict for each rule, the sampled baseline beside the full fit, and the share of
calls whose limit bound. :data:`~tests.studies.refuter_sampled.CELLS_FILE` records the one fit
configuration of each cell, as in ``refuter-calibration-point``.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from tests.parallel import STUDY_JOBS
from tests.studies import refuter_sampled as study
from tests.studies.evidence import property_failures as pf
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.inference import clopper_pearson
from tests.studies.evidence.properties import REPLICATE_COLUMNS
from tests.studies.evidence.property_verdicts import (
    DIAGNOSTIC_ROLE,
    apply_shared_verdicts,
    finish,
    reported_verdicts,
)
from tests.studies.evidence.replication import fit_properties, splat_draw, splat_drawn

STUDY = study.STUDY
CELLS = study.CELLS
CELLS_FILE = study.CELLS_FILE
TRIALS_FILE = study.TRIALS_FILE
REPORTED_FILE = study.REPORTED_FILE
REPORTED_FAMILY = "refuter_reported"
PARTNER = "__partner"

_LEVEL = STUDY.margins.confidence_level


def _midpoint(low: float, high: float) -> float:
    return 0.5 * (low + high)


#: Between the 99% Clopper-Pearson upper endpoints at 73 and 74 rejections of 1,000.
SIZE_CEILING = _midpoint(
    clopper_pearson(study.SIZE_MAX_REJECTIONS, study.SIZE_TRIALS, confidence_level=_LEVEL).high,
    clopper_pearson(study.SIZE_MAX_REJECTIONS + 1, study.SIZE_TRIALS, confidence_level=_LEVEL).high,
)
#: Between the 99% Clopper-Pearson lower endpoints at 37 and 38 detections of 50.
POWER_FLOOR = _midpoint(
    clopper_pearson(
        study.POWER_MIN_DETECTIONS - 1, study.POWER_TRIALS, confidence_level=_LEVEL
    ).low,
    clopper_pearson(study.POWER_MIN_DETECTIONS, study.POWER_TRIALS, confidence_level=_LEVEL).low,
)


def count_rule_agrees() -> bool:
    """Whether the interval thresholds reproduce the count rules at every count."""
    size = all(
        (clopper_pearson(k, study.SIZE_TRIALS, confidence_level=_LEVEL).high <= SIZE_CEILING)
        == (k <= study.SIZE_MAX_REJECTIONS)
        for k in range(study.SIZE_TRIALS + 1)
    )
    power = all(
        (clopper_pearson(k, study.POWER_TRIALS, confidence_level=_LEVEL).low >= POWER_FLOOR)
        == (k >= study.POWER_MIN_DETECTIONS)
        for k in range(study.POWER_TRIALS + 1)
    )
    return size and power


def size_passes(rejections: int, trials: int) -> bool:
    """The absolute size gate on a count."""
    return clopper_pearson(rejections, trials, confidence_level=_LEVEL).high <= SIZE_CEILING


def relative_interval(sampled: np.ndarray, partner: np.ndarray) -> tuple[float, float, float]:
    """The difference of rejection rates and its two-sided 90% Wald interval.

    The two cells' trials are independent, so the variance is the sum of the two binomial
    variances. Each one-sided test at :data:`~tests.studies.refuter_sampled.RELATIVE_LEVEL`
    rejects its side of the margin exactly when the interval's endpoint clears it.
    """
    a = np.asarray(sampled, dtype=float)
    b = np.asarray(partner, dtype=float)
    p, q = float(a.mean()), float(b.mean())
    se = math.sqrt(p * (1.0 - p) / a.size + q * (1.0 - q) / b.size)
    z = float(stats.norm.ppf(1.0 - study.RELATIVE_LEVEL))
    return p - q, p - q - z * se, p - q + z * se


def relative_detects(sampled: np.ndarray, partner: np.ndarray) -> bool:
    """The mutation's relative leg: the interval lies wholly outside ``±`` the margin."""
    _, low, high = relative_interval(sampled, partner)
    return bool(low > study.RELATIVE_MARGIN or high < -study.RELATIVE_MARGIN)


def relative_passes(sampled: np.ndarray, partner: np.ndarray) -> bool:
    """The relative size gate: the interval lies inside ``±`` the margin."""
    _, low, high = relative_interval(sampled, partner)
    return bool(low > -study.RELATIVE_MARGIN and high < study.RELATIVE_MARGIN)


# ---- the rows -----------------------------------------------------------------------------

#: The truth column of a row: the statistic's value under the rule's reference law.
_REFERENCE = {"placebo": 0.0, "subset": 1.0, "negative_control": 0.0, "control": 0.0}


def _reference(reading: str) -> float:
    for prefix, value in _REFERENCE.items():
        if reading.startswith(prefix):
            return value
    raise KeyError(reading)


def _row(
    family: str,
    cell: str,
    role: str,
    trial: int,
    requested: int,
    reading: Any,
    n: int,
    extra: dict[str, Any],
) -> dict[str, Any]:
    name = cell.split("__")[1]
    reference = _reference(name)
    statistic = reading.statistic if math.isfinite(reading.statistic) else reference
    return {
        "property": family,
        "cell": cell,
        "role": role,
        "replicate": trial,
        "n": n,
        "requested_replicates": requested,
        "failed_replicates": 0,
        "truth": reference,
        "estimate": statistic,
        "std_error": 1.0,
        "covered": int(not reading.rejected),
        "rejected": int(reading.rejected),
        **extra,
        "pvalue": reading.pvalue,
        "verdict": "" if reading.verdict is None else str(bool(reading.verdict)),
    }


def _partnered(name: str) -> tuple[str, ...]:
    """The sampled cells read against full cell ``name``."""
    return tuple(other for other in study.RELATIVE_CELLS if CELLS[other].partner == name)


def cells_of(name: str, trial: int) -> list[tuple[str, str]]:
    """Every ``(family, cell)`` one trial of cell ``name`` writes."""
    cell = CELLS[name]
    out: list[tuple[str, str]] = []
    power = trial < study.POWER_TRIALS
    for reading in study.READINGS:
        if trial >= cell.trials:
            continue
        if name in study.GATED_CELLS:
            out.append((study.SIZE_FAMILY, f"{name}__{reading}"))
        else:
            out.append((REPORTED_FAMILY, f"{name}__{reading}"))
        if name in study.RELATIVE_CELLS:
            out.append((study.RELATIVE_FAMILY, f"{name}__{reading}"))
        for other in _partnered(name):
            out.append((study.RELATIVE_FAMILY, f"{other}__{reading}{PARTNER}"))
    if name in study.MUTATION_CELLS:
        out.append((study.MUTATION_FAMILY, f"{name}__subset_full_reference"))
    if power:
        for defect in study.STRONG_DEFECTS:
            family = study.POWER_FAMILY if name in study.GATED_CELLS else REPORTED_FAMILY
            out.append((family, f"{name}__{defect}"))
        out += [(REPORTED_FAMILY, f"{name}__{defect}") for defect in study.WEAK_DEFECTS]
    return out


def _role(family: str, cell: str) -> str:
    if family == REPORTED_FAMILY or cell.endswith(PARTNER):
        return DIAGNOSTIC_ROLE
    if family in (study.POWER_FAMILY, study.MUTATION_FAMILY):
        return "control"
    return "positive"


def _requested(name: str, cell: str) -> int:
    reading = cell.split("__")[1]
    if reading in (*study.STRONG_DEFECTS, *study.WEAK_DEFECTS):
        return study.POWER_TRIALS
    return CELLS[name].trials


#: The streams a trial can draw from: the declared run reads ``property``, the pilot ``pilot``
#: and a smoke ``smoke``, so no declared trial reuses a pilot or smoke draw.
STREAMS = ("property", "pilot", "smoke")


def _draw(name: str, trial: int, stream: str) -> Any:
    return study.draw_law(CELLS[name], study.trial_seed(f"{stream}_sample", name, trial))


def trial_rows(
    name: str, trial: int, frame: Any, *, stream: str = "property"
) -> list[dict[str, Any]]:
    """Every row one trial of cell ``name`` writes."""
    fit_seed = study.trial_seed(f"{stream}_fit", name, trial)
    result = study.trial_readings(
        name, frame, trial, power=trial < study.POWER_TRIALS, fit_seed=fit_seed
    )
    extra = {
        "configuration": json.dumps(result.configuration, sort_keys=True),
        "stream": stream,
        **result.reported,
    }
    rows = []
    for family, cell in cells_of(name, trial):
        source = cell.removesuffix(PARTNER).split("__")[1]
        owner = cell.split("__")[0]
        requested = _requested(owner, cell.removesuffix(PARTNER))
        if cell.endswith(PARTNER):
            requested = CELLS[name].trials
        rows.append(
            _row(
                family,
                cell,
                _role(family, cell),
                trial,
                requested,
                result.readings[source],
                CELLS[name].rows,
                extra,
            )
        )
    return rows


def _replicate(name: str, trial: int, stream: str, drawn: Any) -> list[dict[str, Any]]:
    return trial_rows(name, trial, drawn, stream=stream)


def _cells_of(key: tuple[str, int, str]) -> list[str]:
    return [f"{family}/{cell}" for family, cell in cells_of(key[0], key[1])]


def _cell_of(key: tuple[str, int, str]) -> str:
    family, cell = cells_of(key[0], key[1])[0]
    return f"{family}/{cell}"


def _replicate_of(key: tuple[str, int, str]) -> int:
    return key[1]


#: The share of a cell's requested trials that the accounting lets a run lose.
FAILURE_BOUND = 0.01

#: The draw-level statistic the accounting reports for each lost trial.
PROPERTY_WITNESS = property_verdicts.SAMPLE_WITNESS


def payloads(
    cells: tuple[str, ...] = tuple(CELLS), budget: int | None = None, stream: str = "property"
) -> list[tuple[str, int, str]]:
    """Every property key, cell-major; ``budget`` caps each cell's trials."""
    return [
        (name, trial, stream)
        for name in cells
        for trial in range(
            CELLS[name].trials if budget is None else min(budget, CELLS[name].trials)
        )
    ]


def written_cells() -> set[tuple[str, str]]:
    """Every ``(family, cell)`` a full run writes."""
    return {pair for key in payloads() for pair in cells_of(key[0], key[1])}


# ---- the artifacts ------------------------------------------------------------------------


def cell_configurations(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per cell: the one fit configuration its trials fitted.

    Raises
    ------
    RuntimeError
        If a cell's trials fitted more than one configuration.
    """
    own = rows.loc[~rows["cell"].str.endswith(PARTNER)]
    names = own["cell"].str.split("__", n=1).str[0]
    records = []
    for name in [name for name in CELLS if (names == name).any()]:
        found = set(own.loc[names == name, "configuration"])
        fits = [json.loads(text) for text in found]
        for fit in fits:
            fit.pop("min_event_count", None)
        axes = {json.dumps(fit, sort_keys=True) for fit in fits}
        if len(axes) != 1:
            raise RuntimeError(f"cell {name} fitted {len(axes)} configurations, expected one")
        records.append({"cell": name, **json.loads(axes.pop())})
    return pd.DataFrame(records)


def reported_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Per cell: the printed-verdict share of each rule, and the baseline beside the full fit."""
    size = rows.loc[
        rows["cell"].str.split("__").str[1].isin(study.READINGS)
        & ~rows["cell"].str.endswith(PARTNER)
    ]
    size = size.drop_duplicates(["cell", "replicate"])
    records = []
    for name, cell in CELLS.items():
        own = size.loc[size["cell"].str.split("__").str[0] == name]
        if own.empty:
            continue
        record: dict[str, Any] = {"cell": name, "n_units": cell.n, "m": cell.m or cell.n}
        for reading in study.READINGS:
            part = own.loc[own["cell"] == f"{name}__{reading}"]
            record[f"{reading}_rejection_rate"] = float(part["rejected"].mean())
            record[f"{reading}_verdict_share"] = float((part["verdict"] != "").mean())
        first = own.drop_duplicates("replicate")
        record["sample_applied_share"] = float(first["sample_applied"].astype(bool).mean())
        record["baseline_minus_full_sd"] = float(
            (first["baseline_estimate"] - first["full_estimate"]).std(ddof=1)
        )
        record["baseline_se_mean"] = float(first["baseline_std_error"].mean())
        record["full_se_mean"] = float(first["full_std_error"].mean())
        records.append(record)
    table = pd.DataFrame(records)
    table["verdict_scope"] = [
        "; ".join(scope_problems(record)) or "as declared" for record in table.to_dict("records")
    ]
    return table


def trials_table(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per cell and trial: every reading, the baseline beside the full fit, the seeds.

    Each reading ``r`` contributes ``r_statistic``, ``r_pvalue``, ``r_rejected`` and
    ``r_verdict`` (empty when the call printed none).
    """
    own = rows.loc[~rows["cell"].str.endswith(PARTNER)].copy()
    own["name"] = own["cell"].str.split("__").str[0]
    own["reading"] = own["cell"].str.split("__").str[1]
    own = own.drop_duplicates(["name", "reading", "replicate"])
    keys = ["name", "replicate"]
    wide = own.pivot(index=keys, columns="reading", values=["estimate", "pvalue", "rejected"])
    verdicts = own.pivot(index=keys, columns="reading", values="verdict")
    table = pd.DataFrame(index=wide.index)
    for reading in sorted(own["reading"].unique()):
        table[f"{reading}_statistic"] = wide[("estimate", reading)]
        table[f"{reading}_pvalue"] = wide[("pvalue", reading)]
        table[f"{reading}_rejected"] = wide[("rejected", reading)].astype("Int64")
        table[f"{reading}_verdict"] = verdicts[reading]
    extra = own.drop_duplicates(keys).set_index(keys)
    for column in (
        "full_estimate",
        "full_std_error",
        "baseline_estimate",
        "baseline_std_error",
        "sample_applied",
    ):
        table[column] = extra[column]
    table = table.reset_index().rename(columns={"name": "cell", "replicate": "trial"})
    streams = rows["stream"].iloc[0] if "stream" in rows.columns else "property"
    table["sample_seed"] = [
        study.trial_seed(f"{streams}_sample", c, int(t))
        for c, t in zip(table["cell"], table["trial"], strict=True)
    ]
    table["fit_seed"] = [
        study.trial_seed(f"{streams}_fit", c, int(t))
        for c, t in zip(table["cell"], table["trial"], strict=True)
    ]
    order = {name: k for k, name in enumerate(CELLS)}
    table["_order"] = table["cell"].map(order)
    return table.sort_values(["_order", "trial"]).drop(columns="_order").reset_index(drop=True)


def scope_problems(record: dict[str, Any]) -> list[str]:
    """How one cell's printed verdicts break the declared scope, from its reported row.

    Every I2 trial prints a verdict for each rule; every I1 and clustered trial prints none.
    The run records a break in ``reported.csv`` rather than stopping, and the registry test
    reads it from ``trials.csv``.
    """
    name = record["cell"]
    problems = []
    for reading in study.READINGS:
        share = record[f"{reading}_verdict_share"]
        if name.startswith("I2") and share != 1.0:
            problems.append(f"{reading} printed a verdict on {share:.3f} of trials")
        if name.startswith(("I1", "C-")) and share != 0.0:
            problems.append(f"{reading} printed a verdict on {share:.3f} of trials")
    return problems


def generate_property_artifacts(
    *,
    n_jobs: int = STUDY_JOBS,
    budget: int | None = None,
    cells: tuple[str, ...] = tuple(CELLS),
    stream: str = "property",
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Fit every trial and return the rows and the two committed tables.

    Parameters
    ----------
    n_jobs : int
        Worker processes.
    budget : int or None
        Caps each cell's trials (a smoke or pilot); ``None`` is the declared run.
    cells : tuple of str
        The cells to run; a declared run runs all of them.

    Returns
    -------
    tuple
        The property rows, and :data:`CELLS_FILE` and :data:`REPORTED_FILE` by file name.
    """
    outcomes = fit_properties(
        payloads(cells, budget, stream),
        splat_drawn(_replicate),
        cells_of=_cells_of,
        draw=splat_draw(_draw),
        cell_of=_cell_of,
        replicate_of=_replicate_of,
        n_jobs=n_jobs,
        seed_of=lambda key: study.trial_seed(f"{key[2]}_sample", key[0], key[1]),
    )
    rows = pd.DataFrame([row for records in outcomes for row in records])
    if budget is not None:
        rows["requested_replicates"] = rows["requested_replicates"].clip(upper=budget)
    found = set(zip(rows["property"], rows["cell"], strict=True))
    if budget is None and cells == tuple(CELLS) and found != written_cells():
        raise RuntimeError(f"cells {sorted(found ^ written_cells())} differ from the declaration")
    tables = {
        CELLS_FILE: cell_configurations(rows),
        REPORTED_FILE: reported_table(rows),
        TRIALS_FILE: trials_table(rows),
    }
    return rows.loc[:, list(REPLICATE_COLUMNS)], tables


def generate_property_rows(*, n_jobs: int = STUDY_JOBS, budget: int | None = None) -> pd.DataFrame:
    """Fit every trial and return one row per cell and trial."""
    rows, _ = generate_property_artifacts(n_jobs=n_jobs, budget=budget)
    return rows


# ---- the verdicts -------------------------------------------------------------------------


def _rejections(rows: pd.DataFrame, family: str, cell: str) -> np.ndarray:
    part = rows.loc[(rows["property"] == family) & (rows["cell"] == cell)]
    return part.sort_values("replicate")["rejected"].to_numpy(dtype=float)


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """Each gated cell's rule, and the reported rows with no verdict."""
    rows = rows.loc[:, list(REPLICATE_COLUMNS)]
    summary, rates = apply_shared_verdicts(rows, STUDY, rate_labels=())
    summary["passed"] = summary["passed"].astype(object)
    size = summary["property"] == study.SIZE_FAMILY
    summary.loc[size, "passed"] = summary.loc[size, "rejection_ci_upper"] <= SIZE_CEILING
    power = summary["property"] == study.POWER_FAMILY
    summary.loc[power, "passed"] = summary.loc[power, "rejection_ci_lower"] >= POWER_FLOOR
    for column in ("relative_difference", "relative_ci_lower", "relative_ci_upper"):
        summary[column] = np.nan
    relative = summary["property"] == study.RELATIVE_FAMILY
    for index in summary.index[relative]:
        cell = str(summary.loc[index, "cell"])
        base = cell.removesuffix(PARTNER)
        own = _rejections(rows, study.RELATIVE_FAMILY, base)
        partner = _rejections(rows, study.RELATIVE_FAMILY, base + PARTNER)
        difference, low, high = relative_interval(own, partner)
        summary.loc[index, ["relative_difference", "relative_ci_lower", "relative_ci_upper"]] = (
            difference,
            low,
            high,
        )
        # A partner row is the reference arm: it publishes its interval and states no verdict.
        summary.loc[index, "passed"] = cell.endswith(PARTNER) or relative_passes(own, partner)
    mutation = summary["property"] == study.MUTATION_FAMILY
    for index in summary.index[mutation]:
        cell = str(summary.loc[index, "cell"])
        name = cell.split("__")[0]
        mutated = _rejections(rows, study.MUTATION_FAMILY, cell)
        partner = _rejections(rows, study.RELATIVE_FAMILY, f"{name}__subset{PARTNER}")
        difference, low, high = relative_interval(mutated, partner)
        summary.loc[index, ["relative_difference", "relative_ci_lower", "relative_ci_upper"]] = (
            difference,
            low,
            high,
        )
        detected = not size_passes(int(mutated.sum()), mutated.size) and relative_detects(
            mutated, partner
        )
        summary.loc[index, "passed"] = detected
    reported_verdicts(summary, families=(REPORTED_FAMILY,))
    return finish(summary, rates)


def _filled(values: np.ndarray, lost: int, rejected: int) -> np.ndarray:
    """``values`` with ``lost`` trials filled in, ``rejected`` of them as rejections."""
    return np.r_[values, np.ones(rejected), np.zeros(lost - rejected)]


def _relative_clause(name: str, *, detect: bool, partner: Any, applies: pf.Applies) -> pf.Clause:
    """S3d: the difference of two cells' rejection rates against the margin.

    The two cells' trials are independent indicators, so every split of each cell's lost trials
    into rejections and non-rejections is a candidate, and the set needs no range assumption.
    ``detect`` reads the mutation's leg (the interval wholly outside the margin) rather than
    the equivalence (wholly inside). ``partner(row)`` names the ``(family, cell)`` read against.
    """
    margin = study.RELATIVE_MARGIN

    def judge(low: float, high: float) -> pf.Judged:
        if detect:
            return pf.Judged(max(low - margin, -margin - high), low > margin or high < -margin)
        return pf.Judged(min(low + margin, margin - high), low > -margin and high < margin)

    def read(ctx: pf.Context, row: pd.Series) -> pf.Judged:
        own = _rejections(ctx.rows, str(row["property"]), str(row["cell"]))
        other = _rejections(ctx.rows, *partner(row))
        _, low, high = relative_interval(own, other)
        return judge(low, high)

    def reads(ctx: pf.Context, row: pd.Series) -> list[tuple[str, str]]:
        return [(str(row["property"]), str(row["cell"])), partner(row)]

    def candidates(ctx: pf.Context, row: pd.Series) -> list[tuple[str, pf.Judged]]:
        family, cell = str(row["property"]), str(row["cell"])
        own = _rejections(ctx.rows, family, cell)
        other = _rejections(ctx.rows, *partner(row))
        k1, k2 = ctx.lost_in(family, cell), ctx.lost_in(*partner(row))
        out = []
        for j1 in range(k1 + 1):
            for j2 in range(k2 + 1):
                _, low, high = relative_interval(_filled(own, k1, j1), _filled(other, k2, j2))
                out.append((f"j={j1} rejected, partner j={j2} rejected", judge(low, high)))
        return out

    return pf.Clause(name, "S3d", applies, read, candidates, imputation=pf.SUPPORT, reads=reads)


def _rules() -> tuple[pf.FamilyRule, ...]:
    def own_partner(row: pd.Series) -> tuple[str, str]:
        return study.RELATIVE_FAMILY, str(row["cell"]) + PARTNER

    def subset_partner(row: pd.Series) -> tuple[str, str]:
        return study.RELATIVE_FAMILY, f"{str(row['cell']).split('__')[0]}__subset{PARTNER}"

    return (
        pf.FamilyRule(
            families=frozenset({study.SIZE_FAMILY}),
            shared="summarize_properties",
            clauses=(
                pf.proportion_clause(
                    "rejections at most 73 of 1,000",
                    statistic="S3",
                    judge=pf.ceiling_judge(lambda ctx: SIZE_CEILING),
                    applies=pf.every,
                ),
            ),
        ),
        pf.FamilyRule(
            families=frozenset({study.POWER_FAMILY}),
            shared="summarize_properties",
            clauses=(
                pf.proportion_clause(
                    "detections at least 38 of 50",
                    statistic="S3",
                    judge=pf.floor_judge(lambda ctx: POWER_FLOOR),
                    applies=pf.every,
                ),
            ),
        ),
        pf.FamilyRule(
            families=frozenset({study.RELATIVE_FAMILY}),
            shared="summarize_properties",
            clauses=(
                _relative_clause(
                    "difference of rejection rates inside the margin",
                    detect=False,
                    partner=own_partner,
                    applies=pf.roles("positive"),
                ),
            ),
        ),
        pf.FamilyRule(
            families=frozenset({study.MUTATION_FAMILY}),
            shared="summarize_properties",
            clauses=(
                pf.proportion_clause(
                    "absolute count fails",
                    statistic="S3",
                    judge=lambda ctx, row, iv: pf.Judged(
                        iv.high - SIZE_CEILING, iv.high > SIZE_CEILING
                    ),
                    applies=pf.every,
                ),
                _relative_clause(
                    "difference of rejection rates outside the margin",
                    detect=True,
                    partner=subset_partner,
                    applies=pf.every,
                ),
            ),
        ),
    )


#: The failure accounting's declared rules, one per gated family. The relative and mutation
#: legs enumerate every split of both cells' lost trials (S3d).
PROPERTY_RULES = _rules()
